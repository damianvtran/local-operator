/*
 * The notification sender that carries Local Operator's OWN identity.
 *
 * WHY A COMPILED BUNDLE AT ALL. macOS attributes a notification to the
 * process that posts it, not to the text it carries. `osascript -e 'display
 * notification'` therefore arrives as **Script Editor** — its icon, its name,
 * its notification settings — which is what the operator saw on screen. The
 * title string can say "Local Operator" and the banner still belongs to
 * somebody else.
 *
 * Three routes were measured before this one (see the PR thread):
 *   - An `osacompile` bundle has no CFBundleIdentifier at all.
 *   - A bundle that SHELLS OUT to `osascript` still delivers as Script
 *     Editor: the identity follows the posting process into the helper.
 *   - `UNUserNotificationCenter` aborts outside a signed, authorized bundle,
 *     and pyobjc is not a dependency of this project.
 *
 * A tiny Objective-C binary inside a bundle we own is the one route that
 * works with only the Command Line Tools, no new Python dependency, and no
 * code-signing story: the notification database records
 * `me.damiantran.localoperator`, and the banner shows our name and icon.
 *
 * WHY NSUserNotificationCenter, which is deprecated. Its replacement
 * (UserNotifications.framework) requires the bundle to be a real, signed
 * application that has been GRANTED authorization by the user, and it aborts
 * hard when it is not — unusable for a best-effort toast from a detached
 * background process. The deprecated API still delivers on current macOS and
 * degrades to nothing (not a crash) when it does not, which matches
 * `detached_notify`'s contract exactly. `-Wno-deprecated-declarations` in the
 * build is deliberate and this paragraph is its justification.
 *
 * CLICK-THROUGH. Activation is delivered to a DELEGATE, so the sender has to
 * stay alive to receive it — hence the bounded run loop. argv[3] is the shell
 * command to run when the user clicks; it is built by
 * `notify.resume_click_command`, which routes through
 * `broadcast.resume_argv`, so a click replays a transcript and idles rather
 * than resuming tool execution unattended. Absent argv[3] the banner is
 * simply not clickable.
 *
 * The run loop is BOUNDED (not infinite): an un-clicked notification must not
 * leave a process resident forever. macOS keeps the banner in Notification
 * Centre after we exit; only the click-through stops working.
 *
 * THAT LOSS IS NOT ALWAYS THE RIGHT ONE, AND THE BOUND IS THEREFORE AN ARGUMENT
 * (argv[5], seconds). A fixed 30 s was fine for a toast about a session the
 * user is being pulled back to NOW, and wrong for Aida's morning check-in: it
 * posts at 08:30, the banner sits in Notification Centre, and the user clicks
 * it at 10:00 — to a helper that exited an hour and a half earlier, so the
 * click did nothing and said nothing. A relaunch-on-click is NOT a fallback:
 * a click after exit relaunches this binary with no arguments, which takes the
 * usage branch below and exits (READ FROM THIS FILE, NOT MEASURED: the one
 * real-banner click probe that would have shown it on a live Notification
 * Centre was skipped by the operator's decision). So the helper has to still be
 * alive, and the caller says for how long. The cost is one idle few-MB process
 * per such banner for the length of its window; which callers ask for one, and
 * what limits how many, is stated at `notify.DURABLE_CLICK_WINDOW_S`.
 *
 * AT THE END OF THE WINDOW THE BANNER IS DELIBERATELY LEFT IN NOTIFICATION
 * CENTRE, inert, rather than removed on exit: removing it would make an
 * unanswered check-in vanish from the one place the user can still find it,
 * and the text of the banner is still readable and still says what to do.
 */

#import <Foundation/Foundation.h>

/* Seconds to stay alive waiting for a click when the caller names no window
 * (argv[5]). Long enough to cover a banner's on-screen life plus a user
 * reaching for the trackpad; short enough that a notification nobody touches
 * costs nothing lasting. This is the right answer for a gate or completion
 * toast that is read within moments; it is the WRONG one for a check-in the
 * user answers hours later, which is why the window is a parameter. */
static const NSTimeInterval kDefaultActivationWindow = 30.0;

/* Hard bounds on a CALLER-SUPPLIED window. The ceiling is what keeps the
 * "resident helper per banner" cost finite whatever a caller (or a typo) asks
 * for: a day is far past any banner a person would still click, and a helper
 * that outlives it is a leak, not a feature. The floor stops a zero or a
 * negative from turning "wait for a click" into "exit before the post flushes".
 * An unparsable value falls back to the default rather than to either bound. */
static const NSTimeInterval kMinActivationWindow = 1.0;
static const NSTimeInterval kMaxActivationWindow = 24.0 * 60.0 * 60.0;

/* WHICH COMMAND A CLICK RUNS. The command is stored WITH the banner (its
 * `userInfo`, kept in Notification Centre's own record), not only in the helper
 * process that posted it, because several helpers of ONE bundle id can be alive
 * at once now: a check-in banner keeps its helper for hours while shorter-lived
 * toasts come and go, and macOS routes a click by bundle id. If it hands the
 * click to a different live helper than the one that posted the banner, that
 * helper's own command would open the WRONG session; the carried one cannot.
 * `ownCommand` is only the fallback for a notification that carries none (one
 * posted by an older build, or with no click action). Pure so the dry-run seam
 * can exercise exactly this decision on the real binary without posting. */
static NSString *commandToRun(id notification, NSString *ownCommand) {
    id info = [notification valueForKey:@"userInfo"];
    if ([info isKindOfClass:[NSDictionary class]]) {
        id carried = [(NSDictionary *)info objectForKey:@"command"];
        if ([carried isKindOfClass:[NSString class]] && [carried length] > 0) {
            return carried;
        }
    }
    return ownCommand;
}

@interface LONotifierDelegate : NSObject
@property (copy) NSString *command;
@end

@implementation LONotifierDelegate

/* Present the banner even when our own process happens to be frontmost;
 * without this macOS suppresses it as redundant. */
- (BOOL)userNotificationCenter:(id)center shouldPresentNotification:(id)notification {
    return YES;
}

- (void)userNotificationCenter:(id)center didActivateNotification:(id)notification {
    /* THE COMMAND TRAVELS WITH THE NOTIFICATION, not only with this process
     * (`commandToRun`). */
    NSString *command = commandToRun(notification, self.command);
    if (command.length > 0) {
        /* Detached deliberately: the terminal the user is about to work in
         * must not die with this helper. */
        [NSTask launchedTaskWithLaunchPath:@"/bin/sh"
                                arguments:@[@"-c", command]];
    }
    exit(0);
}

@end

/* The click-wait window for this invocation: argv[5] when it parses cleanly,
 * clamped to [kMin, kMax]; the default otherwise. A function (not inline in
 * main) so the dry-run seam below can report exactly what the run loop would
 * use. */
static NSTimeInterval activationWindow(int argc, const char *argv[]) {
    if (argc > 5 && argv[5][0] != '\0') {
        char *end = NULL;
        double requested = strtod(argv[5], &end);
        /* `end == argv[5]` is "no digits at all"; trailing junk ("30s") is
         * rejected too, so a malformed window can never be half-read into a
         * different one. */
        if (end != argv[5] && *end == '\0') {
            /* `!(x >= min)` rather than `x < min` so NaN lands on the floor. */
            if (!(requested >= kMinActivationWindow)) return kMinActivationWindow;
            if (requested > kMaxActivationWindow) return kMaxActivationWindow;
            return requested;
        }
    }
    return kDefaultActivationWindow;
}

/* True only for LOCAL_OPERATOR_NOTIFIER_DRY_RUN=1. EXACT MATCH, not "is set":
 * a rig variable that leaks into a real runtime as `=0` or empty must not
 * silently turn every banner into a printed line that still reports success. */
static BOOL dryRunRequested(void) {
    const char *value = getenv("LOCAL_OPERATOR_NOTIFIER_DRY_RUN");
    return value != NULL && strcmp(value, "1") == 0;
}

/* What rides the notification itself. The command is stored WITH the banner
 * (in Notification Centre's own record), not only in this process, so the click
 * runs the command of the banner that was clicked even if macOS delivers it to
 * a different live helper of this bundle id. See the delegate. */
static NSDictionary *userInfoForCommand(NSString *command) {
    if (command.length == 0) return nil;
    return @{@"command" : command};
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr,
                    "usage: notifier <title> <body> [click-command] [subtitle] "
                    "[activation-window-seconds]\n");
            return 2;
        }

        Class notificationClass = NSClassFromString(@"NSUserNotification");
        Class centerClass = NSClassFromString(@"NSUserNotificationCenter");
        if (notificationClass == nil || centerClass == nil) {
            /* A future macOS that removed the API: report failure so the
             * caller falls back, rather than crashing a runtime's gate path. */
            return 3;
        }

        id notification = [[notificationClass alloc] init];
        [notification setValue:[NSString stringWithUTF8String:argv[1]] forKey:@"title"];
        [notification setValue:[NSString stringWithUTF8String:argv[2]]
                        forKey:@"informativeText"];
        /* The state category ("Input required") rides the SUBTITLE, which is
         * a field of its own and cannot be clipped away by a long session
         * name — the title used to carry " needs you" appended AFTER the
         * 80-char cap, so the two words explaining the banner were exactly
         * the two the OS truncated (round 3, D11). cmux and the in-band
         * notifier already put it here; this is what makes the three
         * backends agree. */
        if (argc > 4 && argv[4][0] != '\0') {
            [notification setValue:[NSString stringWithUTF8String:argv[4]] forKey:@"subtitle"];
        }

        /* The click command, and the copy of it that rides the banner itself
         * (see `commandToRun`). Written ONCE, here, so the dry-run below reports
         * what a real post would store rather than a parallel computation. */
        NSString *commandText = (argc > 3) ? [NSString stringWithUTF8String:argv[3]] : @"";
        if (commandText.length > 0) {
            [notification setValue:userInfoForCommand(commandText) forKey:@"userInfo"];
        }

        /* DRY-RUN SEAM. With LOCAL_OPERATOR_NOTIFIER_DRY_RUN=1, report what
         * this invocation WOULD do and exit before touching Notification
         * Centre. It exists so the argv contract (the click-wait window) and the
         * click-routing decision (`commandToRun`) can be asserted on the REAL
         * compiled binary in a test or a rig without posting a banner to a real
         * desktop; nothing in the product sets it. The notification object is
         * built exactly as for a real post, so `carried` is what a real banner
         * would store, and `foreign` is what a DIFFERENT live helper (one whose
         * own command is "OTHER-HELPER") would run when handed this banner's
         * click. */
        if (dryRunRequested()) {
            printf("window=%.0f command=%s carried=%s foreign=%s\n",
                   activationWindow(argc, argv), commandText.UTF8String,
                   [[notification valueForKey:@"userInfo"][@"command"] UTF8String] ?: "",
                   [commandToRun(notification, @"OTHER-HELPER") UTF8String]);
            return 0;
        }

        id center = [centerClass performSelector:@selector(defaultUserNotificationCenter)];
        if (center == nil) {
            return 3;
        }

        LONotifierDelegate *delegate = [[LONotifierDelegate alloc] init];
        delegate.command = commandText;
        [center setValue:delegate forKey:@"delegate"];

        [center performSelector:@selector(deliverNotification:) withObject:notification];

        if (delegate.command.length > 0) {
            NSTimeInterval window = activationWindow(argc, argv);
            [[NSRunLoop currentRunLoop]
                runUntilDate:[NSDate dateWithTimeIntervalSinceNow:window]];
        } else {
            /* No click action to wait for; just let the post flush. */
            [NSThread sleepForTimeInterval:1.0];
        }
        return 0;
    }
}
