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
 * Centre after we exit; only the click-through stops working, which is the
 * right thing to lose.
 */

#import <Foundation/Foundation.h>

/* Seconds to stay alive waiting for a click. Long enough to cover a banner's
 * on-screen life plus a user reaching for the trackpad; short enough that a
 * notification nobody touches costs nothing lasting. */
static const NSTimeInterval kActivationWindow = 30.0;

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
    if (self.command.length > 0) {
        /* Detached deliberately: the terminal the user is about to work in
         * must not die with this helper. */
        [NSTask launchedTaskWithLaunchPath:@"/bin/sh"
                                arguments:@[@"-c", self.command]];
    }
    exit(0);
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: notifier <title> <body> [click-command]\n");
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

        id center = [centerClass performSelector:@selector(defaultUserNotificationCenter)];
        if (center == nil) {
            return 3;
        }

        LONotifierDelegate *delegate = [[LONotifierDelegate alloc] init];
        delegate.command = (argc > 3) ? [NSString stringWithUTF8String:argv[3]] : @"";
        [center setValue:delegate forKey:@"delegate"];

        [center performSelector:@selector(deliverNotification:) withObject:notification];

        if (delegate.command.length > 0) {
            [[NSRunLoop currentRunLoop]
                runUntilDate:[NSDate dateWithTimeIntervalSinceNow:kActivationWindow]];
        } else {
            /* No click action to wait for; just let the post flush. */
            [NSThread sleepForTimeInterval:1.0];
        }
        return 0;
    }
}
