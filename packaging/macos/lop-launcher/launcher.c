/*
 * Local Operator launcher — the signed `Program` every Local Operator LaunchAgent names.
 *
 * WHY THIS EXISTS. macOS Background Task Management (BTM: System Settings > Login
 * Items, and the "… can run in the background" notification) attributes a legacy
 * LaunchAgent to the code at its plist's `Program` path. Measured on macOS 27.2
 * (2026-11-14, see the "BTM attribution" note in AGENTS.md):
 *
 *   - an UNSIGNED program is named after its file name ("python3", "python3.12")
 *     with no developer, and EVERY such item posts its own notification;
 *   - a Developer-ID-signed program is named after its bundle and grouped under the
 *     certificate's developer name; once that developer group has been notified,
 *     further items signed by the same developer register as "not notified".
 *
 * So the daemons used to read "python3" and every per-episode job raised a fresh
 * "python3.12 can run in the background" banner. Pointing `Program` at this binary,
 * signed in CI with the Radient Inc. organization Developer ID, makes every item read
 * "Local Operator" under "Radient Inc." and costs one notification per Mac, ever.
 *
 * WHAT IT DOES, and deliberately nothing more:
 *
 *   default mode   exec /bin/sh on the sibling `python3` shim
 *                  (`<bin>/Local Operator.lopruntime/Contents/MacOS/Local Operator`
 *                  -> `<bin>/python3`), passing argv[1..] through unchanged. The shim
 *                  (local_operator/update.py `_DAEMON_SHIM`) stays the ONE place that
 *                  resolves the install generation, so this binary never has to change
 *                  when the layout does — and a changed binary is a changed cdhash,
 *                  whose BTM re-notification behaviour is unmeasured.
 *   target mode    when LOP_LAUNCHER_TARGET names an absolute path, exec THAT with the
 *                  launcher's own argv (argv[0] included) and the variable removed.
 *                  This is for the per-episode owner jobs (local_operator/
 *                  owned_process.py), whose argv[0] is a specific developer venv's
 *                  interpreter: routing them through the shim would silently swap the
 *                  interpreter and its site-packages. The variable comes from the plist's
 *                  EnvironmentVariables, written by the same code that writes `Program`.
 *
 * WHY THAT IS NOT A NEW CAPABILITY. The launcher carries no entitlements and runs as the
 * invoking user; it can exec nothing the user could not exec directly. The shim mode
 * already runs lop's interpreter with arbitrary arguments, so target mode adds no class
 * of behaviour, only an attribution for jobs Local Operator itself writes.
 *
 * CONSTRAINTS:
 *   - never reads $HOME: the shim is located from this image's own resolved path
 *     (_NSGetExecutablePath + realpath), so a redirected HOME cannot redirect it;
 *   - exit 78 (EX_CONFIG, the shim's own "no generation" code) when the path cannot be
 *     resolved, and 126 when exec itself fails, mirroring sh's convention;
 *   - no allocation beyond the argv copy, no CoreFoundation, no Objective-C runtime:
 *     a launchd respawn should cost nothing measurable.
 *
 * Linux has no BTM (systemd user units name the shim and this file is not built);
 * Windows would need an Authenticode-signed .exe and is deferred.
 */
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define EX_CONFIG_CODE 78
#define EXEC_FAILED_CODE 126
#define TARGET_ENV "LOP_LAUNCHER_TARGET"

/* Strip the last path component in place; returns 0 on success. */
static int up_one(char *path) {
    char *slash = strrchr(path, '/');
    if (slash == NULL || slash == path) {
        return -1;
    }
    *slash = '\0';
    return 0;
}

int main(int argc, char *argv[]) {
    const char *target = getenv(TARGET_ENV);
    if (target != NULL && target[0] != '\0') {
        if (target[0] != '/') {
            fprintf(stderr, "Local Operator: %s must be an absolute path\n", TARGET_ENV);
            return EX_CONFIG_CODE;
        }
        /* Copied before unsetenv, which may free the storage `target` points into. */
        char chosen[PATH_MAX];
        if (strlen(target) >= sizeof(chosen)) {
            return EX_CONFIG_CODE;
        }
        strcpy(chosen, target);
        /* Not inherited: a child of the episode must not re-target its own spawns. */
        unsetenv(TARGET_ENV);
        execv(chosen, argv);
        fprintf(stderr, "Local Operator: cannot exec %s: %s\n", chosen, strerror(errno));
        return EXEC_FAILED_CODE;
    }

    char raw[PATH_MAX];
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) != 0) {
        return EX_CONFIG_CODE;
    }
    char shim[PATH_MAX];
    if (realpath(raw, shim) == NULL) {
        return EX_CONFIG_CODE;
    }
    /* .../bin/Local Operator.lopruntime/Contents/MacOS/Local Operator -> .../bin */
    for (int i = 0; i < 4; i++) {
        if (up_one(shim) != 0) {
            return EX_CONFIG_CODE;
        }
    }
    if (strlen(shim) + sizeof("/python3") > sizeof(shim)) {
        return EX_CONFIG_CODE;
    }
    strcat(shim, "/python3");

    /* sh, the shim, then argv[1..]: argc - 1 passthrough args + 2 + NULL. */
    char **args = calloc((size_t)argc + 2, sizeof(char *));
    if (args == NULL) {
        return EXEC_FAILED_CODE;
    }
    args[0] = "sh";
    args[1] = shim;
    for (int i = 1; i < argc; i++) {
        args[i + 1] = argv[i];
    }
    args[argc + 1] = NULL;
    execv("/bin/sh", args);
    fprintf(stderr, "Local Operator: cannot exec /bin/sh %s: %s\n", shim, strerror(errno));
    return EXEC_FAILED_CODE;
}
