/* lop-keyagent — the presence-bearing process for the operator key.

   WHY THIS BINARY EXISTS AT ALL, and why it lives inside an .app.

   A Secure Enclave key can only live in the DATA-PROTECTION keychain, and that
   keychain admits a caller only when the caller's code signature carries the
   keychain entitlement AND a provisioning profile authorizes it. A Python
   interpreter cannot be signed with that profile: the runtime is installed by
   `uv tool install` / `pip`, on many hosts and at many paths, and the profile is
   bound to a bundle identifier. So a Python process can never create, find or
   use this key on any Mac, however correct its CoreFoundation calls are.

   Measured on macOS 27.0 (26A428), arm64, with the Developer ID identity this
   project ships under, one shape per row (probe: ~/tools/lop-signing/probe):

     adhoc bare tool, no entitlement                          -> -34018
     Developer ID bare tool, no entitlement                    -> -34018
     Developer ID bare tool + keychain-access-groups           -> SIGKILL (137)
     app bundle + entitlement + embedded profile               -> PASS
     bare tool + application-identifier only                   -> -34018
     bundle + application-identifier + embedded profile        -> PASS

   Two consequences are baked into this file:

   1. THE BUNDLE IS LOAD-BEARING. TN3137 says a command-line tool needs an
      app-like bundle to embed a profile, and the measurement above agrees: a
      bare Mach-O carrying the entitlement is refused (-34018) or killed.
   2. THE PROFILE IS LOAD-BEARING, AND ITS ABSENCE IS NOT AN ERROR. Without it,
      an *entitled* bundle is SIGKILLed by the kernel — nothing this program can
      catch, print or exit with, which is why the Python side verifies the bundle
      BEFORE exec'ing it (local_operator/operator/macos/keyagent.py).

   OWNERSHIP DISCIPLINE, which the Python twin of this sequence got wrong three
   times before it was deleted in favour of this file: every dictionary is built
   with the TYPED CoreFoundation callbacks (NULL callbacks retain nothing, and a
   generation dictionary so built SIGSEGVs inside SecKeyCreateRandomKey — 6 of 6
   runs, #1547); every object this program creates is released exactly once, by
   the scope that created it; the application tag is created ONCE in main() and
   outlives every ladder iteration, so a failed rung can never hand a released
   CFDataRef to the next one. `-fsanitize=address` runs over the create/exists/
   doctor path in CI, and the repeated-call loop is what makes leaks visible.

   THE OWNERSHIP MODEL IS ASSERTED IN THIS BINARY (`selftest`), not by a sanitizer:
   measured on macOS 27.0, `clang -fsanitize=address` links a runtime dyld refuses
   to load into a signed process ("Sanitizer load violates platform policy", exit
   134), and `leaks --atExit` reports "0 leaks" for a control binary that leaks
   4 KiB on purpose. A dead instrument returns a reading, so neither is used and
   the invariant is checked with CFGetRetainCount in `cmd_selftest` instead. The
   verb set is therefore seven: create, public, exists, sign, doctor, selftest,
   purge.

   WHAT THIS PROGRAM NEVER DOES. It never exports private key material: the only
   key it reads back is the PUBLIC half (SecKeyCopyPublicKey +
   SecKeyCopyExternalRepresentation), and a Secure Enclave key cannot export its
   private half in any case. It never signs anything but the bytes it is given on
   stdin — domain separation is the caller's job, because any framing here would
   be a second definition of the wire format. Its stdout is JSON on one line and
   nothing else; every diagnostic goes to stderr.

   It is a ONE-SHOT process: no daemon, no socket, no lifecycle, no shared state,
   so any leak is bounded by a process measured in milliseconds.

   Build (see .github/workflows/publish.yml, job `keyagent-macos`):
     clang -O2 -arch arm64 -arch x86_64 -mmacosx-version-min=11 \
       -o lop-keyagent se-keyagent.c -framework Security -framework CoreFoundation
*/

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecAccessControl.h>
#include <Security/SecItem.h>
#include <Security/SecKey.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*: The protocol version, echoed in every reply. Helper and client ship in the
   same wheel, so a mismatch means a mixed or broken install, and the honest
   answer is to refuse rather than to interpret (exit 5). */
#define PROTOCOL 1

/*: The application tag the product's operator key is stored under. The Python
   side owns the same constant (keychain.APPLICATION_TAG) and always passes it
   with --tag; this is the default, and it is also the one tag `purge` refuses,
   so no code path — including this program run by hand — can delete the
   operator's real key. */
#define DEFAULT_TAG "com.local-operator.operator.v1"

/*: Exit codes. See docs/design/operator-key-agent.md §4.3. */
#define EXIT_CANCELLED 2 /* the human cancelled the presence prompt (-128)    */
#define EXIT_NO_KEY 3    /* nothing stored under the tag (-25300)              */
#define EXIT_REFUSED 4   /* entitlement / signature / profile refused          */
#define EXIT_USAGE 5     /* protocol mismatch, bad verb, bad flag               */

/*: The protection classes, strictest FIRST — the same order the Python ladder
   documents (keychain.SecureEnclaveBackend.PROTECTION_LADDER). Measured under an
   entitled process: BOTH are accepted, so the order decides which one a key is
   actually made with, and the strictest one wins. */
static const char *const LADDER[] = {
    "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
    "kSecAttrAccessibleWhenUnlockedThisDeviceOnly",
};
#define LADDER_LEN ((int)(sizeof(LADDER) / sizeof(LADDER[0])))

/*: The most refusals one run can report: one per ladder rung, at each of two
   sites. A fixed array keeps this file allocation-free. */
#define MAX_REFUSALS (LADDER_LEN * 2)

typedef struct {
    const char *site;       /* "access control" | "key generation" | "signature" */
    const char *protection; /* the rung that refused, or "" where there is none */
    OSStatus status;
    char detail[512]; /* the framework's own sentence, truncated if huge */
} refusal_t;

/* ---------------------------------------------------------------------------
 * JSON output
 *
 * Hand-rolled because this program's whole output is five small objects and a
 * JSON library would be a build dependency. The escaping is the part that
 * matters: framework detail text routinely contains quotes, backslashes and, on
 * some paths, newlines, and a raw quote there would produce a reply the client
 * cannot parse — a failure that reads as a broken install.
 * ------------------------------------------------------------------------- */

static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        switch (*p) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*p < 0x20) {
                    printf("\\u%04x", (unsigned)*p);
                } else {
                    putchar(*p);
                }
        }
    }
    putchar('"');
}

/*: base64url, unpadded, written straight to stdout. Both payloads this program
   returns (a 65-byte P-256 point and a DER signature) are binary, and the client
   already speaks this alphabet for the wire format. */
static void json_b64url(const unsigned char *raw, size_t len) {
    static const char *alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    putchar('"');
    for (size_t i = 0; i < len; i += 3) {
        unsigned value = (unsigned)raw[i] << 16;
        int remaining = (int)(len - i);
        if (remaining > 1) value |= (unsigned)raw[i + 1] << 8;
        if (remaining > 2) value |= (unsigned)raw[i + 2];
        putchar(alphabet[(value >> 18) & 0x3F]);
        putchar(alphabet[(value >> 12) & 0x3F]);
        if (remaining > 1) putchar(alphabet[(value >> 6) & 0x3F]);
        if (remaining > 2) putchar(alphabet[value & 0x3F]);
    }
    putchar('"');
}

static const char *describe_status(OSStatus status) {
    switch (status) {
        case errSecSuccess: return "errSecSuccess";
        case errSecParam: return "errSecParam";
        case errSecItemNotFound: return "errSecItemNotFound";
        case errSecDuplicateItem: return "errSecDuplicateItem";
        case errSecUserCanceled: return "errSecUserCanceled";
        case errSecAuthFailed: return "errSecAuthFailed";
        case errSecMissingEntitlement: return "errSecMissingEntitlement";
        default: return "other";
    }
}

/*: One refusal, recorded from a CFErrorRef or from a bare OSStatus.
 *
 *  The DETAIL is read with CFErrorCopyDescription, and the error object is
 *  released here — the caller must not touch it afterwards. The code is captured
 *  BEFORE this call for the same reason: reading a released CFErrorRef's code is
 *  the use-after-free this repository already shipped once (#1547). */
static void record_refusal(refusal_t *refusal, const char *site, const char *protection,
                           OSStatus status, CFErrorRef err) {
    refusal->site = site;
    refusal->protection = protection;
    refusal->status = status;
    refusal->detail[0] = '\0';
    if (!err) {
        snprintf(refusal->detail, sizeof refusal->detail, "%s", describe_status(status));
        return;
    }
    CFStringRef text = CFErrorCopyDescription(err);
    if (text) {
        CFStringGetCString(text, refusal->detail, (CFIndex)sizeof refusal->detail,
                           kCFStringEncodingUTF8);
        CFRelease(text);
    }
    if (refusal->detail[0] == '\0') snprintf(refusal->detail, sizeof refusal->detail, "%s",
                                             describe_status(status));
}

/*: A failure reply, as docs/design/operator-key-agent.md §4.3 specifies it.
 *
 *  `site`/`status`/`detail` are the PRIMARY refusal — the last rung's, i.e. the
 *  one that ended the attempt — because that is what the Python diagnosis table
 *  is keyed on. `refusals` carries every rung's refusal as well, so the caller can
 *  build ONE message that names each protection class beside the refusal they
 *  shared instead of guessing which class produced the code it was handed. */
static void emit_failure(const refusal_t *refusals, int count, const refusal_t *primary) {
    printf("{\"ok\":false,\"protocol\":%d,\"site\":", PROTOCOL);
    json_string(primary->site);
    printf(",\"status\":%d,\"detail\":", (int)primary->status);
    json_string(primary->detail);
    if (count > 1) {
        fputs(",\"refusals\":[", stdout);
        for (int i = 0; i < count; i++) {
            if (i) putchar(',');
            fputs("{\"site\":", stdout);
            json_string(refusals[i].site);
            fputs(",\"protection\":", stdout);
            json_string(refusals[i].protection);
            printf(",\"status\":%d,\"detail\":", (int)refusals[i].status);
            json_string(refusals[i].detail);
            putchar('}');
        }
        putchar(']');
    }
    fputs("}\n", stdout);
}

static void emit_usage_error(const char *detail) {
    printf("{\"ok\":false,\"protocol\":%d,\"site\":\"usage\",\"status\":%d,\"detail\":",
           PROTOCOL, EXIT_USAGE);
    json_string(detail);
    fputs("}\n", stdout);
}

/* ---------------------------------------------------------------------------
 * The key: find, and the public half
 * ------------------------------------------------------------------------- */

/*: The key stored under `tag`, or NULL with `*status` set.
 *
 *  The returned SecKeyRef is +1 (SecItemCopyMatching follows the CF ownership
 *  rule), so the caller releases it. The query deliberately does NOT set
 *  kSecUseDataProtectionKeychain: measured on this host through the entitled
 *  path, the round trip is identical with and without it (both the create this
 *  file does and this find), so carrying it would be an unvalidated attribute.
 *  What puts a Secure Enclave key in the data-protection keychain is
 *  kSecAttrTokenIDSecureEnclave, not this attribute. */
static SecKeyRef find_key(CFDataRef tag, OSStatus *status) {
    const void *keys[] = {kSecClass, kSecAttrApplicationTag, kSecAttrKeyType,
                          kSecReturnRef, kSecMatchLimit};
    const void *values[] = {kSecClassKey, tag, kSecAttrKeyTypeECSECPrimeRandom,
                            kCFBooleanTrue, kSecMatchLimitOne};
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values, 5,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    CFTypeRef found = NULL;
    *status = SecItemCopyMatching(query, &found);
    CFRelease(query);
    if (*status == errSecSuccess && found) return (SecKeyRef)found;
    if (found) CFRelease(found);
    return NULL;
}

/*: The 65-byte uncompressed P-256 point of `key`'s public half, or NULL.
 *
 *  PUBLIC half only, by construction: SecKeyCopyPublicKey then
 *  SecKeyCopyExternalRepresentation. The returned CFDataRef is +1. */
static CFDataRef public_point(SecKeyRef key) {
    SecKeyRef public_key = SecKeyCopyPublicKey(key);
    if (!public_key) return NULL;
    CFErrorRef err = NULL;
    CFDataRef raw = SecKeyCopyExternalRepresentation(public_key, &err);
    if (err) CFRelease(err); /* released here, and never read afterwards */
    CFRelease(public_key);
    return raw;
}

/*: THE ACCESS-CONTROL FLAG PAIR, defined once for this whole file.
 *
 *  Apple's own symbols, never a shift literal: an earlier revision of the Python
 *  twin of this sequence shipped ``1<<0 | 1<<2``, which
 *  ``SecAccessControlCreateWithFlags`` refuses with errSecParam for every
 *  protection class, so key generation was never reached (#1547). The
 *  header-asserting unit test pins these two values against
 *  ``Security.framework/Headers/SecAccessControl.h``, and pins the fact that this
 *  file names the SYMBOLS rather than numbers. */
#define ACCESS_FLAGS (kSecAccessControlPrivateKeyUsage | kSecAccessControlUserPresence)

/*: The access-control object for one ladder rung, or NULL with ``*err`` set.
 *
 *  Returns +1 (CF naming convention for a Create). The caller releases it, and
 *  releases ``*err`` if the call failed. */
static SecAccessControlRef make_access_control(int rung, CFErrorRef *err) {
    return SecAccessControlCreateWithFlags(
        NULL, /* kCFAllocatorDefault */
        (rung == 0) ? kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly
                    : kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ACCESS_FLAGS, err);
}

/*: ``kSecPrivateKeyAttrs``: permanent, held under ``tag``, access-controlled by
 *  ``access``. Returns +1. */
static CFMutableDictionaryRef make_private_attrs(CFDataRef tag, SecAccessControlRef access) {
    CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(attrs, kSecAttrIsPermanent, kCFBooleanTrue);
    CFDictionarySetValue(attrs, kSecAttrApplicationTag, tag);
    CFDictionarySetValue(attrs, kSecAttrAccessControl, access);
    return attrs;
}

/*: The generation dictionary: EC P-256, on the Secure Enclave token, with
 *  ``private_attrs``. Returns +1. ``bits`` is consumed but not owned. */
static CFMutableDictionaryRef make_generation_dict(CFMutableDictionaryRef private_attrs,
                                                  CFNumberRef bits) {
    CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(attrs, kSecAttrKeyType, kSecAttrKeyTypeECSECPrimeRandom);
    CFDictionarySetValue(attrs, kSecAttrKeySizeInBits, bits);
    CFDictionarySetValue(attrs, kSecAttrTokenID, kSecAttrTokenIDSecureEnclave);
    CFDictionarySetValue(attrs, kSecPrivateKeyAttrs, private_attrs);
    return attrs;
}

/*: ``kSecAttrKeySizeInBits`` as a CFNumberRef. Returns +1. */
static CFNumberRef make_size_number(void) {
    int bits = 256;
    return CFNumberCreate(NULL, kCFNumberSInt32Type, &bits);
}

/* ---------------------------------------------------------------------------
 * The verbs
 * ------------------------------------------------------------------------- */

/*: create — make the key if it is not there, and report the public half.
 *
 *  IDEMPOTENT. A second call on a host that already has a key must not make a
 *  second key: the anchored contract is one key per machine, a new anchor
 *  invalidates every paired device's certificate, and the repository's rotation
 *  rule is "write a new key, install a new anchor" — not "delete the old one".
 *  So a duplicate item is answered by finding the existing key and returning it
 *  with "reused":true.
 *
 *  WHY THE CREATE IS ATTEMPTED BEFORE THE FIND: a find on a presence-gated key
 *  is a query the framework may authenticate, and creation is measured not to
 *  raise user presence at all. Attempting the create first therefore keeps the
 *  common (fresh host) path free of any interaction, and the duplicate case costs
 *  one extra find. */
static int cmd_create(CFDataRef tag) {
    refusal_t refusals[MAX_REFUSALS];
    int count = 0;

    for (int i = 0; i < LADDER_LEN; i++) {
        /* The access control object, built by the same helper the selftest
           asserts on (make_access_control), so the ownership model verified in
           CI is the ownership model this loop runs. */
        CFErrorRef access_err = NULL;
        SecAccessControlRef access = make_access_control(i, &access_err);
        if (!access) {
            OSStatus status = access_err ? (OSStatus)CFErrorGetCode(access_err) : errSecParam;
            record_refusal(&refusals[count++], "access control", LADDER[i], status, access_err);
            if (access_err) CFRelease(access_err);
            continue;
        }

        /* kSecPrivateKeyAttrs: permanent, the tag, the access control. */
        CFMutableDictionaryRef private_attrs = make_private_attrs(tag, access);
        /* The dictionary RETAINS access (typed callbacks), so this program's
           reference is released as soon as the dictionary holds it. */
        CFRelease(access);

        CFNumberRef size = make_size_number();
        CFMutableDictionaryRef attrs = make_generation_dict(private_attrs, size);
        /* Both were retained by the generation dictionary; this program's own
           references go now, which is why every release below is paired with
           exactly one creation. */
        CFRelease(size);
        CFRelease(private_attrs);

        CFErrorRef err = NULL;
        SecKeyRef key = SecKeyCreateRandomKey(attrs, &err);
        CFRelease(attrs);
        if (!key) {
            OSStatus status = err ? (OSStatus)CFErrorGetCode(err) : errSecParam;
            if (status == errSecDuplicateItem) {
                /* A key is already stored under this tag: report the one that is
                   there, and never a second key. */
                if (err) CFRelease(err);
                OSStatus found_status = errSecSuccess;
                SecKeyRef existing = find_key(tag, &found_status);
                if (!existing) {
                    refusal_t one = {0};
                    record_refusal(&one, "key lookup", "", found_status, NULL);
                    emit_failure(&one, 1, &one);
                    return EXIT_REFUSED;
                }
                CFDataRef point = public_point(existing);
                CFRelease(existing);
                if (!point) {
                    emit_usage_error("the stored key has no exportable public half");
                    return EXIT_USAGE;
                }
                fputs("{\"ok\":true,\"protocol\":", stdout);
                printf("%d,\"spki\":", PROTOCOL);
                json_b64url((const unsigned char *)CFDataGetBytePtr(point),
                            (size_t)CFDataGetLength(point));
                fputs(",\"reused\":true}\n", stdout);
                CFRelease(point);
                return 0;
            }
            record_refusal(&refusals[count++], "key generation", LADDER[i], status, err);
            if (err) CFRelease(err);
            continue;
        }

        CFDataRef point = public_point(key);
        CFRelease(key);
        if (!point) {
            emit_usage_error("the new key has no exportable public half");
            return EXIT_USAGE;
        }
        fputs("{\"ok\":true,\"protocol\":", stdout);
        printf("%d,\"spki\":", PROTOCOL);
        json_b64url((const unsigned char *)CFDataGetBytePtr(point),
                    (size_t)CFDataGetLength(point));
        fputs(",\"reused\":false,\"rung\":", stdout);
        json_string(LADDER[i]);
        fputs("}\n", stdout);
        CFRelease(point);
        return 0;
    }

    /* Every rung refused. */
    emit_failure(refusals, count, &refusals[count - 1]);
    return EXIT_REFUSED;
}

/*: public / exists share the one query; the difference is what they report. */
static int cmd_public_or_exists(CFDataRef tag, int existence_only) {
    OSStatus status = errSecSuccess;
    SecKeyRef key = find_key(tag, &status);
    if (!key) {
        /* `exists` answers a QUESTION, so "there is none" is its success reply,
           not a failure: the caller that must tell the two apart is `public`,
           where errSecItemNotFound is exit 3 ("no operator key on this host"). */
        if (existence_only && status == errSecItemNotFound) {
            printf("{\"ok\":true,\"protocol\":%d,\"present\":false}\n", PROTOCOL);
            return 0;
        }
        refusal_t one = {0};
        record_refusal(&one, "key lookup", "", status, NULL);
        emit_failure(&one, 1, &one);
        return (status == errSecItemNotFound) ? EXIT_NO_KEY : EXIT_REFUSED;
    }
    if (existence_only) {
        CFRelease(key);
        printf("{\"ok\":true,\"protocol\":%d,\"present\":true}\n", PROTOCOL);
        return 0;
    }
    CFDataRef point = public_point(key);
    CFRelease(key);
    if (!point) {
        emit_usage_error("the stored key has no exportable public half");
        return EXIT_USAGE;
    }
    printf("{\"ok\":true,\"protocol\":%d,\"spki\":", PROTOCOL);
    json_b64url((const unsigned char *)CFDataGetBytePtr(point),
                (size_t)CFDataGetLength(point));
    fputs("}\n", stdout);
    CFRelease(point);
    return 0;
}

/*: sign — sign EXACTLY the bytes on stdin, and nothing else.
 *
 *  The message arrives on stdin rather than in argv because argv is world-readable
 *  through `ps` and the message carries a session id, a card id and a single-use
 *  challenge. This is the one verb that raises the presence sheet: it blocks in
 *  SecKeyCreateSignature until a human answers, and it deliberately has NO internal
 *  timeout — the caller bounds the wait (local_operator.operator.macos.keyagent
 *  SIGN_TIMEOUT_SECONDS), because only the caller knows what is at stake.
 *
 *  A cancelled sheet is exit 2 rather than a failure: the human said no, which is
 *  a normal outcome and is reported as "nothing was signed". */
static int cmd_sign(CFDataRef tag) {
    unsigned char *message = NULL;
    size_t length = 0;
    unsigned char buffer[8192];
    size_t got;
    while ((got = fread(buffer, 1, sizeof buffer, stdin)) > 0) {
        unsigned char *grown = realloc(message, length + got);
        if (!grown) {
            free(message);
            emit_usage_error("out of memory reading the message to sign");
            return EXIT_USAGE;
        }
        message = grown;
        memcpy(message + length, buffer, got);
        length += got;
    }

    OSStatus status = errSecSuccess;
    SecKeyRef key = find_key(tag, &status);
    if (!key) {
        free(message);
        refusal_t one = {0};
        record_refusal(&one, "key lookup", "", status, NULL);
        emit_failure(&one, 1, &one);
        return (status == errSecItemNotFound) ? EXIT_NO_KEY : EXIT_REFUSED;
    }

    CFDataRef payload = CFDataCreate(NULL, message, (CFIndex)length);
    free(message);
    CFErrorRef err = NULL;
    CFDataRef signature = SecKeyCreateSignature(
        key, kSecKeyAlgorithmECDSASignatureMessageX962SHA256, payload, &err);
    /* Released here, once, on both paths: the call does not take ownership. */
    CFRelease(payload);
    CFRelease(key);
    if (!signature) {
        OSStatus refused = err ? (OSStatus)CFErrorGetCode(err) : errSecParam;
        refusal_t one = {0};
        record_refusal(&one, "signature", "", refused, err);
        if (err) CFRelease(err);
        emit_failure(&one, 1, &one);
        return (refused == errSecUserCanceled) ? EXIT_CANCELLED : EXIT_REFUSED;
    }
    printf("{\"ok\":true,\"protocol\":%d,\"signature\":", PROTOCOL);
    json_b64url((const unsigned char *)CFDataGetBytePtr(signature),
                (size_t)CFDataGetLength(signature));
    fputs("}\n", stdout);
    CFRelease(signature);
    return 0;
}

/*: doctor — what this installation can and cannot do, measured rather than claimed.
 *
 *  Three independent facts, none of which writes anything or raises a prompt:
 *
 *   * `rung`   — the strictest protection class this process can build an access
 *                control object for. This is the measurement that would have caught
 *                the released flag-pair bug, now asked inside the entitled process
 *                instead of from an interpreter that could never use the answer.
 *   * `keychain` — whether the data-protection keychain answers this process at all
 *                (a query; errSecMissingEntitlement here means the entitlement is
 *                not in effect).
 *   * `profile` — whether this bundle carries its embedded provisioning profile.
 *                Reported because its absence is a kernel SIGKILL rather than an
 *                error, so the only place it can be *reported* is before exec.
 */
static int cmd_doctor(CFDataRef tag) {
    const char *rung = "none";
    OSStatus rung_status = errSecParam;
    for (int i = 0; i < LADDER_LEN; i++) {
        CFErrorRef access_err = NULL;
        SecAccessControlRef access = SecAccessControlCreateWithFlags(
            NULL, (i == 0) ? kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly
                           : kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecAccessControlPrivateKeyUsage | kSecAccessControlUserPresence, &access_err);
        if (access_err) CFRelease(access_err);
        if (access) {
            CFRelease(access);
            rung = LADDER[i];
            rung_status = errSecSuccess;
            break;
        }
    }

    OSStatus query_status = errSecSuccess;
    SecKeyRef key = find_key(tag, &query_status);
    if (key) CFRelease(key);

    char executable[4096];
    uint32_t size = (uint32_t)sizeof executable;
    int profile_ok = 0;
    if (_NSGetExecutablePath(executable, &size) == 0) {
        /* .../lop-keyagent.app/Contents/MacOS/lop-keyagent -> strip three components. */
        char *cursor = executable;
        for (int i = 0; i < 3; i++) {
            char *slash = strrchr(cursor, '/');
            if (!slash) break;
            *slash = '\0';
        }
        char profile[4096];
        snprintf(profile, sizeof profile, "%s/Contents/embedded.provisionprofile", executable);
        FILE *handle = fopen(profile, "rb");
        if (handle) {
            profile_ok = 1;
            fclose(handle);
        }
    }

    int ok = (rung_status == errSecSuccess) && (query_status == errSecSuccess ||
                                                query_status == errSecItemNotFound);
    printf("{\"ok\":%s,\"protocol\":%d,\"rung\":", ok ? "true" : "false", PROTOCOL);
    json_string(rung);
    fputs(",\"keychain\":", stdout);
    json_string(query_status == errSecSuccess || query_status == errSecItemNotFound ? "ok"
                                                                                     : describe_status(query_status));
    /* The raw status travels beside the word: a caller that has to branch on it
       must not read prose, and a report should quote the number. */
    printf(",\"keychain_status\":%d,\"profile\":", (int)query_status);
    json_string(profile_ok ? "ok" : "missing");
    fputs("}\n", stdout);
    return ok ? 0 : EXIT_REFUSED;
}

/*: One asserted property of the ownership model, and the two numbers behind it. */
typedef struct {
    const char *name;
    long expected;
    long actual;
} selftest_check_t;

/*: The selftest's report: every check, its expectation and its reading. */
static void report_selftest(const selftest_check_t *checks, int count, int failed) {
    fputs("{\"ok\":", stdout);
    fputs(failed ? "false" : "true", stdout);
    printf(",\"protocol\":%d,\"checks\":[", PROTOCOL);
    for (int i = 0; i < count; i++) {
        if (i) putchar(',');
        fputs("{\"name\":", stdout);
        json_string(checks[i].name);
        printf(",\"expected\":%ld,\"actual\":%ld}", checks[i].expected, checks[i].actual);
    }
    fputs("]}\n", stdout);
}

/* ---------------------------------------------------------------------------
 * selftest — the C-side ownership check, and why it is NOT an ASan pass
 * ---------------------------------------------------------------------------
 *
 * WHAT WAS ASKED FOR HERE was an ASan/leaks pass over this binary in CI. Both
 * instruments are unavailable on the macOS this ships to, measured 2026-09-24 on
 * macOS 27.0 (26A428) — and the second one is the dangerous kind of unavailable:
 *
 *   - `clang -fsanitize=address` links libclang_rt.asan_osx_dynamic.dylib, and
 *     dyld REFUSES to load it into a signed process: "Sanitizer load violates
 *     platform policy", SIGABRT, exit 134. There is no ASan reading to trust.
 *   - `leaks --atExit` RUNS and prints a report, and says
 *     "0 leaks for 0 total leaked bytes" for a control binary that deliberately
 *     leaks 4 KiB. A dead instrument that returns a reading is worse than no
 *     instrument, so nothing may be concluded from its silence.
 *
 * So the ownership model is asserted IN THIS BINARY, with CFGetRetainCount —
 * CoreFoundation's own instrument for exactly this question. Two properties make
 * it a real check rather than a ritual:
 *
 *   1. the assertions are INVARIANTS (a dictionary retains what it holds, and
 *      releasing the dictionary returns that reference) rather than absolute
 *      counts, so a CoreFoundation that starts objects at a different count does
 *      not break them;
 *   2. they are made against the CONSTRUCTORS `create` ITSELF RUNS
 *      (make_access_control / make_private_attrs / make_generation_dict), so what
 *      CI verifies and what ships cannot diverge without editing both in one
 *      place.
 *
 * NO ENTITLEMENT, NO KEYCHAIN, NO PROMPT, NO WRITES: it creates no key and
 * touches no store, so CI can run it on the assembled binary before signing.
 */
static int cmd_selftest(CFDataRef tag) {
    selftest_check_t checks[12];
    int count = 0;
    int failed = 0;

#define CHECK(label, want, got)                                                                    \
    do {                                                                                           \
        checks[count].name = (label);                                                              \
        checks[count].expected = (long)(want);                                                     \
        checks[count].actual = (long)(got);                                                        \
        if ((long)(want) != (long)(got)) failed++;                                                 \
        count++;                                                                                   \
    } while (0)

    /* 1. THE FLAG PAIR, at both rungs, before any keychain is involved. This is
       the assertion that catches a transposed or guessed pair: the released build
       passed the framework a pair it refuses with errSecParam, so key generation
       was never reached at all. */
    CFErrorRef rung_err = NULL;
    SecAccessControlRef strict = make_access_control(0, &rung_err);
    if (rung_err) {
        CFRelease(rung_err);
        rung_err = NULL;
    }
    CHECK("rung 0 (passcode-set) accepts the flag pair", 1, strict != NULL);
    SecAccessControlRef relaxed = make_access_control(1, &rung_err);
    if (rung_err) {
        CFRelease(rung_err);
        rung_err = NULL;
    }
    CHECK("rung 1 (unlocked) accepts the flag pair", 1, relaxed != NULL);
    if (!strict || !relaxed) {
        /* Nothing to assert ownership over. Report and stop — releasing only what
           exists, because CFRelease(NULL) is a crash. */
        if (strict) CFRelease(strict);
        if (relaxed) CFRelease(relaxed);
        report_selftest(checks, count, failed);
        return EXIT_REFUSED;
    }

    /* 2. THE PRIVATE-ATTRIBUTE DICTIONARY, in create()'s exact order: it must
       RETAIN what it is handed (with NULL callbacks it would not, which is the
       SIGSEGV inside SecKeyCreateRandomKey measured 6/6 in #1547), and the
       program must then give up its OWN reference. */
    const CFIndex tag_base = CFGetRetainCount(tag);
    const CFIndex access_base = CFGetRetainCount(strict);
    CFMutableDictionaryRef private_attrs = make_private_attrs(tag, strict);
    CHECK("the private-attribute dictionary retains the tag", tag_base + 1, CFGetRetainCount(tag));
    CHECK("the private-attribute dictionary retains the access control", access_base + 1,
          CFGetRetainCount(strict));
    CFRelease(strict);
    /* The mirror of the check above: giving up OUR reference leaves exactly the
       dictionary's one, so a future edit that dropped this release would read 2.
       CFGetRetainCount cannot be called on a freed object, so this is the closest
       reading to "released exactly once" there is. */
    CHECK("the access control is held by the dictionary alone", access_base,
          CFGetRetainCount(strict));

    /* 3. THE SIZE NUMBER, whose historical defect was the mirror image: it was
       never released, because a docstring claimed the dictionary would do it.

       The number here is a FRACTIONAL DOUBLE (256.5) and not the 256 the product
       passes, because CoreFoundation serves an integral value as an IMMORTAL
       tagged object: measured on macOS 27.0, CFGetRetainCount reads LONG_MAX for
       kCFNumberSInt32Type 256, for kCFNumberLongType 2^40 AND for
       kCFNumberDoubleType 256.0, while a fractional double is an ordinary object
       with a count of 1. That measurement is itself the answer to part of the
       historical defect — a CFNumberRef the old ladder never released cannot have
       leaked, because the small integer in question has no reference to leak — so
       what is checked here is the DICTIONARY's ownership of a number (the code
       path that was wrong), on an object whose reference is countable. */
    double bits_value = 256.5; /* deliberately NOT a key size: see below */
    CFNumberRef size = CFNumberCreate(NULL, kCFNumberDoubleType, &bits_value);
    CHECK("the size number starts with one reference", 1, CFGetRetainCount(size));
    const CFIndex attrs_base = CFGetRetainCount(private_attrs);
    CFMutableDictionaryRef generation = make_generation_dict(private_attrs, size);
    CHECK("the generation dictionary retains the size number", 2, CFGetRetainCount(size));
    CHECK("the generation dictionary retains the private attributes", attrs_base + 1,
          CFGetRetainCount(private_attrs));
    CFRelease(size);
    CHECK("the size number is held by the dictionary alone", 1, CFGetRetainCount(size));
    CFRelease(private_attrs);
    CHECK("the private attributes are held by the generation dictionary alone", attrs_base,
          CFGetRetainCount(private_attrs));
    CFRelease(generation);

    /* 4. AND THE TAG SURVIVES EVERY LADDER ITERATION, which is what a released
       tag across iterations broke in the Python twin. main() owns it, so after
       two rungs of local objects the count is exactly where it started. */
    CHECK("the tag outlives the ladder with one reference", tag_base, CFGetRetainCount(tag));
    CFRelease(relaxed);

    report_selftest(checks, count, failed);
    return failed ? EXIT_REFUSED : 0;
}

/*: purge — remove the item stored under a TEST tag, and report SecItemDelete.
 *
 *  THIS IS NOT A PRODUCT VERB, and it cannot delete the operator's key: the tag
 *  DEFAULT_TAG is refused outright, so `lop operator purge` does not exist and
 *  `lop-keyagent purge --tag com.local-operator.operator.v1` fails with exit 5.
 *  It exists because deletion is gated by the SAME entitlement as creation — an
 *  unsigned process asking to delete the item gets errSecItemNotFound (-25300),
 *  measured — so without an entitled delete verb the keychain item a test or a QA
 *  run creates could never be removed, and this repository's rule is that an
 *  agent leaves nothing behind. The status is printed either way (errSecSuccess
 *  when something was deleted, errSecItemNotFound when there was nothing to
 *  delete) and "nothing was there" is not a failure for a cleanup verb. */
static int cmd_purge(CFDataRef tag, const char *tagtext) {
    if (strcmp(tagtext, DEFAULT_TAG) == 0) {
        emit_usage_error("purge refuses the operator's own application tag: the product has no "
                         "delete verb by design (rotation writes a new key and installs a new "
                         "anchor)");
        return EXIT_USAGE;
    }
    const void *keys[] = {kSecClass, kSecAttrApplicationTag};
    const void *values[] = {kSecClassKey, tag};
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values, 2,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    OSStatus status = SecItemDelete(query);
    CFRelease(query);
    printf("{\"ok\":true,\"protocol\":%d,\"deleted\":%d}\n", PROTOCOL, (int)status);
    /* The human-readable form of the same number, for a log or a terminal. */
    fprintf(stderr, "SecItemDelete=%d (%s)\n", (int)status, describe_status(status));
    return 0;
}

/* --------------------------------------------------------------------------- */

static void usage(void) {
    fputs("usage: lop-keyagent <create|public|exists|sign|doctor|selftest|purge> [--tag <tag>]\n"
          "  sign reads the message to sign from stdin\n"
          "  selftest asserts this binary's CoreFoundation ownership model (no keychain)\n",
          stderr);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        emit_usage_error("no verb given");
        usage();
        return EXIT_USAGE;
    }
    const char *verb = argv[1];
    const char *tagtext = DEFAULT_TAG;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--tag") == 0 && i + 1 < argc) {
            tagtext = argv[++i];
        } else {
            emit_usage_error("unknown argument");
            usage();
            return EXIT_USAGE;
        }
    }

    /* THE APPLICATION TAG IS CREATED ONCE, HERE, AND OUTLIVES EVERY LADDER
       ITERATION. It used to be created and released inside a loop in Python,
       which handed a released CFDataRef to the next iteration's dictionary — the
       use-after-free #1547 fixed. Making the tag the one object main owns is how
       that class of bug is structured out rather than re-reviewed. */
    CFDataRef tag = CFDataCreate(NULL, (const UInt8 *)tagtext, (CFIndex)strlen(tagtext));
    if (!tag) {
        emit_usage_error("could not build the application tag");
        return EXIT_USAGE;
    }

    int rc;
    if (strcmp(verb, "create") == 0) rc = cmd_create(tag);
    else if (strcmp(verb, "public") == 0) rc = cmd_public_or_exists(tag, 0);
    else if (strcmp(verb, "exists") == 0) rc = cmd_public_or_exists(tag, 1);
    else if (strcmp(verb, "sign") == 0) rc = cmd_sign(tag);
    else if (strcmp(verb, "doctor") == 0) rc = cmd_doctor(tag);
    else if (strcmp(verb, "selftest") == 0) rc = cmd_selftest(tag);
    else if (strcmp(verb, "purge") == 0) rc = cmd_purge(tag, tagtext);
    else {
        emit_usage_error("unknown verb");
        usage();
        rc = EXIT_USAGE;
    }
    CFRelease(tag);
    return rc;
}
