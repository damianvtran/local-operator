/** Compile-only contract regression, included by the real mobile tsc build.
 * No runtime import or catalogue bootstrap: the four rejected calls pin the
 * two-overload signature independently of the forthcoming translation UI. */
import type { Translator } from "./keys.gen";

declare const t: Translator;

t("relay.login.page.title");
t("wire.exec.error_line", { error: "fixture refusal" });

// @ts-expect-error Unknown keys must not become untyped dynamic envelopes.
t("relay.shell.not_yet_extracted");
// @ts-expect-error Parameterized messages require their arguments.
t("wire.exec.error_line");
// @ts-expect-error ICU argument types come from the source catalogue parser.
t("wire.exec.error_line", { error: 42 });
// @ts-expect-error Bare messages cannot accept parameters.
t("relay.login.page.title", { extra: "not allowed" });
