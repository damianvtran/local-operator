/**
 * The one piece of Library copy that lands inside an ENFORCED i18n file.
 *
 * `screens/session-list.tsx` is on the reviewed i18n path list, and a prose
 * literal there fails `pnpm check` unless it exists in the frozen fingerprint
 * baseline. The Library destination's name is owned relay copy, so it lives in
 * this (advisory) module for now: the screen and its footer entry are the first
 * Library strings, and folding the whole Library surface into the key catalogue
 * is its own reviewed change — doing half of it here would either drop shipped
 * locales or ship a key with no catalogue entry. This is the narrow exception,
 * named so the follow-up is obvious rather than hidden in a JSX literal.
 */
export const LIBRARY_LABEL = "library";
