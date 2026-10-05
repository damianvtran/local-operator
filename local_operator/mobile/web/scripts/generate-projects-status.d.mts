/**
 * Types for `generate-projects-status.mjs`.
 *
 * The script is plain ESM JavaScript that Node runs directly (`pnpm
 * gen-projects-status`), so it carries no TypeScript of its own. Its `--check`
 * path is also the one `src/projects-status.test.tsx` asserts against — the test
 * imports `readStatusOrder`/`renderModule` rather than re-implementing the
 * parse, because a second copy of one rule is how the two drift. This file is
 * what makes that import typed instead of `any`.
 */

/** The Python module that owns the board order. */
export declare const PY_SOURCE: string;
/** The committed generated module. */
export declare const OUT: string;
/** The ordered status list read out of one Python source text. Throws when the
    `STATUS_RANK` dict is missing, unbalanced or empty. */
export declare function readStatusOrder(pySource: string): string[];
/** The generated module's text for one ordered status list. */
export declare function renderModule(order: string[]): string;
