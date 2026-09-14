/**
 * The one typed bridge failure, in a module of its own.
 *
 * This looks like an unnecessary indirection and is not one. `cdp.ts` owns
 * `BridgeCommandError` historically, but it also registers a
 * `chrome.debugger.onDetach` listener at MODULE level, so any module that
 * imports the class from `./cdp` drags the whole CDP layer into its bundle.
 * That was invisible until `settle.ts` (the per-call `deadline` helper) and
 * `state.ts` (which bounds its storage awaits) started needing the class in
 * common: the resulting `cdp → settle → cdp` cycle reordered module evaluation
 * and made a storage-only bundle touch `chrome.debugger` — which the pure
 * storage/grouping unit harnesses do not provide, so they failed at load time
 * with `Cannot read properties of undefined (reading 'onDetach')`.
 *
 * Keeping the error value here breaks the cycle at its root: `errors.ts`
 * imports nothing, `settle.ts` and `state.ts` reach the class without reaching
 * `cdp.ts`, and `cdp.ts` still re-exports it so every existing
 * `from "./cdp"` import site keeps working unchanged.
 */
export class BridgeCommandError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly data: Record<string, unknown> = {},
  ) {
    super(message);
  }
}
