/** Page-side native-select operation, serialized as a FIXED function through CDP.
 *
 * Unlike hover/key presses this needs no compositor input: setting option state
 * through native prototype setters and emitting input/change works on the
 * extension's intentionally inactive tab, just like typeText. Never opens the
 * OS popup or focuses a window. All requested options are resolved/validated
 * before any setter runs, so a typo/disabled option cannot partly change a form.
 *
 * This function must stay self-contained: its source runs in the target node's
 * realm, including for a same-origin iframe, and imports/worker globals do not
 * exist there. Errors are data (not thrown page exceptions), preserving the
 * existing wire ErrorCode vocabulary across older daemons.
 */
export function selectNative(this: Element, requested: Array<string | number>): Record<string, unknown> {
  const win = this.ownerDocument.defaultView!;
  const fail = (reason: string, message: string, data: Record<string, unknown> = {}) => ({
    error: { code: "element_not_found", message, data: { reason, ...data } },
  });
  if (!(this instanceof win.HTMLSelectElement)) {
    // Native controls are safely driveable without Input.*. ARIA controls need
    // an application-specific popup association, click and verification flow;
    // this extension does not pretend native setters implement those semantics.
    return { error: { code: "capability_unsupported", message: "ARIA selection requires the desktop app's browser; open with backend='ui'", data: { reason: "aria_select_unsupported" } } };
  }
  const select = this as HTMLSelectElement;
  if (select.disabled || select.matches(":disabled")) {
    return fail("select_disabled", "the select is disabled; nothing was changed");
  }
  if (requested.length === 0 || (!select.multiple && requested.length !== 1)) {
    return fail("invalid_options", "options must be non-empty; a single select needs exactly one option; nothing was changed");
  }
  const options = Array.from(select.options);
  const label = (option: HTMLOptionElement) => option.label.trim();
  const describe = (option: HTMLOptionElement, index: number) => ({ index, value: option.value, label: label(option) });
  const available = () => options.slice(0, 50).map((option, index) => ({
    index, label: label(option).slice(0, 120), value: option.value.slice(0, 120),
  }));
  const chosen: number[] = [];
  for (const entry of requested) {
    // Exact trimmed label, then case-insensitive label, then value; DOM order
    // breaks duplicate-label ties consistently with the desktop contract.
    const wanted = typeof entry === "string" ? entry.trim() : entry;
    let index = typeof wanted === "number" ? wanted : options.findIndex((o) => label(o) === wanted);
    if (typeof wanted === "string" && index < 0) index = options.findIndex((o) => label(o).toLowerCase() === wanted.toLowerCase());
    if (typeof wanted === "string" && index < 0) index = options.findIndex((o) => o.value === entry);
    if (!Number.isInteger(index) || index < 0 || index >= options.length) {
      return fail("option_not_found", "no option matches the requested label/value/index; nothing was changed", { available: available() });
    }
    const option = options[index]!;
    if (option.disabled || (option.parentElement?.tagName === "OPTGROUP" && (option.parentElement as HTMLOptGroupElement).disabled)) {
      return fail("option_disabled", "the requested option is disabled; nothing was changed", { index });
    }
    if (!chosen.includes(index)) chosen.push(index);
  }
  // Bypass framework-instrumented OWN setters, as the existing type path does.
  // For a single select, set value through the prototype, then pin the exact
  // selected option too (values can be duplicated; index matching must be exact).
  const setSelected = Object.getOwnPropertyDescriptor(win.HTMLOptionElement.prototype, "selected")!.set!;
  if (!select.multiple) {
    Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, "value")!.set!.call(select, options[chosen[0]!]!.value);
  }
  for (let index = 0; index < options.length; index++) setSelected.call(options[index], chosen.includes(index));
  // Capture intended values BEFORE handlers can detach or mutate these nodes.
  // This is apply evidence, never verification: controlled frameworks can replace
  // the select during input/change, leaving this object perfectly readable but
  // no longer representing the form. The command re-resolves the live target.
  const expected = chosen.map((index) => describe(options[index]!, index));
  select.dispatchEvent(new win.Event("input", { bubbles: true }));
  select.dispatchEvent(new win.Event("change", { bubbles: true }));
  return { kind: "native", expected, selected: [], verified: false };
}

/** Read-only second phase on the freshly resolved LIVE node, not saved options.
 * Must stay self-contained for Runtime.callFunctionOn, just like selectNative.
 * A ref that still points at a detached node is explicitly unverified; a CSS/AX
 * target can instead resolve its replacement and report that node's real state.
 */
export function readNativeSelection(
  this: Element, expected: Array<{ index: number; value: string; label: string }>,
): Record<string, unknown> {
  const win = this.ownerDocument.defaultView!;
  if (!this.isConnected || !(this instanceof win.HTMLSelectElement)) {
    return { kind: "native", selected: [], verified: false };
  }
  const options = Array.from((this as HTMLSelectElement).options);
  const selected = options.flatMap((option, index) => option.selected ? [{
    index, value: option.value, label: option.label.trim(),
  }] : []);
  const verified = selected.length === expected.length && selected.every((option) =>
    expected.some((wanted) => wanted.index === option.index && wanted.value === option.value && wanted.label === option.label),
  );
  return { kind: "native", selected, verified };
}
