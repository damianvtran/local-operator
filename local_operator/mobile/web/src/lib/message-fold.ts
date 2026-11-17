/**
 * The press signal a message's own "Show more / Show less" control sends up to
 * the transcript scroller BEFORE the message changes height.
 *
 * WHY AN EVENT. The scroller owns the reader's place (`pinnedRef`, the anchor
 * follow, `[overflow-anchor:none]`), and a row deep in the tree has no handle on
 * any of it. The bar's own toggle can reach the scroller through a prop because
 * the transcript renders it; a user bubble is rendered by `Entry`, a plain
 * function several calls away. A bubbling event is the same shape the repo
 * already uses for "a row changed height after it painted"
 * (`lib/content-resized.ts`), and it keeps `Entry` free of scroller props.
 *
 * `detail.id` is the pressed control's `data-message-toggle` value. The target
 * is the control itself, so the listener reads the control's top before the
 * toggle moves it, then holds that top through the commit (the pressed-header
 * recipe the condensed bars already use).
 */
export const MESSAGE_PRESSED_EVENT = "lo:message-pressed";

/** How many wrapped lines a condensed turn shows of the person's own message. */
export const MESSAGE_PREVIEW_LINES = 3;
