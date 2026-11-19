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

/** How many wrapped lines a condensed turn shows of a folded message. */
export const MESSAGE_PREVIEW_LINES = 3;

/**
 * How long a message must be, in wrapped lines at the reader's own width, before
 * a condensed turn folds it at all. Shorter messages show in full with no
 * control: the operator asked that the fold appear "only for very long messages,
 * so it doesn't appear all the time". Five or more hidden lines is what earns a
 * button; a message that is merely a paragraph is read in place.
 */
export const MESSAGE_FOLD_MIN_LINES = 8;
