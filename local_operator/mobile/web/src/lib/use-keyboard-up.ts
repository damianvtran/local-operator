import { useEffect, useState } from "react";

/** How much shorter the visual viewport must be than the layout viewport before
 *  the page counts as having an on-screen keyboard up.
 *
 *  A JUDGEMENT, not a measurement: a virtual keyboard is an overlay that shrinks
 *  `visualViewport.height` and leaves `innerHeight` alone (index.html sets no
 *  `interactive-widget`, so the default `resizes-visual` applies; see
 *  `screens/session-view.tsx`), and the smallest phone keyboards are well over
 *  200 px tall. Browser chrome sliding away changes the gap by well under this,
 *  so 120 px separates the two with room on both sides. */
export const KEYBOARD_MIN_PX = 120;

/** Whether an on-screen keyboard looks to be up, from the `visualViewport` the
 *  session column already pins itself to. False where the browser has no
 *  `visualViewport` (and in tests that do not stub one): a surface that cannot
 *  say a keyboard is up keeps showing everything, which is the safe side. */
export function useKeyboardUp(): boolean {
	const read = () => {
		const vv = typeof window === "undefined" ? undefined : window.visualViewport;
		// A pinch-zoom shrinks `vv.height` by the zoom factor with no keyboard in
		// play, so the gap is measured against the zoomed height: at scale 2 a
		// 422 px visual viewport on an 844 px page is no gap at all.
		return vv ? window.innerHeight - vv.height * (vv.scale || 1) > KEYBOARD_MIN_PX : false;
	};
	const [up, setUp] = useState(read);
	useEffect(() => {
		const vv = window.visualViewport;
		if (!vv) return;
		const sync = () => setUp(read());
		sync();
		vv.addEventListener("resize", sync);
		return () => vv.removeEventListener("resize", sync);
		// `read` closes over nothing that changes.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);
	return up;
}
