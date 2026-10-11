/**
 * One generated component in its sandboxed frame (turn-supplements memo
 * §4.1/§4.2, lane R).
 *
 * THE INVARIANT, and nothing beyond it: `<iframe sandbox="allow-scripts" …>` —
 * an opaque origin, no bridge/preload, no network (the document's own CSP
 * closes `connect-src`), no file access, no popups. The navigation control on
 * this host has no interception point (no Electron guard, no native
 * `onShouldStartLoadWithRequest`), so it is exactly the memo's relay pair:
 *
 *   - the EMBEDDING document's `frame-src data:` policy, declared as a
 *     `<meta http-equiv>` in `index.html` (a response header is overwritten
 *     in transit by the tunnel gateway, §2.7) — Chromium checks every
 *     navigation of an already-loaded child frame and blocks it before the
 *     request is issued; and
 *   - THIS component counts `load` events: the second `load` is a hostile
 *     navigation or a blocked attempt (measured on Chrome 155: a blocked
 *     navigation fires a second `load` when it is attempted after the
 *     frame's first `load`; attempted during the initial parse it does not,
 *     and the ping watchdog's `no-pong` ends that attempt instead — QA
 *     round 1, Q2), so the frame is torn down into the frame-level fallback
 *     line and every later message from it is dropped.
 *
 * THE MESSAGE WIRE (host side) is exactly the contract's: the host sends
 * `{lo:"supplement-host", t:"theme", mode, vars, nonce}` (the nonce minted per
 * frame, carried in the FIRST push) and `{t:"ping"}`; it accepts
 * `{lo:"supplement", v:1, t:"ready"|"resize"|"error"|"pong"}` — with the nonce
 * echoed on every state-moving type — from `frame.contentWindow` at
 * `Origin: "null"` only, and clamps every height to the contract's [120, 480]
 * range regardless of what arrived.
 *
 * LIVENESS: at most one `resize`/`error` per animation frame (latest wins),
 * and a watchdog — a frame that never posts `ready`, or that fails to answer
 * a ping within 5 s, is unmounted into the frame-level fallback line. A frame
 * error NEVER rewrites the row: it replaces this frame only.
 *
 * EVERYTHING HERE IS PER MOUNT. The nonce, the load counter, the watchdog and
 * its timers live in the component instance (refs), never at module scope:
 * two components on one page are two independently bounded frames, and one
 * frame's timeout must not tear down another's.
 *
 * WHAT THIS HOST DOES NOT HAVE, BY RULE. There is no arm channel and no
 * custom scheme: the phone app hosting this page injects no bridge, so the
 * bytes always ride the `data:` URL (module header of `lib/supplement.ts`,
 * rule 1). A future addition that gave this frame a privileged channel would
 * be a bridge and needs the security round, not a prop.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
	PING_INTERVAL_MS,
	WATCHDOG_MS,
	acceptFrameMessage,
	clampFrameHeight,
	clampedHeightHint,
	mintNonce,
	readSupplementTheme,
	sourceLine,
	subscribeSupplementTheme,
	supplementFrameUrl,
	type SupplementComponent,
	type SupplementTheme,
} from "../lib/supplement";

/** The debug line a torn-down frame logs once ("one debug line", §2.8). */
const HOSTILE_TEARDOWN_LOG =
	"[supplement] frame torn down after a second load (hostile navigation or a blocked attempt)";

/**
 * The nearest scrollable ancestor, or null when the frame is not inside one
 * (an isolated fixture). The resize gate needs it: a frame growing ABOVE the
 * reader's scroll anchor moves the reader's content (§2.11 — "a resize applies
 * … only when the row is below the viewport's scroll anchor"), so a growth
 * whose row starts above the scroll port's visible top is held and re-applied
 * on the next scroll instead of moving what the reader is looking at. The
 * transcript's own scroller qualifies (`overflow-y-auto`).
 */
function nearestScrollPort(element: HTMLElement): HTMLElement | null {
	let node: HTMLElement | null = element.parentElement;
	while (node) {
		const style = window.getComputedStyle(node);
		if (
			(style.overflowY === "auto" || style.overflowY === "scroll") &&
			node.scrollHeight > node.clientHeight
		) {
			return node;
		}
		node = node.parentElement;
	}
	return null;
}

export function SupplementFrame({
	component,
	documentHtml,
	documentFailed = false,
	onRetry,
	onHeightChange,
}: {
	component: SupplementComponent;
	/**
	 * The assembled document, `null` until the relay route answers. The
	 * reserved box holds the component's clamped `height_hint` while it is
	 * null.
	 */
	documentHtml: string | null;
	/**
	 * The document could not be fetched (a pruned blob, a refusal, a malformed
	 * body): the frame paints the frame-level fallback line instead of holding
	 * its reserved box forever.
	 */
	documentFailed?: boolean;
	/**
	 * The row's `supplement_restart` arm, offered as `· Retry` beside the
	 * fallback line when the session is addressable (§2.8). Absent handler,
	 * absent control.
	 */
	onRetry?: () => void;
	/** The frame's current rendered height, on every change — the block's
	 *  aggregate budget accrues these once a frame is on screen. */
	onHeightChange?: (height: number) => void;
}) {
	const reserved = clampedHeightHint(component);
	const [height, setHeight] = useState(reserved);
	const [failed, setFailed] = useState(false);
	/**
	 * Whether the frame has yet to paint anything: true from mount until the
	 * document's first `resize` beat. While true, the reserved box carries the
	 * state line's own shimmer at low intensity (§2.8 round-1 D4): the box
	 * used to paint nothing between mount and first paint — up to the 5 s
	 * watchdog — and on a busy transcript that read as a hole under the answer.
	 */
	const [awaitingPaint, setAwaitingPaint] = useState(true);
	const boxRef = useRef<HTMLDivElement | null>(null);
	const frameRef = useRef<HTMLIFrameElement | null>(null);
	const nonceRef = useRef("");
	const failedRef = useRef(false);
	/**
	 * WHY a frame fell back, carried to the DOM as the fallback element's
	 * attribute value: six distinct arms paint that line and they have
	 * completely different causes — `document` is a missing blob, `no-ready`
	 * is a hung document, `second-load` is a navigation, `frame-error` is the
	 * document's own report. Diagnosing a fallback from a screenshot alone
	 * cost the desktop lane a bisect; one attribute ends that here too.
	 */
	const failedReasonRef = useRef<string>("");
	/** Starts the ping cadence; armed by the mount effect, called on the first
	 *  `load` immediately after the theme push (see that effect for why the
	 *  cadence may not start earlier). */
	const startPingsRef = useRef<() => void>(() => {});
	const loadCountRef = useRef(0);
	const heightRef = useRef(reserved);
	/**
	 * The newest height a resize carried while the row sat ABOVE the scroll
	 * port; re-applied on the next scroll so a message that arrived while held
	 * is not lost.
	 */
	const pendingHeightRef = useRef<number | null>(null);
	/** What the element handlers call, reassigned each render to fresh closures. */
	const onLoadRef = useRef<() => void>(() => {});
	const onHeightChangeRef = useRef<((height: number) => void) | undefined>(onHeightChange);
	const themePushRef = useRef<(theme: SupplementTheme) => void>(() => {});

	const fail = useCallback((reason: string) => {
		if (failedRef.current) return;
		failedRef.current = true;
		failedReasonRef.current = reason;
		setFailed(true);
	}, []);

	/*
	 * ARM, WIRE, THEN NAVIGATE — one effect, and the order inside it is the
	 * point.
	 *
	 * The listeners and the watchdog are attached BEFORE the frame is given a
	 * `src`. That ordering is not tidiness: a `data:` document begins
	 * executing in a task the browser may interleave before React's passive
	 * effects run, so a frame whose `src` is set during the commit can post
	 * its `ready` broadcast into a page that is not listening yet — and since
	 * the prelude sends `ready` exactly once, the mount then dies at the boot
	 * deadline while the document is alive and painting (measured on the
	 * desktop lane's Storybook captures, 2026-10-11: the chart drew while the
	 * host counted the frame dead). Assigning `src` from inside the armed
	 * effect closes that window by construction, which is why the attribute is
	 * never set in JSX and lives only on this line.
	 */
	useEffect(() => {
		if (documentHtml === null) return;
		/*
		 * A NEW DOCUMENT STARTS A NEW LOAD COUNT. The mount effect below re-runs
		 * for a new `documentHtml` on the SAME frame instance, and the counter
		 * must not carry over: without this reset, the new document's first
		 * `load` counts as the hostile SECOND load and tears down a frame the
		 * host itself mounted (agent review round 1, R3 — instances are keyed
		 * by attachment today, so this is future-proofing, not a live defect).
		 */
		loadCountRef.current = 0;
		const frameEl = frameRef.current;
		if (!frameEl) return;
		nonceRef.current = mintNonce();
		const url = supplementFrameUrl(documentHtml);
		let live = false;
		let pingsStarted = false;
		let pingTimer: number | null = null;
		let pongDeadline: number | null = null;
		let raf: number | null = null;
		const coalesced: { resize: number | null; error: boolean } = {
			resize: null,
			error: false,
		};
		const bootDeadline = window.setTimeout(() => {
			// A frame that never posted `ready` is not a live document (§4.2).
			if (!live) fail("no-ready");
		}, WATCHDOG_MS);

		const post = (message: Record<string, unknown>) => {
			frameEl.contentWindow?.postMessage(message, "*");
		};
		const sendPing = () => {
			/*
			 * A FAILED frame is not pinged: this cadence belongs to a live
			 * document, and a torn-down frame has none to answer with — the
			 * interval used to keep arming a 5 s timeout per cycle against a
			 * detached `contentWindow` until the row unmounted (round-1 R6).
			 */
			if (failedRef.current) return;
			// ONE outstanding ping at a time, never coalesced away (§4.2).
			if (pongDeadline !== null) return;
			post({ lo: "supplement-host", t: "ping" });
			pongDeadline = window.setTimeout(() => {
				pongDeadline = null;
				fail("no-pong");
			}, WATCHDOG_MS);
		};
		/**
		 * The ping cadence starts WITH THE THEME PUSH, not at `ready`. A
		 * `pong` is nonce-required, and the nonce only reaches the frame inside
		 * the theme message — so a ping sent at `ready` (which precedes that
		 * message) is answered by a pong this host must reject, and the pong
		 * deadline would then fail a live frame five seconds later. `onLoad`
		 * calls this immediately after the first theme push, which is the
		 * earliest moment the frame can answer with the nonce.
		 */
		startPingsRef.current = () => {
			if (pingsStarted || failedRef.current) return;
			pingsStarted = true;
			sendPing();
			pingTimer = window.setInterval(sendPing, PING_INTERVAL_MS);
		};

		const applyHeight = (next: number) => {
			const box = boxRef.current;
			if (box) {
				const port = nearestScrollPort(box);
				if (port && box.getBoundingClientRect().top < port.getBoundingClientRect().top) {
					pendingHeightRef.current = next;
					onHeightChangeRef.current?.(next);
					return;
				}
			}
			pendingHeightRef.current = null;
			onHeightChangeRef.current?.(next);
			if (next !== heightRef.current) {
				heightRef.current = next;
				setHeight(next);
			}
		};
		const scheduleCoalesced = () => {
			if (raf !== null) return;
			raf = window.requestAnimationFrame(() => {
				raf = null;
				const { resize, error } = coalesced;
				coalesced.resize = null;
				coalesced.error = false;
				if (error) {
					fail("frame-error");
					return;
				}
				if (resize !== null) applyHeight(resize);
			});
		};

		const onMessage = (event: MessageEvent) => {
			if (failedRef.current) return;
			// The browsing context first (`source`), then the opaque origin
			// every sandboxed document presents, then the contract's
			// shape+nonce rule.
			if (event.source !== frameEl.contentWindow) return;
			if (event.origin !== "null") return;
			/*
			 * THE LOAD COUNTER IS DELIBERATELY NOT A GATE HERE. The prelude
			 * posts `ready` during parsing — before the first `load` event —
			 * so requiring "the first load has fired" drops exactly the
			 * message the boot deadline waits for, and a live frame dies at
			 * five seconds (measured on the desktop lane, 2026-10-11).
			 * Navigation control lives where the design puts it: the parent
			 * page's `frame-src data:` refuses the navigation, and the SECOND
			 * load tears the frame down (after which this component renders no
			 * iframe at all and no message can be answered by it).
			 */
			const message = acceptFrameMessage(event.data, nonceRef.current);
			if (!message) return;
			if (message.t === "ready") {
				if (!live) {
					live = true;
					window.clearTimeout(bootDeadline);
				}
				return;
			}
			if (message.t === "pong") {
				if (pongDeadline !== null) {
					window.clearTimeout(pongDeadline);
					pongDeadline = null;
				}
				return;
			}
			if (message.t === "resize") {
				coalesced.resize = clampFrameHeight(message.h as number);
				// The document painted: the reserved box owes its shimmer no
				// longer (the shimmer's own reduced-motion handling is the
				// animation's, not a second rule here).
				setAwaitingPaint(false);
				scheduleCoalesced();
				return;
			}
			// `error`: the frame's own report. One quiet line replaces it; the
			// row is never rewritten (§2.8).
			coalesced.error = true;
			scheduleCoalesced();
		};
		const onScroll = () => {
			const pending = pendingHeightRef.current;
			if (pending !== null) applyHeight(pending);
		};
		window.addEventListener("message", onMessage);
		// Capture: the scroll that clears the hold happens on the transcript's
		// own scroller, which does not bubble a scroll event to the window.
		window.addEventListener("scroll", onScroll, true);

		/* ---- the wire is live; now the document may start ---- */
		frameEl.src = url;

		return () => {
			window.clearTimeout(bootDeadline);
			if (pingTimer !== null) window.clearInterval(pingTimer);
			if (pongDeadline !== null) window.clearTimeout(pongDeadline);
			if (raf !== null) window.cancelAnimationFrame(raf);
			window.removeEventListener("message", onMessage);
			window.removeEventListener("scroll", onScroll, true);
			startPingsRef.current = () => {};
		};
	}, [documentHtml, fail, component.attachment]);

	/*
	 * The RENDERED height is reported the moment a resize beat is APPLIED (or
	 * held for the scroll port — a held height is still the rendered one): the
	 * block's aggregate budget accrues these heights instead of trusting the
	 * write-time hints. Reporting is deliberately NOT tied to the paint flag
	 * or to the `height` state: the first accepted resize flips `awaitingPaint`
	 * while `height` can still hold the RESERVED value, and a 120-hint frame
	 * vouching for itself as "measured 120" mounted the next frame under a
	 * budget that had accrued nothing (both defects were read off the desktop
	 * lane's re-captured frames before this spelling replaced them).
	 */
	onHeightChangeRef.current = onHeightChange;

	/*
	 * The theme channel. The FIRST push rides the frame's first `load` (the
	 * prelude installs its listener synchronously and stays hidden until a
	 * theme arrives, with a 400 ms prefers-color-scheme fallback as the last
	 * resort); every later change is one more push, never a reload.
	 */
	themePushRef.current = (theme: SupplementTheme) => {
		frameRef.current?.contentWindow?.postMessage(
			{
				lo: "supplement-host",
				t: "theme",
				mode: theme.mode,
				vars: theme.vars,
				nonce: nonceRef.current,
			},
			"*",
		);
	};
	useEffect(() => {
		return subscribeSupplementTheme((theme) => themePushRef.current(theme));
	}, []);

	onLoadRef.current = () => {
		const count = loadCountRef.current + 1;
		loadCountRef.current = count;
		if (count > 1) {
			// The second load is a hostile navigation OR a blocked attempt (a
			// blocked navigation fires a second `load` when the attempt lands
			// after the first load, measured on Chrome 155; a parse-time attempt
			// is reached by the ping watchdog instead — QA round 1, Q2). Either
			// way this frame is not the document the host mounted.
			console.debug(HOSTILE_TEARDOWN_LOG);
			fail("second-load");
			return;
		}
		themePushRef.current(readSupplementTheme());
		// The theme push is what delivers the nonce, so the ping cadence starts
		// here — see `startPingsRef`'s comment for why any earlier ping would
		// fail a live frame.
		startPingsRef.current();
	};

	if (failed || documentFailed) {
		/*
		 * The frame-level fallback line (§2.8): one quiet line where the
		 * graphic was, never a rewrite of the row's own state. The row's
		 * committed state decides which line renders — a job-level `failed`
		 * paints its own line, and components never mount under it — so this
		 * only ever replaces a frame on a settled block.
		 */
		return (
			<div
				className="text-meta text-ink-dim"
				data-supplement-frame-fallback={documentFailed ? "document" : failedReasonRef.current}
			>
				Couldn't render this graphic
				{/*
				 * `· Retry`, the row's own `supplement_restart` arm where the
				 * job is settled. Absent handler, absent control — the block
				 * wires nothing when no session is addressable.
				 *
				 * `px-2`: the word alone measured 30 px wide, under the surface's
				 * 44 px touch rule the audit sweeps for (D1/U1 — same fix as the
				 * state line's Retry).
				 */}
				{onRetry && (
					<>
						<span aria-hidden="true" className="px-1 text-ink-dim">
							·
						</span>
						<button
							type="button"
							className="min-h-11 px-2 text-ink-dim"
							data-supplement-frame-retry=""
							onClick={() => onRetry()}
						>
							Retry
						</button>
					</>
				)}
			</div>
		);
	}

	return (
		<div className="flex flex-col gap-1" data-supplement-frame-box="">
			{/*
			 * The reserved box holds its space while the document loads and
			 * draws no title (§2.8): the height is the write-time clamped
			 * `height_hint` until the first resize, which is already within the
			 * same [120, 480] range. While nothing has painted yet the box
			 * carries the state line's own shimmer at low intensity (§2.8
			 * round-1 D4): a bare transparent box read as a hole under the
			 * answer while the `Source:` caption sat painted below it.
			 */}
			<div
				ref={boxRef}
				data-supplement-frame=""
				className="relative overflow-hidden"
				style={{ height, width: "100%" }}
			>
				{documentHtml !== null && (
					<iframe
						ref={frameRef}
						title={component.title}
						sandbox="allow-scripts"
						/* `src` is assigned by the mount effect, AFTER the message
						   wire is armed — see that effect's header for the race it
						   closes. */
						onLoad={() => onLoadRef.current()}
						className="block h-full w-full border-0"
						// The document's own CSP is a `<meta>` inside the bytes (a
						// response header does not survive the tunnel, and the iframe
						// `csp` attribute is unimplemented in Chromium — §2.7/§4.1);
						// the sandbox above is the boundary and the parent page's
						// `frame-src data:` is the navigation control.
					/>
				)}
				{awaitingPaint && (
					/*
					 * The sweep over a STANDING faint fill, not the sweep alone:
					 * the shimmer's parked frame is off-tile by design
					 * (`styles/index.css`'s reduced-motion cap truncates to `to`),
					 * so a box whose only mark was the moving band would
					 * photograph — and park, for a reduced-motion reader — as an
					 * empty hole again. The fill is the low-intensity placeholder
					 * under it; both leave with the first rendered resize.
					 */
					<span
						aria-hidden="true"
						className="lo-shimmer-bar pointer-events-none absolute inset-0 bg-elevated/40"
					/>
				)}
			</div>
			{/*
			 * The one caption the host renders: exactly one `Source: …` line
			 * per component, `text-meta`/`ink-muted`, single line with the full
			 * text in the tooltip. The document prints none (§2.8, D13).
			 */}
			<span
				className="block truncate text-meta text-ink-muted"
				title={sourceLine(component.source)}
				data-supplement-source=""
			>
				{sourceLine(component.source)}
			</span>
		</div>
	);
}
