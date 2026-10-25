/** Outcomes float above the thumb action rather than changing list geometry.
 * The caller owns the receipt (some survive navigation); this view owns only
 * its four-second dismissal and swipe accelerator. The close button remains
 * the accessible path, and reduced motion is handled by the lo-snack CSS. */
import { useEffect, useRef } from "react";
import { messageRuntime } from "../../store";

export function Snackbar({ text, danger = false, onDismiss }: {
	text: string;
	danger?: boolean;
	onDismiss: () => void;
}) {
	const dismiss = useRef(onDismiss);
	dismiss.current = onDismiss;
	const origin = useRef<{ x: number; y: number } | null>(null);
	useEffect(() => {
		if (!text) return;
		let timer: ReturnType<typeof setTimeout> | undefined;
		const arm = () => {
			clearTimeout(timer);
			if (document.visibilityState !== "hidden") timer = setTimeout(() => dismiss.current(), 4000);
		};
		arm();
		document.addEventListener("visibilitychange", arm);
		return () => { clearTimeout(timer); document.removeEventListener("visibilitychange", arm); };
	}, [text]);
	if (!text) return null;
	return <div className="lo-snack absolute inset-x-4 bottom-[calc(5.5rem+env(safe-area-inset-bottom))] z-20 flex items-center gap-2 rounded-md bg-elevated ps-4 pe-1 py-1 shadow-overlay"
		onPointerDown={event => { origin.current = { x: event.clientX, y: event.clientY }; }}
		onPointerUp={event => {
			const start = origin.current;
			origin.current = null;
			if (start && Math.abs(event.clientX - start.x) > 64 && Math.abs(event.clientY - start.y) < 32) onDismiss();
		}}>
		<span role={danger ? "alert" : "status"} className={`min-w-0 flex-1 text-body-sm ${danger ? "text-danger" : "text-ink"}`}>{text}</span>
		<button type="button" aria-label={messageRuntime.t("relay.chrome.snackbar.dismiss")} onClick={onDismiss}
			className="flex size-11 shrink-0 items-center justify-center rounded-md text-ink-muted active:bg-surface">×</button>
	</div>;
}
