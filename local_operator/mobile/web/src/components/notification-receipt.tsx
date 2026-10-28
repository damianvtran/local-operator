/**
 * The inbox's receipt bar: the outcome of one write, with Undo.
 *
 * WHY NOT `ui/snackbar.tsx`: that primitive carries no action, and it belongs
 * to the borderless-redesign lane, which this lane was asked not to edit. This
 * bar keeps its geometry and classes (the same `lo-snack` float above the thumb
 * action, the same 44 px close) and adds the one thing the operator asked for
 * here: an exact receipt that can be taken back. Design §9 gives it 6 s rather
 * than the snackbar's 4 s because it carries a decision (Undo), not a notice.
 * If the shared primitive later gains an action slot, this file should fold
 * into it rather than drift beside it.
 */
import { useEffect, useRef } from "react";
import { messageRuntime } from "../store";

const UNDO_WINDOW_MS = 6000;

export function NotificationReceipt({ text, danger = false, onUndo, onDismiss }: {
	text: string;
	danger?: boolean;
	/** Absent when nothing changed (there is nothing to take back). */
	onUndo?: () => void;
	onDismiss: () => void;
}) {
	const dismiss = useRef(onDismiss);
	dismiss.current = onDismiss;
	useEffect(() => {
		if (!text) return;
		let timer: ReturnType<typeof setTimeout> | undefined;
		/* The countdown pauses while the page is hidden, the snackbar's rule: a
		   receipt nobody could see must not expire its Undo. */
		const arm = () => {
			clearTimeout(timer);
			if (document.visibilityState !== "hidden") timer = setTimeout(() => dismiss.current(), UNDO_WINDOW_MS);
		};
		arm();
		document.addEventListener("visibilitychange", arm);
		return () => { clearTimeout(timer); document.removeEventListener("visibilitychange", arm); };
	}, [text]);
	if (!text) return null;
	return <div className="lo-snack absolute inset-x-4 bottom-[calc(5.5rem+env(safe-area-inset-bottom))] z-20 flex items-center gap-1 rounded-md bg-elevated ps-4 pe-1 py-1 shadow-overlay">
		<span role={danger ? "alert" : "status"} className={`min-w-0 flex-1 text-body-sm ${danger ? "text-danger" : "text-ink"}`}>{text}</span>
		{onUndo ? <button type="button" onClick={onUndo}
			className="min-h-11 shrink-0 rounded-md px-3 text-body-sm font-semibold text-accent active:bg-surface">{messageRuntime.t("relay.notifications.receipt.undo")}</button> : null}
		<button type="button" aria-label={messageRuntime.t("relay.chrome.snackbar.dismiss")} onClick={onDismiss}
			className="flex size-11 shrink-0 items-center justify-center rounded-md text-ink-muted active:bg-surface">×</button>
	</div>;
}
