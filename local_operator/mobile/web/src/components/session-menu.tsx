/**
 * Session menu — the session header's overflow door.
 *
 * WHY A MENU. The header used to carry four controls on a 390-px screen:
 * back, a `★` pin toggle, an unlabelled `? N` asks entry, the bordered `wide`
 * mode toggle, and a word-shaped `approvals` button. Measured on the operator's
 * phone, `? 4` and `[wide]` had no label a first-time reader could learn from,
 * and `approvals` read as a noun where a state or an action belongs. Each was
 * also a second door: the ask dock above the composer already says
 * "N questions waiting · <question>", and the gate is a per-session setting,
 * not a destination worth a permanent header slot. So the header keeps what a
 * reader needs at a glance — back, the title, `needs you` only while a gate
 * actually blocks — and the rest move here, each on a full-width row that
 * names itself and states its current value.
 *
 * The sheet is the app's existing bottom sheet (`ui/sheet.tsx`), the same
 * idiom as the model and gate sheets: it traps focus, closes on Escape and
 * scrim, and restores focus to the opener.
 *
 * THE APPROVALS ROW CLOSES THIS SHEET BEFORE OPENING THE GATE SHEET. Two
 * dialogs would leave the second layered over the first with an inert column
 * between them; the gate sheet is the destination, so this one steps aside
 * first and the gate sheet then owns focus.
 */
import { useEffect, useState, type RefObject } from "react";
import { applyWideView, getWideView } from "../lib/viewport";
import { Sheet } from "./ui/sheet";

/** One full-width menu row: the 44px touch floor the header controls use. */
const MENU_ROW =
	"flex min-h-11 w-full items-center gap-2 rounded-sm px-3 text-left text-body active:bg-surface";

export function SessionMenu({
	open,
	onClose,
	pinned,
	onTogglePin,
	onOpenApprovals,
	returnFocusRef,
}: {
	open: boolean;
	onClose: () => void;
	/** The list store's pin state for this session, mark included (the same
	    value the header's old ★ read), so the row cannot disagree with the
	    session list. */
	pinned: boolean;
	/** The header's own optimistic mark/refusal path — this row does not own
	    a second pin implementation. */
	onTogglePin: () => void;
	/** Summon the existing gate sheet. The menu closes first. */
	onOpenApprovals: () => void;
	/** The ⋮ button, so closing returns focus to the control that opened it. */
	returnFocusRef?: RefObject<HTMLElement | null>;
}) {
	/* Seeded from the persisted preference on mount and re-read on open: the
	   preference can be changed from the conversation list while this screen is
	   not mounted, and the row must state the truth when it appears. */
	const [wide, setWide] = useState(getWideView);
	useEffect(() => {
		if (open) setWide(getWideView());
	}, [open]);
	return (
		<Sheet open={open} onClose={onClose} title="Session" returnFocusRef={returnFocusRef}>
			<div className="flex flex-col gap-1 p-2">
				<button
					type="button"
					onClick={onTogglePin}
					aria-pressed={pinned}
					/* The old ★ control's exact labels, kept so the pin's accessible
					   name did not change when the control moved. */
					aria-label={pinned ? "unpin this session" : "pin this session"}
					className={MENU_ROW}
				>
					<span className="min-w-0 flex-1 truncate">
						{pinned ? "Unpin session" : "Pin session"}
					</span>
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{pinned ? "pinned" : "not pinned"}
					</span>
				</button>
				<button
					type="button"
					onClick={() => {
						const next = !wide;
						applyWideView(next);
						setWide(next);
					}}
					aria-pressed={wide}
					aria-label="wide view"
					className={MENU_ROW}
				>
					<span className="min-w-0 flex-1 truncate">Wide view</span>
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{wide ? "on" : "off"}
					</span>
				</button>
				<button
					type="button"
					onClick={() => {
						/* Close first: the gate sheet is the destination, and two
						   stacked dialogs would leave the column inert twice over. */
						onClose();
						onOpenApprovals();
					}}
					className={MENU_ROW}
				>
					<span className="min-w-0 flex-1 truncate">Approvals…</span>
					<span aria-hidden className="shrink-0 text-ink-dim">
						▸
					</span>
				</button>
			</div>
		</Sheet>
	);
}
