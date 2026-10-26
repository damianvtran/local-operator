import { useScrollCues } from "../../lib/use-scroll-cues";
/** Destinations leave the conversation surface, not the column. This drawer
 * shares Sheet's modal layer (including opening-press suppression) and never
 * portals to body: viewport pinning and phone capture depend on that ancestry.
 * Items are data, so capability-owned destinations can join without another
 * navigation layout. Missing capabilities render no disabled placeholder. */
import { useId, type ReactNode } from "react";
import { columnBox } from "../../lib/column";
import { messageRuntime } from "../../store";
import { useModalLayer } from "./use-modal-layer";

export interface DrawerItem {
	id: string;
	icon: ReactNode;
	label: string;
	count?: number;
	href?: string;
	onSelect?: () => void;
	available?: boolean;
	current?: boolean;
	preference?: boolean;
}

export function Drawer({ open, onClose, items, status }: {
	open: boolean;
	onClose: () => void;
	items: DrawerItem[];
	status: string;
}) {
	const titleId = useId();
	const scrollCue = useScrollCues();
	const { dialogRef, closeRef, focusEdge } = useModalLayer({ open, onClose });
	if (!open) return null;
	return <div id="relay-navigation" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby={titleId}
		className="inset-x-0 z-50 mx-auto w-full max-w-[var(--lo-column-max,28rem)]" style={columnBox()}>
		<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(true), 0)} />
		<button type="button" tabIndex={-1} aria-hidden data-testid="drawer-scrim"
			className="lo-scrim absolute inset-0 bg-scrim" onClick={onClose} />
		<nav aria-label={messageRuntime.t("relay.chrome.drawer.label")}
			className="lo-drawer-panel absolute start-0 inset-y-0 flex w-[min(320px,85%)] flex-col bg-surface px-3 pb-[max(env(safe-area-inset-bottom),0.75rem)] pt-[max(env(safe-area-inset-top),0.5rem)] shadow-overlay">
			<div className="flex min-h-12 items-center gap-2">
				<span id={titleId} className="flex-1 px-3 text-heading font-semibold">{messageRuntime.t("relay.chrome.drawer.label")}</span>
				<button ref={closeRef} type="button" onClick={onClose} aria-label={messageRuntime.t("relay.chrome.drawer.close")}
					className="flex size-11 items-center justify-center rounded-md text-ink-muted active:bg-elevated">✕</button>
			</div>
			<div ref={scrollCue} className="lo-scroll lo-fade-y min-h-0 flex-1 overflow-y-auto">
				{items.filter(item => item.available !== false).map((item, i, visible) => {
					const classes = `flex min-h-12 w-full items-center gap-3 rounded-md px-3 text-start text-body ${item.current ? "bg-accent-wash text-accent" : "text-ink active:bg-elevated"} ${item.preference && !visible[i - 1]?.preference ? "mt-4" : ""}`;
					const content = <><span aria-hidden className="flex size-5 shrink-0 items-center justify-center text-ink-muted">{item.icon}</span><span className="min-w-0 flex-1">{item.label}</span>{item.count !== undefined ? <span className="text-meta tabular-nums text-ink-dim">{item.count}</span> : null}</>;
					return item.href ? <a key={item.id} href={item.href} className={classes} aria-current={item.current ? "page" : undefined} onClick={onClose}>{content}</a>
						: <button key={item.id} type="button" className={classes} aria-current={item.current ? "page" : undefined} onClick={() => { onClose(); item.onSelect?.(); }}>{content}</button>;
				})}
			</div>
			<p className="px-3 pt-4 text-meta text-ink-dim">{status}</p>
		</nav>
		<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(false), 0)} />
	</div>;
}
