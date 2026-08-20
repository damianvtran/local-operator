/**
 * Composer — the bottom-docked input cluster: textarea, send/steer, stop,
 * queued count, plus the three sheets it can raise (slash commands, model,
 * effort rungs) and the post-abort "resume" row.
 *
 * Ergonomics: the textarea renders at 16px so iOS does not zoom on focus;
 * it auto-grows to six lines then scrolls. While the session is streaming
 * the send button switches to the `steer` op (same action, different
 * command) and a stop button appears beside it.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { getCommands, sendCommand } from "../api";
import { Sheet } from "./ui/sheet";
import { cn } from "../lib/cn";
import { useDraft } from "../store";
import type { SessionProjection, SlashCommand } from "../types";


/** Detect "/cmd args" at the very start of the draft — the slash trigger. */
function slashQuery(text: string): string | null {
	if (!text.startsWith("/")) return null;
	if (text.includes("\n")) return null;
	const space = text.indexOf(" ");
	return (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
}

function SlashSheet({
	open,
	onClose,
	onPick,
}: {
	open: boolean;
	onClose: () => void;
	/** fill: text to place in the composer; submit: send immediately. */
	onPick: (fill: string, submit: boolean) => void;
}) {
	const [commands, setCommands] = useState<SlashCommand[]>([]);
	const [filter, setFilter] = useState("");

	useEffect(() => {
		if (!open) return;
		getCommands()
			.then((r) => setCommands(r.commands))
			.catch(() => setCommands([]));
	}, [open]);

	const filtered = useMemo(() => {
		const q = filter.trim().toLowerCase();
		if (!q) return commands;
		return commands.filter(
			(c) =>
				c.name.toLowerCase().includes(q) ||
				c.aliases.some((a) => a.toLowerCase().includes(q)) ||
				c.description.toLowerCase().includes(q),
		);
	}, [commands, filter]);

	return (
		<Sheet open={open} onClose={onClose} title="commands">
			<div className="flex flex-col gap-1 p-2">
				<input
					value={filter}
					onChange={(e) => setFilter(e.target.value)}
					placeholder="filter commands"
					spellCheck={false}
					autoCapitalize="off"
					autoCorrect="off"
					className="mb-1 min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
				/>
				{filtered.map((c) => (
					<button
						key={c.name}
						type="button"
						onClick={() => {
							if (c.arguments === "none") {
								onPick(`/${c.name}`, true);
							} else {
								onPick(`/${c.name} `, false);
							}
							onClose();
						}}
						className="flex min-h-11 items-center gap-2 rounded-sm px-3 text-left active:bg-surface"
					>
						<span className="shrink-0 font-mono text-mono-sm text-ink">
							/{c.name}
							{c.arguments === "required" ? (
								<span className="text-ink-dim"> …</span>
							) : null}
						</span>
						<span className="min-w-0 flex-1 truncate text-body-sm text-ink-dim">
							{c.description}
						</span>
					</button>
				))}
				{filtered.length === 0 ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						no matching commands
					</p>
				) : null}
			</div>
		</Sheet>
	);
}

function EffortSheet({
	open,
	onClose,
	projection,
}: {
	open: boolean;
	onClose: () => void;
	projection: SessionProjection;
}) {
	const set = async (effort: string) => {
		try {
			await sendCommand(projection.pid, { op: "set_effort", effort });
		} catch {
			/* The next repaint shows the truth; a failed set leaves it. */
		}
		onClose();
	};
	return (
		<Sheet open={open} onClose={onClose} title="effort">
			<div className="flex flex-col p-2">
				{projection.effort_ladder.map((rung) => (
					<button
						key={rung}
						type="button"
						onClick={() => set(rung)}
						className="flex min-h-11 items-center gap-3 rounded-sm px-3 text-left active:bg-surface"
					>
						<span
							className={cn(
								"size-2 shrink-0 rounded-full",
								rung === projection.effort
									? "bg-accent"
									: "bg-hairline",
							)}
							aria-hidden
						/>
						<span className="font-mono text-mono text-ink">
							{rung}
						</span>
					</button>
				))}
			</div>
		</Sheet>
	);
}

const MAX_TEXTAREA_PX = 6 * 22; /* six lines at body line-height */

export function Composer({
	projection,
	onOpenModels,
	onOpenEffort,
	effortOpen,
	onCloseEffort,
}: {
	projection: SessionProjection;
	onOpenModels: () => void;
	onOpenEffort: () => void;
	effortOpen: boolean;
	onCloseEffort: () => void;
}) {
	const pid = projection.pid;
	const [text, setText] = useDraft(pid);
	const [slashOpen, setSlashOpen] = useState(false);
	const [sending, setSending] = useState(false);
	const [error, setError] = useState("");
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	/* The resume affordance is driven by the WIRE fact (stop_reason), not an
	   inference from the streaming flag: a turn that completes also flips
	   streaming off, and only an aborted turn should offer "resume". */
	const showResume = projection.stop_reason === "aborted";

	/* Auto-grow: reset to auto so shrink works, then clamp at six lines. */
	useEffect(() => {
		const el = textareaRef.current;
		if (!el) return;
		el.style.height = "auto";
		el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_PX)}px`;
	}, [text]);

	const disabled = projection.ended;

	const send = async (raw: string, op?: "prompt" | "steer") => {
		const trimmed = raw.trim();
		if (!trimmed || sending || disabled) return;
		setSending(true);
		setError("");
		try {
			/* Slash input routes to the slash op rather than prompt. */
			if (trimmed.startsWith("/") && !trimmed.includes("\n")) {
				const space = trimmed.indexOf(" ");
				const command =
					space === -1 ? trimmed.slice(1) : trimmed.slice(1, space);
				const args = space === -1 ? "" : trimmed.slice(space + 1);
				await sendCommand(pid, { op: "slash", command, args });
			} else {
				const chosen =
					op ?? (projection.streaming ? "steer" : "prompt");
				await sendCommand(pid, { op: chosen, text: trimmed });
			}
			setText("");
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setSending(false);
		}
	};

	const abort = async () => {
		try {
			await sendCommand(pid, { op: "abort" });
		} catch (e) {
			setError(String((e as Error).message ?? e));
		}
	};

	const onChange = (value: string) => {
		setText(value);
		setSlashOpen(slashQuery(value) !== null);
	};

	/* The sheet ALSO watches the value: the driver/IME paths that set it
	   without an onChange still open the sheet a frame later. The onChange
	   call above is the zero-latency path for real typing. */
	useEffect(() => {
		setSlashOpen(slashQuery(text) !== null);
	}, [text]);


	const onSlashPick = (fill: string, submit: boolean) => {
		setText(fill);
		if (submit) {
			void send(fill);
		} else {
			textareaRef.current?.focus();
		}
	};

	return (
		<div className="flex flex-col gap-2 px-3 pt-1 pb-[max(env(safe-area-inset-bottom),0.5rem)]">
			{showResume && !projection.streaming && !disabled ? (
				<button
					type="button"
					onClick={() => void send("continue", "prompt")}
					className="flex min-h-9 items-center justify-center rounded-sm border border-hairline bg-surface text-body-sm text-ink-muted active:bg-elevated"
				>
					interrupted — tap to resume
				</button>
			) : null}

			{error ? <p className="text-body-sm text-danger">{error}</p> : null}

			<div className="flex items-end gap-2">
				<div className="flex min-w-0 flex-1 items-end rounded-frame border border-control bg-elevated px-3 py-2">
					<textarea
						ref={textareaRef}
						value={text}
						onChange={(e) => onChange(e.target.value)}
						placeholder={
							disabled
								? "session ended"
								: projection.streaming
									? "steer this turn…"
									: "message"
						}
						disabled={disabled}
						rows={1}
						enterKeyHint="send"
						onKeyDown={(e) => {
							/* Hardware keyboards: Enter sends, Shift+Enter
							   newline. Touch keyboards use the button. */
							if (e.key === "Enter" && !e.shiftKey) {
								e.preventDefault();
								void send(text);
							}
						}}
						className="lo-scroll max-h-33 min-h-6 w-full resize-none bg-transparent text-[16px] leading-[1.4] text-ink outline-none placeholder:text-ink-dim"
					/>
				</div>

				{projection.streaming ? (
					<button
						type="button"
						onClick={abort}
						aria-label="stop"
						className="flex size-11 shrink-0 items-center justify-center rounded-full border border-danger-border text-danger active:bg-danger-wash"
					>
						■
					</button>
				) : null}

				<button
					type="button"
					onClick={() => void send(text)}
					disabled={!text.trim() || sending || disabled}
					aria-label={projection.streaming ? "steer" : "send"}
					className="flex size-11 shrink-0 items-center justify-center rounded-full bg-accent text-on-accent active:bg-accent-active disabled:bg-sunken disabled:text-ink-disabled"
				>
					↑
				</button>
			</div>

			<div className="flex items-center gap-3 px-1">
				{projection.queued_count > 0 ? (
					<span className="font-mono text-mono-sm text-ink-dim">
						{projection.queued_count} queued
					</span>
				) : null}
				<span className="flex-1" />
				<button
					type="button"
					onClick={onOpenModels}
					className="flex min-h-11 items-center font-mono text-mono-sm text-ink-dim active:text-ink-muted"
				>
					{projection.model_label || "model"}
				</button>
				{projection.effort_ladder.length > 0 ? (
					<button
						type="button"
						onClick={onOpenEffort}
						className="flex min-h-11 items-center font-mono text-mono-sm text-ink-dim active:text-ink-muted"
					>
						{projection.effort || "effort"}
					</button>
				) : null}
			</div>

			<SlashSheet
				open={slashOpen && !disabled}
				onClose={() => setSlashOpen(false)}
				onPick={onSlashPick}
			/>
			<EffortSheet
				open={effortOpen}
				onClose={onCloseEffort}
				projection={projection}
			/>
		</div>
	);
}
