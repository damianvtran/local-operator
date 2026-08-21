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
import type { PromptImage, SessionProjection, SlashCommand } from "../types";

/** One attached image, kept as the wire form plus a local object URL for the
    thumbnail strip. */
interface AttachedImage extends PromptImage {
	/** Object URL for the thumbnail preview; revoked on remove/send. */
	preview: string;
}

/* Read a pasted/dropped image File into the wire form. Oversize images are
   downscaled first: a 12 MP phone photo is several MB of base64 that the
   provider would reject anyway, and the rebound the session would do on the
   way in costs the same pixels. 1568px matches the session's own bound. */
const MAX_IMAGE_EDGE = 1568;
async function fileToImage(file: File): Promise<AttachedImage | null> {
	if (!file.type.startsWith("image/")) return null;
	const preview = URL.createObjectURL(file);
	try {
		const bmp = await createImageBitmap(file);
		let { width, height } = bmp;
		const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(width, height));
		width = Math.round(width * scale);
		height = Math.round(height * scale);
		const canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext("2d");
		if (!ctx) return null;
		ctx.drawImage(bmp, 0, 0, width, height);
		const blob: Blob | null = await new Promise((res) =>
			canvas.toBlob(res, file.type === "image/png" ? "image/png" : "image/jpeg", 0.9),
		);
		if (!blob) return null;
		const buf = await blob.arrayBuffer();
		let bin = "";
		const bytes = new Uint8Array(buf);
		for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
		return {
			data_b64: btoa(bin),
			mime_type: blob.type,
			preview,
		};
	} catch {
		URL.revokeObjectURL(preview);
		return null;
	}
}


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
	query,
}: {
	open: boolean;
	onClose: () => void;
	/** fill: text to place in the composer; submit: send immediately. */
	onPick: (fill: string, submit: boolean) => void;
	/** The token after `/` in the composer — seeds the filter. */
	query: string;
}) {
	const [commands, setCommands] = useState<SlashCommand[]>([]);
	const [filter, setFilter] = useState(query);
	const [loaded, setLoaded] = useState(false);

	useEffect(() => {
		if (!open) return;
		setFilter(query);
		setLoaded(false);
		getCommands()
			.then((r) => {
				setCommands(r.commands);
				setLoaded(true);
			})
			.catch(() => {
				setCommands([]);
				setLoaded(true);
			});
	}, [open, query]);

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
					className="mb-1 min-h-9 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
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
						className="flex min-h-8 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
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
				{!loaded ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						loading…
					</p>
				) : filtered.length === 0 ? (
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
	pid,
	projection,
}: {
	open: boolean;
	onClose: () => void;
	pid: number;
	projection: SessionProjection;
}) {
	const set = async (effort: string) => {
		try {
			await sendCommand(pid, { op: "set_effort", effort });
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
						className="flex min-h-8 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
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
	pid,
	projection,
	onOpenModels,
	onOpenEffort,
	effortOpen,
	onCloseEffort,
}: {
	/** Route pid — the discovery record's, not the fold's (which stamps 0). */
	pid: number;
	projection: SessionProjection;
	onOpenModels: () => void;
	onOpenEffort: () => void;
	effortOpen: boolean;
	onCloseEffort: () => void;
}) {
	const [text, setText] = useDraft(pid);
	const [slashOpen, setSlashOpen] = useState(false);
	const [sending, setSending] = useState(false);
	const [error, setError] = useState("");
	const [images, setImages] = useState<AttachedImage[]>([]);
	const [dragOver, setDragOver] = useState(false);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const fileInputRef = useRef<HTMLInputElement>(null);

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

	const addFiles = async (files: Iterable<File>) => {
		for (const f of files) {
			const img = await fileToImage(f);
			if (img) setImages((cur) => [...cur, img]);
		}
	};

	/* The attach button's file picker. The pipeline is image-only today
	   (paste/drop already are), so the input scopes to images; a non-image
	   pick is ignored rather than sent as a broken base64 block. */
	const onPickFiles = (list: FileList | null) => {
		if (!list) return;
		void addFiles(Array.from(list));
		/* Reset so picking the SAME file twice still fires change. */
		if (fileInputRef.current) fileInputRef.current.value = "";
	};

	const removeImage = (preview: string) => {
		setImages((cur) => {
			const hit = cur.find((i) => i.preview === preview);
			if (hit) URL.revokeObjectURL(hit.preview);
			return cur.filter((i) => i.preview !== preview);
		});
	};

	const send = async (raw: string, op?: "prompt" | "steer") => {
		const trimmed = raw.trim();
		if ((!trimmed && images.length === 0) || sending || disabled) return;
		setSending(true);
		setError("");
		try {
			/* Slash input routes to the slash op rather than prompt — and only
			   when there is no attachment, since a "/…" caption with an image
			   is a prompt, not a command. */
			if (trimmed.startsWith("/") && !trimmed.includes("\n") && images.length === 0) {
				const space = trimmed.indexOf(" ");
				const command =
					space === -1 ? trimmed.slice(1) : trimmed.slice(1, space);
				const args = space === -1 ? "" : trimmed.slice(space + 1);
				await sendCommand(pid, { op: "slash", command, args });
			} else {
				const chosen =
					op ?? (projection.streaming ? "steer" : "prompt");
				await sendCommand(pid, {
					op: chosen,
					text: trimmed,
					images: images.length
						? images.map(({ data_b64, mime_type }) => ({ data_b64, mime_type }))
						: undefined,
				});
				images.forEach((i) => URL.revokeObjectURL(i.preview));
				setImages([]);
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
		<div className="flex flex-col gap-1.5 px-3 pt-1.5 pb-[max(env(safe-area-inset-bottom),0.5rem)]">
			{showResume && !projection.streaming && !disabled ? (
				<button
					type="button"
					onClick={() => void send("continue", "prompt")}
					className="flex min-h-11 items-center justify-center rounded-sm border border-control bg-surface text-body-sm text-ink active:bg-elevated"
				>
					interrupted — tap to resume
				</button>
			) : null}

			{error ? <p className="text-body-sm text-danger">{error}</p> : null}

			{images.length > 0 ? (
				<div className="flex flex-wrap gap-1.5">
					{images.map((img) => (
						<button
							key={img.preview}
							type="button"
							onClick={() => removeImage(img.preview)}
							aria-label="remove attachment"
							className="relative size-14 overflow-hidden rounded-sm border border-control"
						>
							<img
								src={img.preview}
								alt=""
								className="size-full object-cover"
							/>
							<span className="absolute inset-0 flex items-center justify-center bg-scrim text-meta text-on-accent opacity-0 active:opacity-100">
								remove
							</span>
						</button>
					))}
				</div>
			) : null}

			<div
				className="flex items-end gap-2"
				onDragOver={(e) => {
					e.preventDefault();
					if (!disabled) setDragOver(true);
				}}
				onDragLeave={() => setDragOver(false)}
				onDrop={(e) => {
					e.preventDefault();
					setDragOver(false);
					if (!disabled) void addFiles(e.dataTransfer.files);
				}}
			>
				<input
					ref={fileInputRef}
					type="file"
					accept="image/*"
					multiple
					className="hidden"
					onChange={(e) => onPickFiles(e.target.files)}
				/>
				<button
					type="button"
					onClick={() => fileInputRef.current?.click()}
					disabled={disabled}
					aria-label="attach image"
					className="flex size-11 shrink-0 items-center justify-center rounded-full border border-control text-ink-muted active:bg-elevated disabled:opacity-50"
				>
					{/* paperclip, drawn so it needs no icon font */}
					<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
						<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
					</svg>
				</button>
				<div
					className={cn(
						"flex min-w-0 flex-1 items-end rounded-md border bg-elevated px-3 py-2",
						dragOver ? "border-accent" : "border-control",
					)}
				>
					<textarea
						ref={textareaRef}
						value={text}
						onChange={(e) => onChange(e.target.value)}
						onPaste={(e) => {
							const files = Array.from(e.clipboardData.files);
							if (files.length > 0) {
								e.preventDefault();
								void addFiles(files);
							}
						}}
						placeholder={
							disabled
								? "session ended"
								: projection.streaming
									? "steer this turn…"
									: "Message…"
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
						className="lo-scroll max-h-33 min-h-6 w-full resize-none bg-transparent text-[16px] leading-[1.4] text-ink outline-none placeholder:text-ink-muted"
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
					disabled={(!text.trim() && images.length === 0) || sending || disabled}
					aria-label={projection.streaming ? "steer" : "send"}
					className="flex size-11 shrink-0 items-center justify-center rounded-full bg-accent text-on-accent active:bg-accent-active disabled:bg-sunken disabled:text-ink-disabled"
				>
					↑
				</button>
			</div>

			<div className="flex items-center gap-2 px-0.5">
				{projection.queued_count > 0 ? (
					<span className="font-mono text-mono-sm text-ink-dim">
						{projection.queued_count} queued
					</span>
				) : null}
				<span className="flex-1" />
				<button
					type="button"
					onClick={onOpenModels}
					className="flex min-h-8 items-center font-mono text-mono-sm text-ink-dim active:text-ink-muted"
				>
					{projection.model_label || "model"}
				</button>
				{projection.effort_ladder.length > 0 ? (
					<button
						type="button"
						onClick={onOpenEffort}
						className="flex min-h-8 items-center font-mono text-mono-sm text-ink-dim active:text-ink-muted"
					>
						{projection.effort || "effort"}
					</button>
				) : null}
			</div>

			<SlashSheet
				open={slashOpen && !disabled}
				onClose={() => setSlashOpen(false)}
				onPick={onSlashPick}
				query={slashQuery(text) ?? ""}
			/>
			<EffortSheet
				open={effortOpen}
				onClose={onCloseEffort}
				pid={pid}
				projection={projection}
			/>
		</div>
	);
}
