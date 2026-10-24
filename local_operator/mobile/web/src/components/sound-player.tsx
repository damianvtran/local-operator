/**
 * THE mobile sound player — one implementation, every seat that plays audio.
 *
 * WHERE IT SITS. The desktop consolidated on one `sound-player.tsx` for four
 * seats; on the phone the same rule holds with three: a Highlights beat's
 * sounds, a Library audio tile, and the Library asset screen. `variant` is the
 * only thing those seats disagree about (a compact row versus a full-width
 * preview); the transport states, the seek gestures and the accessibility
 * contract are identical by construction because they live here.
 *
 * WHY A REGISTRY AND NOT A `<audio>` PER SEAT. The phone has no shared audio
 * engine (the desktop has an `audio-engine-store`); the browser's media
 * elements are per-mount, so two players left alone would happily play over
 * each other. A module-level registry that PAUSES the previously-playing
 * element is therefore the engine this surface has, not a convenience: it is
 * deliberately tiny (two functions and one variable) and the one place that
 * answers "what is audible right now".
 *
 * THE GESTURE CONTRACT (ported from the desktop's `SoundPlayer`):
 *   - a press that moves more than {@link CLICK_SLOP_PX} is a DRAG: it seeks
 *     and must never start playback (a scrub is not a play press);
 *   - a press that does not move is a CLICK: it seeks to the point, then plays
 *     (an already-playing player keeps playing — the seek is the whole act);
 *   - dragging uses pointer capture, so a finger that leaves the track keeps
 *     seeking instead of dropping the scrub.
 *
 * NO DECODING IN THE RENDERER. `peaks` arrives from the relay's waveform route
 * (`null` when the sidecar is absent). `null` draws an honest FLAT scrubber —
 * a suggested envelope the audio does not have would be a lie about the sound.
 *
 * The played ink is a per-bar class (not a second canvas layer): a row here is
 * a handful of bars, and one state write per `timeupdate` must
 * not become a paint hierarchy. The play control keeps a ≥ 44 px target in both
 * variants: a phone has no hover, so a smaller control would not exist.
 */
import {
	type KeyboardEvent,
	type PointerEvent,
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";
import { cn } from "../lib/cn";

/** Movement past this many CSS px turns a press into a drag (the desktop's own slop). */
const CLICK_SLOP_PX = 6;

/** Bars drawn per track. Fixed rather than measured: the container is a phone
 *  column, and a ResizeObserver per player would be cost without a visible gain. */
const BAR_COUNT = 48;

/** Track geometry in viewBox units; `preserveAspectRatio="none"` stretches x only. */
const TRACK_HEIGHT = 32;
const BAR_WIDTH = 0.6;
const BAR_BASE_VALUE = 0.08;
const VALUE_SCALE = 0.9;

export type SoundPlayerVariant = "full" | "compact";

export type SoundPlayerProps = {
	/** The bytes' URL — the relay's digest route. */
	src: string;
	/** The stored waveform bars, or `null` when this sound has no sidecar. */
	peaks: number[] | null;
	/** The sound's name, used for the control and slider accessible names. */
	label: string;
	/** Seconds from the supplement record; `0`/absent means "unknown". */
	durationHint?: number;
	variant?: SoundPlayerVariant;
};

/* ---------------------------------------------------------------------------
 * The module-level playback registry — this surface's audio engine
 * ------------------------------------------------------------------------- */

let audible: HTMLAudioElement | null = null;

/** Make `element` the one audible player, pausing whoever held the slot. */
function claimPlayback(element: HTMLAudioElement): void {
	if (audible !== null && audible !== element) audible.pause();
	audible = element;
}

/** Release the slot only if it is still ours (a later claim owns it otherwise). */
function releasePlayback(element: HTMLAudioElement): void {
	if (audible === element) audible = null;
}

/* ---------------------------------------------------------------------------
 * Geometry and formatting — pure, so the tests can reason about them
 * ------------------------------------------------------------------------- */

/** Downsample `peaks` onto `count` bars by bucket MAXIMUM (the desktop rule:
 *  a transient a mean would flatten is exactly what a waveform exists to show). */
export function waveformValues(
	peaks: readonly number[] | null,
	count = BAR_COUNT,
): number[] | null {
	if (peaks === null) return null;
	if (count <= 0) return [];
	if (peaks.length === 0) return new Array(count).fill(BAR_BASE_VALUE);
	const out: number[] = new Array(count);
	const step = peaks.length / count;
	for (let index = 0; index < count; index += 1) {
		const start = Math.floor(index * step);
		const end = Math.max(start + 1, Math.floor((index + 1) * step));
		let peak = 0;
		for (let cursor = start; cursor < end && cursor < peaks.length; cursor += 1) {
			const value = peaks[cursor];
			if (Number.isFinite(value) && value > peak) peak = value;
		}
		out[index] = Math.max(0, Math.min(1, peak));
	}
	return out;
}

/** The `0:07` clock; an unknown or negative duration is `0:00`, never `-1:-1`. */
export function formatClock(seconds: number): string {
	if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
	const total = Math.floor(seconds);
	const minutes = Math.floor(total / 60);
	const rest = total % 60;
	return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

/**
 * Fetch one waveform payload and return its `peaks`, or `null`.
 *
 * EVERY FAILURE IS THE FLAT SCRUBBER: a 404 (no sidecar), a transport failure,
 * a malformed body — the player still renders and still plays. The relay's
 * `no-store` policy is respected by the browser; nothing here caches across
 * mounts, because a digest's peaks are cheap to re-ask and a stale envelope is
 * the one thing a waveform must not show.
 */
export function useWaveformPeaks(url: string | null, enabled = true): number[] | null {
	const [peaks, setPeaks] = useState<number[] | null>(null);
	useEffect(() => {
		setPeaks(null);
		if (!enabled || url === null) return;
		let live = true;
		const controller = new AbortController();
		fetch(url, { credentials: "include", signal: controller.signal })
			.then(async (response) => {
				if (!response.ok) return null;
				const body: unknown = await response.json();
				if (!body || typeof body !== "object") return null;
				const raw = (body as Record<string, unknown>).peaks;
				if (!Array.isArray(raw)) return null;
				const values = raw.filter(
					(value): value is number => typeof value === "number" && Number.isFinite(value),
				);
				/* Length is part of the payload's contract (`bars`): a partial list
				   is a broken sidecar, and drawing it would misplace every bar. */
				return values.length === raw.length && values.length > 0 ? values : null;
			})
			.then((values) => {
				if (live) setPeaks(values);
			})
			.catch(() => {
				/* An aborted fetch is this effect's own cleanup, not a failure; a
				   real one degrades to the flat scrubber either way. */
				if (live) setPeaks(null);
			});
		return () => {
			live = false;
			controller.abort();
		};
	}, [url, enabled]);
	return peaks;
}

/* ---------------------------------------------------------------------------
 * The player
 * ------------------------------------------------------------------------- */

type Status = "idle" | "loading" | "playing" | "paused" | "error";

export function SoundPlayer({
	src,
	peaks,
	label,
	durationHint = 0,
	variant = "compact",
}: SoundPlayerProps) {
	const audioRef = useRef<HTMLAudioElement | null>(null);
	const trackRef = useRef<HTMLDivElement | null>(null);
	/** The press in progress: where it began, and whether it has become a drag. */
	const pressRef = useRef<{ x: number; moved: boolean } | null>(null);
	const [status, setStatus] = useState<Status>("idle");
	const [time, setTime] = useState(0);
	const [mediaDuration, setMediaDuration] = useState(0);
	const [dragging, setDragging] = useState(false);

	const duration = mediaDuration > 0 ? mediaDuration : Math.max(0, durationHint);
	const progress = duration > 0 ? Math.max(0, Math.min(1, time / duration)) : 0;
	const playing = status === "playing";
	const values = waveformValues(peaks);
	/* A bar's played/unplayed state is a class swap; the fraction comes from the
	   same `progress` the clock shows, so ink and time never disagree. */
	const playedBars = values === null ? 0 : Math.round(values.length * progress);

	useEffect(() => {
		return () => {
			const element = audioRef.current;
			if (element !== null) {
				/* Stop on unmount: the element leaves the DOM with this component,
				   so an unmounted player that kept sounding would be audio with no
				   control anywhere on the surface (a windowed transcript row unmounts
				   as the reader scrolls away). */
				element.pause();
				releasePlayback(element);
			}
		};
	}, []);

	/** Seek to a fraction of the known duration. Unknown duration cannot be
	 *  mapped to seconds, so the track is inert rather than guessing. */
	const seekFraction = useCallback(
		(fraction: number) => {
			if (duration <= 0) return;
			const seconds = Math.max(0, Math.min(duration, fraction * duration));
			setTime(seconds);
			const element = audioRef.current;
			if (element !== null) {
				try {
					element.currentTime = seconds;
				} catch {
					/* A media element between states can refuse a seek; the next
					   metadata event makes the position coherent again. */
				}
			}
		},
		[duration],
	);

	const fractionAt = useCallback((clientX: number): number | null => {
		const rect = trackRef.current?.getBoundingClientRect();
		if (!rect || rect.width <= 0) return null;
		return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
	}, []);

	const start = useCallback(() => {
		const element = audioRef.current;
		if (element === null) return;
		claimPlayback(element);
		setStatus("loading");
		try {
			const attempt = element.play();
			if (attempt !== undefined) {
				void attempt
					.then(() => setStatus((held) => (held === "loading" ? "playing" : held)))
					.catch(() => setStatus("error"));
			}
		} catch {
			setStatus("error");
		}
	}, []);

	const toggle = useCallback(() => {
		const element = audioRef.current;
		if (element === null) return;
		if (playing) {
			element.pause();
			return;
		}
		start();
	}, [playing, start]);

	const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
		if (duration <= 0) return;
		event.preventDefault();
		const fraction = fractionAt(event.clientX);
		if (fraction === null) return;
		pressRef.current = { x: event.clientX, moved: false };
		try {
			/* Capture is an optimisation (a finger that leaves the track keeps
			   seeking), never the gesture itself: a SYNTHETIC pointer event has
			   no active pointer to capture and throws NotFoundError, which would
			   otherwise abort the press and lose the seek (qa round 1, Q2). */
			trackRef.current?.setPointerCapture?.(event.pointerId);
		} catch {
			/* Uncapturable pointer: the press still seeks and still ends. */
		}
		seekFraction(fraction);
	};

	const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
		const press = pressRef.current;
		if (press === null) return;
		if (!press.moved && Math.abs(event.clientX - press.x) > CLICK_SLOP_PX) {
			press.moved = true;
			setDragging(true);
		}
		if (!press.moved) return;
		const fraction = fractionAt(event.clientX);
		if (fraction !== null) seekFraction(fraction);
	};

	/**
	 * End the press. `click` is true only for a real pointer-up: a CANCEL is the
	 * opposite of a play intent, so it drops the gesture without seeking further
	 * and without playing (round-1 review N1).
	 */
	const finishPress = (event: PointerEvent<HTMLDivElement>, click: boolean) => {
		const press = pressRef.current;
		if (press === null) return;
		pressRef.current = null;
		try {
			trackRef.current?.releasePointerCapture?.(event.pointerId);
		} catch {
			/* Released, or never captured: nothing to undo. */
		}
		if (!click || press.moved) {
			/* A scrub only seeks; starting playback here would make every drag a
			   play press, which is the gesture contract's one prohibition. */
			setDragging(false);
			return;
		}
		/* A click seeks (on press) and then plays. An already-playing player
		   keeps playing from the new point — the seek was the whole act. */
		if (!playing) start();
	};

	const onTrackKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		if (duration <= 0) return;
		const step = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
		if (step === 0 && event.key !== "Home" && event.key !== "End") return;
		event.preventDefault();
		const next =
			event.key === "Home"
				? 0
				: event.key === "End"
					? duration
					: Math.max(0, Math.min(duration, time + step));
		seekFraction(duration > 0 ? next / duration : 0);
	};

	const errorLine = status === "error" ? (
		<p className="text-meta text-ink-dim" data-sound-error="">
			could not play
		</p>
	) : null;

	return (
		<div
			className={cn(
				"flex min-w-0 flex-col",
				variant === "full" ? "gap-2" : "gap-1",
			)}
			data-sound-player={variant}
			data-sound-state={status}
			data-waveform={values === null ? "flat" : "peaks"}
		>
			{/* The media element is the engine, never a control surface: the
			    player's own transport is the one interface (a native
			    `<audio controls>` restyled into place is a second interface, and
			    the one this PR must not add). `preload="none"`: bytes are fetched
			    when the reader presses play, not when a list mounts. */}
			<audio
				ref={audioRef}
				src={src}
				preload="none"
				aria-hidden="true"
				className="hidden"
				onLoadedMetadata={(event) => {
					const value = event.currentTarget.duration;
					if (Number.isFinite(value) && value > 0) setMediaDuration(value);
				}}
				onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
				onPlay={(event) => {
					claimPlayback(event.currentTarget);
					setStatus("playing");
				}}
				onPause={() => setStatus((held) => (held === "error" ? held : "paused"))}
				onEnded={(event) => {
					releasePlayback(event.currentTarget);
					setStatus("idle");
					setTime(0);
					event.currentTarget.currentTime = 0;
				}}
				onError={() => setStatus("error")}
			/>
			<div className="flex min-w-0 items-center gap-2">
				<button
					type="button"
					onClick={toggle}
					aria-label={`${playing ? "pause" : "play"} ${label}`}
					aria-pressed={playing}
					className="flex min-h-11 min-w-11 shrink-0 items-center justify-center rounded-sm text-body text-ink-muted active:bg-elevated disabled:opacity-50"
				>
					<span aria-hidden="true">{playing ? "❚❚" : "▶"}</span>
				</button>
				<div
					ref={trackRef}
					role="slider"
					tabIndex={0}
					aria-label={`${label} position`}
					aria-valuemin={0}
					aria-valuemax={Math.max(0, Math.round(duration))}
					aria-valuenow={Math.round(time)}
					aria-valuetext={`${formatClock(time)} of ${formatClock(duration)}`}
					onPointerDown={onPointerDown}
					onPointerMove={onPointerMove}
					onPointerUp={(event) => finishPress(event, true)}
					onPointerCancel={(event) => finishPress(event, false)}
					onKeyDown={onTrackKeyDown}
					className={cn(
						"relative flex min-h-11 min-w-0 flex-1 touch-none select-none items-center overflow-hidden rounded-sm",
						dragging && "bg-elevated",
					)}
					data-sound-track=""
				>
					<svg
						viewBox={`0 0 ${BAR_COUNT} ${TRACK_HEIGHT}`}
						preserveAspectRatio="none"
						aria-hidden="true"
						className={variant === "full" ? "h-10 w-full" : "h-8 w-full"}
					>
						{(values ?? new Array(BAR_COUNT).fill(BAR_BASE_VALUE)).map((value, index) => {
							const height = Math.max(
								1.5,
								Math.min(TRACK_HEIGHT, value * TRACK_HEIGHT * VALUE_SCALE),
							);
							return (
								<rect
									// Bars are positional and never reordered: the index is the identity.
									key={index}
									x={index + (1 - BAR_WIDTH) / 2}
									y={TRACK_HEIGHT - height}
									width={BAR_WIDTH}
									height={height}
									rx={0.25}
									fill="currentColor"
									className={index < playedBars ? "text-accent" : "text-ink-dim"}
								/>
							);
						})}
					</svg>
				</div>
				<span
					className="shrink-0 font-mono text-meta text-ink-dim tabular-nums"
					data-sound-clock=""
				>
					{/* The compact seat spends its width on the bars: a grid cell
					    holds one number. The full seat is the asset screen, where
					    "elapsed / total" is the reader's own orientation. */}
					{variant === "full"
						? `${formatClock(time)} / ${formatClock(duration)}`
						: formatClock(time)}
				</span>
			</div>
			{errorLine}
		</div>
	);
}
