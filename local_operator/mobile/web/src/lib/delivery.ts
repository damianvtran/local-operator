/**
 * The busy-send delivery mode on the phone: steer / queue / both.
 *
 * WHY THIS IS A MODULE. The mode is a per-device preference with three rules
 * that must agree everywhere it is read — the composer chip's visibility, the
 * send annotation's shape, and the retry envelope's identity. Kept pure
 * (apart from the two localStorage accessors at the bottom) so the rule is
 * unit-testable without a DOM, the same shape `composer-slots.ts` holds for
 * the action slots.
 *
 * THE VOCABULARY IS THE ENGINE'S. `steer` / `queue` / `both` are the wire
 * values (`delivery` in `mobile/types.py`, `BusyDelivery` in `session.py`):
 * a client inventing its own words would refuse or mis-deliver. `steer` is
 * today's urgent behaviour; `queue` waits for the step and never interrupts
 * the tools; `both` waits and escalates to a steer when it has waited too
 * long. A mode absent from the preference is `steer` — the default and the
 * legacy reading, so an old phone and a new one send byte-identical frames.
 */
import type { Capabilities } from "../types";

export type DeliveryMode = "steer" | "queue" | "both";

/** The shipped default (design D6): a busy send behaves exactly as it always
    has until the reader changes the mode. */
export const DEFAULT_DELIVERY_MODE: DeliveryMode = "steer";

/** The chip cycles in this order; `steer` is the resting default. */
export const DELIVERY_MODES: readonly DeliveryMode[] = ["steer", "queue", "both"];

/** Whether a preference value is a real mode, or the legacy absent reading. */
export function isDeliveryMode(value: unknown): value is DeliveryMode {
	return value === "steer" || value === "queue" || value === "both";
}

/** The next mode in cycle order, wrapping. Unknown input reads as `steer`. */
export function nextDeliveryMode(mode: DeliveryMode): DeliveryMode {
	const index = DELIVERY_MODES.indexOf(mode);
	return DELIVERY_MODES[(index + 1) % DELIVERY_MODES.length]!;
}

/**
 * Whether the relay can honour the busy-send delivery modes.
 *
 * The relay lifts core's feature dict into `capabilities.features`; the build
 * fact is `busy_delivery: 1` beside `input_mode`. Presence is the gate: an
 * older relay has no key and must hide the selector rather than offer a
 * control whose send it would 400. The value is read as truthy, never
 * invented — a `0` or an absent key is the same "no".
 */
export function busyDeliverySupported(capabilities: Capabilities | null): boolean {
	return Boolean(capabilities?.features?.busy_delivery);
}

/**
 * The `delivery` value a fresh send carries, or `undefined` for a legacy frame.
 *
 * `steer` (and an unsupported relay) sends NOTHING: the frame stays
 * byte-identical to a pre-feature send, so an owner from before the feature
 * never sees a key it does not read. `queue` / `both` ride the field ONLY
 * when the relay advertised it; otherwise the selector is hidden upstream and
 * the mode is `steer` here (fail-safe without offering the control).
 */
export function deliveryForSend(
	mode: DeliveryMode,
	streaming: boolean,
	supported: boolean,
): DeliveryMode | undefined {
	if (!streaming || !supported) return undefined;
	return mode === "steer" ? undefined : mode;
}

/** The per-device storage key (survives sign-out, like the theme). */
const KEY = "lo-mobile-delivery-mode";

export function getDeliveryMode(): DeliveryMode {
	try {
		const saved = localStorage.getItem(KEY);
		return isDeliveryMode(saved) ? saved : DEFAULT_DELIVERY_MODE;
	} catch {
		return DEFAULT_DELIVERY_MODE;
	}
}

export function setDeliveryMode(mode: DeliveryMode): void {
	try {
		if (mode === DEFAULT_DELIVERY_MODE) localStorage.removeItem(KEY);
		else localStorage.setItem(KEY, mode);
	} catch {
		/* A preference that cannot persist is still usable this session. */
	}
}
