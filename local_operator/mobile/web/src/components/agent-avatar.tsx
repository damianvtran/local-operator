/**
 * An agent's profile picture on the phone, with the existing glyph as fallback.
 *
 * WHERE THE PICTURE COMES FROM. The relay's `/api/sessions` rows carry the bound
 * agent's `agent_id` and `avatar_sha256` (contract: core `local_operator/
 * agent_avatar.py`), so this component never fetches to learn whether a picture
 * exists: it builds `/v1/agents/{agent_id}/avatar?v=<sha>` from the row. The
 * phone reaches ONLY the relay, which serves that same path behind its own
 * cookie auth (`api_agent_avatar` in `daemon.py`); the `?v=` sha makes the URL
 * content-addressed, so a changed picture is a new URL and the browser cache can
 * never paint a stale one. A row without both values renders `fallback`.
 *
 * Builds on the desktop proof of concept by bbqben (local-operator-ui#1517,
 * `AgentAvatar`): same round crop and glyph fallback, but backend-owned bytes.
 *
 * STATUS STAYS LEGIBLE. The list row's status ladder (spinner / danger pulse /
 * unread dot / delegating) is NEVER replaced by the picture: `AvatarWithStatus`
 * keeps the ladder's mark and pins it as a BADGE on the picture's lower-end
 * corner, ringed in the canvas colour so it reads against any photo. This is
 * the designer-chosen "badge on corner" variant; `STATUS_PLACEMENT` switches it
 * to the "beside" variant (picture, then the unchanged slot) in one line.
 */
import { type ReactNode, useState } from "react";
import { cn } from "../lib/cn";

/** Which layout pairs the picture with the row's status mark. */
export const STATUS_PLACEMENT: "badge" | "beside" = "badge";

/** The relay URL for an agent's stored picture, or null when the row has none. */
export function agentAvatarUrl(
	agentId: string | null | undefined,
	sha: string | null | undefined,
): string | null {
	if (!agentId || !sha) return null;
	return `/v1/agents/${encodeURIComponent(agentId)}/avatar?v=${encodeURIComponent(sha)}`;
}

/**
 * The picture a row/header should lay out for, or `null` once it failed to load.
 *
 * WHY THE LAYOUT OWNER HOLDS THIS, NOT THE `<img>`. A 404 (picture removed
 * between frames, a stale sha, an expired cookie) must leave EXACTLY the
 * no-picture layout: no 24px box, no 48px title indent, no invisible header
 * button. The `<img>` alone cannot give that, because its fallback renders
 * inside a wrapper whose size and indent the PARENT already decided from "a URL
 * exists" (design review D1: a blank hole, a spinner stranded at an empty box's
 * corner). So the parent asks this hook, passes `onFailed` down, and re-renders
 * without the slot when the image reports an error. Keyed by URL, so the next
 * sha retries; component-local, so a remount (or a reload) retries too.
 */
export function useAgentAvatar(
	agentId: string | null | undefined,
	sha: string | null | undefined,
): { url: string | null; onFailed: () => void } {
	const url = agentAvatarUrl(agentId, sha);
	const [failed, setFailed] = useState<string | null>(null);
	return { url: url && failed !== url ? url : null, onFailed: () => setFailed(url) };
}

/**
 * The picture itself. DECORATIVE by default (`alt=""`): it always sits inside a
 * control that already has a name (the row's button, the header title), and a
 * non-empty alt made a screen reader say the conversation twice, the first time
 * wrongly ("Inbox triage's picture working Inbox triage …" -- design review D2).
 * A 1px `ring-control` edge keeps a near-white picture on the light canvas and a
 * near-black one on the dark canvas from dissolving into it (1.09:1 / 1.01:1
 * without it); it is the ring `BrandMark` already uses for the same reason.
 */
export function AgentAvatar({
	url,
	size,
	onFailed,
	className,
}: {
	url: string;
	/** Edge in CSS px; the stored file is 512², so any phone density is sharp. */
	size: number;
	onFailed?: () => void;
	className?: string;
}) {
	return (
		<img
			src={url}
			alt=""
			width={size}
			height={size}
			style={{ width: size, height: size }}
			className={cn("shrink-0 rounded-full object-cover bg-surface ring-1 ring-control", className)}
			draggable={false}
			onError={onFailed}
			data-agent-avatar=""
		/>
	);
}

/**
 * The list row's leading slot when the row has a (loadable) picture: the picture
 * with the status mark kept. Without one it returns `status` untouched, so every
 * pictureless row's geometry is byte-for-byte what it was.
 *
 * SIZES MATCH THE NATIVE APP (local-operator-mobile#143, design-approved): a 24px
 * picture, the 12px status mark on a 16px disc (a real 2px ring around it), the
 * disc in the ROW's surface colour. `group-active:bg-surface` follows the pressed
 * row (the card carries `group`), so the ring never reads as a patch on a
 * pressed row (design review D3/N2).
 */
export function AvatarWithStatus({
	avatar,
	status,
	hasStatus,
}: {
	avatar: { url: string | null; onFailed: () => void };
	/** The existing 12px status slot, unchanged. */
	status: ReactNode;
	/** Whether the slot currently shows a mark (idle rows show a transparent one). */
	hasStatus: boolean;
}) {
	if (!avatar.url) return <>{status}</>;
	const picture = <AgentAvatar url={avatar.url} size={24} onFailed={avatar.onFailed} />;
	if (STATUS_PLACEMENT === "beside") {
		return (
			<span className="flex shrink-0 items-center gap-1.5">
				{picture}
				{status}
			</span>
		);
	}
	return (
		<span className="relative inline-flex size-6 shrink-0" data-avatar-status="badge">
			{picture}
			{hasStatus ? (
				/* The badge: the ladder's own mark on a row-coloured disc at the
				   end-bottom corner (logical inset, so RTL mirrors it). The disc is
				   what keeps a red pulse or an accent dot readable over a busy photo. */
				<span className="absolute -bottom-0.5 -end-0.5 flex size-4 items-center justify-center rounded-full bg-canvas group-active:bg-surface">
					{status}
				</span>
			) : null}
		</span>
	);
}
