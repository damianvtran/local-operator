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
import { messageRuntime } from "../store";

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

export function AgentAvatar({
	agentId,
	sha,
	name,
	size,
	fallback = null,
	className,
}: {
	agentId?: string | null;
	sha?: string | null;
	/** The agent's display name, for the alt text. */
	name: string;
	/** Edge in CSS px; the stored file is 512², so any phone density is sharp. */
	size: number;
	fallback?: ReactNode;
	className?: string;
}) {
	const url = agentAvatarUrl(agentId, sha);
	/* A 404/401 (picture removed between frames, session cookie expired) falls
	   back to the glyph instead of a broken-image icon. Keyed by URL so the next
	   sha retries. */
	const [failed, setFailed] = useState<string | null>(null);
	if (!url || failed === url) return <>{fallback}</>;
	return (
		<img
			src={url}
			alt={messageRuntime.t("relay.agent_avatar.alt", { name })}
			width={size}
			height={size}
			style={{ width: size, height: size }}
			className={cn("shrink-0 rounded-full object-cover bg-surface", className)}
			draggable={false}
			onError={() => setFailed(url)}
			data-agent-avatar=""
		/>
	);
}

/**
 * The list row's leading slot when the row has a picture: the picture with the
 * status mark kept. Without a picture it returns `status` untouched, so every
 * pictureless row's geometry is byte-for-byte what it was.
 */
export function AvatarWithStatus({
	agentId,
	sha,
	name,
	status,
	hasStatus,
}: {
	agentId?: string | null;
	sha?: string | null;
	name: string;
	/** The existing 12px status slot, unchanged. */
	status: ReactNode;
	/** Whether the slot currently shows a mark (idle rows show a transparent one). */
	hasStatus: boolean;
}) {
	if (!agentAvatarUrl(agentId, sha)) return <>{status}</>;
	const picture = <AgentAvatar agentId={agentId} sha={sha} name={name} size={24} />;
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
				/* The badge: the ladder's own mark on a canvas-coloured disc at the
				   end-bottom corner (logical inset, so RTL mirrors it). The disc is
				   what keeps a red pulse or an accent dot readable over a busy photo. */
				<span className="absolute -bottom-0.5 -end-0.5 flex size-3.5 items-center justify-center rounded-full bg-canvas">
					{status}
				</span>
			) : null}
		</span>
	);
}
