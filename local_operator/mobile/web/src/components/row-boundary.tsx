/**
 * Error boundary around one transcript row. A single malformed entry (a
 * details shape the renderer didn't expect) must not unmount the whole app —
 * the field failure that motivated this: one bad tool row blanked the entire
 * screen. Confine the throw to the row and render a quiet fallback, so the
 * rest of the transcript, the header, and the composer stay live.
 */
import { Component, type ReactNode } from "react";

export class RowBoundary extends Component<
	{ children: ReactNode },
	{ failed: boolean }
> {
	state = { failed: false };

	static getDerivedStateFromError(): { failed: boolean } {
		return { failed: true };
	}

	componentDidCatch(error: unknown): void {
		// Surface the real error for debugging without taking the app down.
		console.error("transcript row failed to render", error);
	}

	render(): ReactNode {
		if (this.state.failed) {
			return (
				<div className="rounded-sm px-1.5 py-1 font-mono text-mono-sm text-ink-dim">
					— row failed to render —
				</div>
			);
		}
		return this.props.children;
	}
}
