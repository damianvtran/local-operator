import type { SubagentDetail } from "./types";

/**
 * Keep one selected-detail request in flight and remember only the newest
 * projection version that arrived behind it. Projection frames can arrive
 * faster than a phone round trip, so cancelling on every frame starves the
 * route; coalescing guarantees the current request completes and schedules at
 * most one catch-up request.
 */
export class DetailRequestCoordinator {
	private inFlight = false;
	private requestedVersion = -1;
	private pendingVersion = -1;
	private disposed = false;

	constructor(
		private readonly load: () => Promise<SubagentDetail>,
		private readonly accept: (detail: SubagentDetail) => void,
		private readonly reject: (reason: unknown) => void,
	) {}

	request(version: number): void {
		if (this.disposed || version <= this.requestedVersion && !this.inFlight) return;
		if (this.inFlight) {
			this.pendingVersion = Math.max(this.pendingVersion, version);
			return;
		}
		this.start(version);
	}

	dispose(): void {
		this.disposed = true;
	}

	private start(version: number): void {
		this.inFlight = true;
		this.requestedVersion = Math.max(this.requestedVersion, version);
		void this.load()
			.then((detail) => {
				if (!this.disposed) this.accept(detail);
			})
			.catch((reason: unknown) => {
				if (!this.disposed) this.reject(reason);
			})
			.finally(() => {
				this.inFlight = false;
				if (this.disposed) return;
				const followUp = this.pendingVersion;
				this.pendingVersion = -1;
				if (followUp > this.requestedVersion) this.start(followUp);
			});
	}
}
