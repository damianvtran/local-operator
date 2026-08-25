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
	private completedVersion = -1;
	private activeVersion = -1;
	private pendingVersion = -1;
	private retryActiveOnFailure = false;
	private disposed = false;

	constructor(
		private readonly load: () => Promise<SubagentDetail>,
		private readonly accept: (detail: SubagentDetail) => void,
		private readonly reject: (reason: unknown) => void,
	) {}

	request(version: number): void {
		if (this.disposed || version <= this.completedVersion && !this.inFlight) return;
		if (this.inFlight) {
			if (version > this.activeVersion) {
				this.pendingVersion = Math.max(this.pendingVersion, version);
			} else if (version === this.activeVersion) {
				/* Reconnect is allowed to race the active GET. Remember one retry,
				   but consume it only if that GET fails so a healthy response does
				   not create a same-version request loop. */
				this.retryActiveOnFailure = true;
			}
			return;
		}
		this.start(version);
	}

	dispose(): void {
		this.disposed = true;
	}

	private start(version: number): void {
		this.inFlight = true;
		this.activeVersion = version;
		this.retryActiveOnFailure = false;
		let failed = false;
		void this.load()
			.then((detail) => {
				/* Only a successful response satisfies a version. A transient failure
				   must remain retryable when the stream reconnects without advancing. */
				this.completedVersion = Math.max(this.completedVersion, version);
				if (!this.disposed) this.accept(detail);
			})
			.catch((reason: unknown) => {
				failed = true;
				if (!this.disposed) this.reject(reason);
			})
			.finally(() => {
				this.inFlight = false;
				if (this.disposed) return;
				const followUp = Math.max(
					this.pendingVersion,
					failed && this.retryActiveOnFailure ? version : -1,
				);
				this.pendingVersion = -1;
				this.retryActiveOnFailure = false;
				if (followUp > this.completedVersion) this.start(followUp);
			});
	}
}
