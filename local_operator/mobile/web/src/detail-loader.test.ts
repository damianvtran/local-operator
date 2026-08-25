import { describe, expect, it, vi } from "vitest";
import { DetailRequestCoordinator } from "./detail-loader";
import type { SubagentDetail } from "./types";

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((done) => { resolve = done; });
	return { promise, resolve };
}

function detail(version: number): SubagentDetail {
	return {
		version, job_id: "child", label: "child", agent: "coder", status: "running",
		progress: "", elapsed_s: 1, model_label: "", result_text: "", error_text: "",
		parent_job_id: null, session_id: "child-session", prompt: "", effort: "",
		ancestors: [], ancestor_ids: [], child_ids: [], peer_ids: [], transcript: [], todos: [], activity: "thinking",
	};
}

describe("DetailRequestCoordinator", () => {
	it("completes an in-flight fetch and coalesces rapid projection versions", async () => {
		const first = deferred<SubagentDetail>();
		const second = deferred<SubagentDetail>();
		const load = vi.fn()
			.mockReturnValueOnce(first.promise)
			.mockReturnValueOnce(second.promise);
		const accepted: number[] = [];
		const loader = new DetailRequestCoordinator(load, (value) => accepted.push(value.version), vi.fn());

		loader.request(1);
		loader.request(2);
		loader.request(3);
		expect(load).toHaveBeenCalledTimes(1);
		first.resolve(detail(1));
		await first.promise;
		await vi.waitFor(() => expect(load).toHaveBeenCalledTimes(2));
		second.resolve(detail(3));
		await second.promise;
		await vi.waitFor(() => expect(accepted).toEqual([1, 3]));
	});

	it("ignores completion after route disposal", async () => {
		const pending = deferred<SubagentDetail>();
		const accept = vi.fn();
		const loader = new DetailRequestCoordinator(() => pending.promise, accept, vi.fn());
		loader.request(1);
		loader.dispose();
		pending.resolve(detail(1));
		await pending.promise;
		await Promise.resolve();
		expect(accept).not.toHaveBeenCalled();
	});
});
