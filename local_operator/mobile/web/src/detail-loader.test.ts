import { describe, expect, it, vi } from "vitest";
import { DetailRequestCoordinator } from "./detail-loader";
import type { SubagentDetail } from "./types";

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;
	const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
	return { promise, resolve, reject };
}

function detail(version: number): SubagentDetail {
	return {
		version, job_id: "child", label: "child", agent: "coder", status: "running",
		progress: "", elapsed_s: 1, model_label: "", result_text: "", error_text: "",
		parent_job_id: null, session_id: "child-session", prompt: "", launch_message_id: "", effort: "",
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

	it("retains reconnect retry while the same-version request is still active", async () => {
		const first = deferred<SubagentDetail>();
		const second = deferred<SubagentDetail>();
		const load = vi.fn()
			.mockReturnValueOnce(first.promise)
			.mockReturnValueOnce(second.promise);
		const accept = vi.fn();
		const reject = vi.fn();
		const loader = new DetailRequestCoordinator(load, accept, reject);

		loader.request(41);
		loader.request(41);
		expect(load).toHaveBeenCalledTimes(1);
		first.reject(new Error("relay unavailable"));
		await expect(first.promise).rejects.toThrow("relay unavailable");
		await vi.waitFor(() => expect(load).toHaveBeenCalledTimes(2));
		second.resolve(detail(41));
		await second.promise;
		await vi.waitFor(() => expect(accept).toHaveBeenCalledWith(detail(41)));
		expect(load).toHaveBeenCalledTimes(2);
	});

	it("does not retry a successful version only because reconnect overlapped it", async () => {
		const pending = deferred<SubagentDetail>();
		const load = vi.fn(() => pending.promise);
		const accept = vi.fn();
		const loader = new DetailRequestCoordinator(load, accept, vi.fn());

		loader.request(41);
		loader.request(41);
		pending.resolve(detail(41));
		await pending.promise;
		await vi.waitFor(() => expect(accept).toHaveBeenCalledOnce());
		expect(load).toHaveBeenCalledTimes(1);
	});

	it("retries a failed version after reconnect without a version advance", async () => {
		const first = deferred<SubagentDetail>();
		const second = deferred<SubagentDetail>();
		const load = vi.fn()
			.mockReturnValueOnce(first.promise)
			.mockReturnValueOnce(second.promise);
		const accept = vi.fn();
		const reject = vi.fn();
		const loader = new DetailRequestCoordinator(load, accept, reject);

		loader.request(41);
		first.reject(new Error("relay unavailable"));
		await expect(first.promise).rejects.toThrow("relay unavailable");
		await vi.waitFor(() => expect(reject).toHaveBeenCalledOnce());

		loader.request(41);
		expect(load).toHaveBeenCalledTimes(2);
		second.resolve(detail(41));
		await second.promise;
		await vi.waitFor(() => expect(accept).toHaveBeenCalledWith(detail(41)));
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
