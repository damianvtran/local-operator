// @vitest-environment happy-dom
/** Exercise the real Composer and continuation outbox: closing presentation must
 * never acknowledge delivery or turn an uncertain envelope into a fresh send. */
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Composer } from "./composer";
import { getPendingContinuation, clearPendingContinuation } from "../continuation-command";
import { setDraft } from "../store";
import { clearPrivateSessionStorage } from "../private-storage";
import type { SessionProjection } from "../types";

vi.mock("../api", async (original) => ({
  ...await original<typeof import("../api")>(),
  getCommands: vi.fn(async () => ({ commands: [] })),
  getSkills: vi.fn(async () => ({ skills: [] })),
  getDirectories: vi.fn(async () => ({ home: "/", recent: [], tmp: "" })),
}));
vi.mock("./slash-sheet", () => ({ SlashSheet: () => null }));
vi.mock("./model-sheet", () => ({ ModelSheet: () => null }));
vi.mock("./usage-sheet", () => ({ UsageSheet: () => null }));
const projection = { session_id: "dismiss-test", pid: 1, kind: "tui", cwd: "", model_label: "", effort: "", effort_ladder: [], streaming: false, queued_count: 0, transcript: [], usage: {}, version: 1 } as unknown as SessionProjection;
function composer(pid = "dismiss-test") {
  return <Composer pid={pid} projection={{ ...projection, session_id: pid }} onOpenModels={() => {}} onReturnToModels={() => {}} onOpenEffort={() => {}} onCloseEffort={() => {}} effortOpen={false} ended={false} />;
}
const input = () => screen.getByRole("textbox") as HTMLTextAreaElement;
function send(text: string) {
  fireEvent.change(input(), { target: { value: text } });
  fireEvent.keyDown(input(), { key: "Enter", code: "Enter" });
}
beforeEach(() => {
  setDraft("dismiss-test", "");
  setDraft("other-test", "");
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:attachment");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.stubGlobal("createImageBitmap", vi.fn(async () => ({ width: 10, height: 10 })));
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({ drawImage: vi.fn() } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(callback => callback(new Blob(["png"], { type: "image/png" })));
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ message: "temporary refusal" }), { status: 422, headers: { "Content-Type": "application/json" } })));
});
afterEach(() => { cleanup(); clearPendingContinuation("dismiss-test"); clearPendingContinuation("other-test"); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
it("closes a transient refusal without changing the exact draft", async () => {
  render(composer());
  send("/test unchanged arguments");
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(input().value).toBe("/test unchanged arguments");
});

async function attach() {
  fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [new File(["png"], "kept.png", { type: "image/png" })] } });
  await screen.findByRole("button", { name: "remove attachment" });
}
/** Capture the callback INSTALLED on the real DOM button, not a replica of its
 * guard. React's delegated listener always selects the latest props, so invoking
 * an older installed callback explicitly supplies the adversarial queued event. */
function installedDismiss() {
  const node = screen.getByRole("button", { name: "Dismiss" });
  const propsKey = Object.keys(node).find(key => key.startsWith("__reactProps$"))!;
  return (node as unknown as Record<string, { onClick: () => void }>)[propsKey].onClick;
}
function ambiguous() {
  vi.mocked(fetch).mockImplementation(async () => new Response(JSON.stringify({ message: "ack lost" }), { status: 504, headers: { "Content-Type": "application/json" } }));
}
it("keeps typed text and attached image on ordinary failure dismissal", async () => {
  render(composer());
  await attach();
  send(" exact draft with image ");
  await screen.findByRole("alert");
  expect(getPendingContinuation("dismiss-test")).toBeNull();
  // Submission already normalizes whitespace before restoring a refused send;
  // dismissal must preserve the exact new draft the user types afterwards.
  fireEvent.change(input(), { target: { value: " exact draft with image " } });
  const requests = vi.mocked(fetch).mock.calls.length;
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(input().value).toBe(" exact draft with image ");
  expect(screen.getByRole("button", { name: "remove attachment" })).toBeTruthy();
  expect(fetch).toHaveBeenCalledTimes(requests);
});
it("retains uncertain envelope, preview, retry and duplicate-send barrier; identical retry failure reappears", async () => {
  ambiguous();
  render(composer());
  await attach();
  send("original instruction");
  await screen.findByRole("alert");
  const firstCopy = screen.getByRole("alert").textContent;
  const envelope = getPendingContinuation("dismiss-test");
  const stored = localStorage.getItem("lo-mobile-command:dismiss-test");
  expect(envelope?.images).toHaveLength(1);
  fireEvent.change(input(), { target: { value: "next exact draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByText("Resolve the earlier instruction first.")).toBeTruthy();
  expect(screen.getByText("original instruction")).toBeTruthy();
  expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(true);
  expect(localStorage.getItem("lo-mobile-command:dismiss-test")).toBe(stored);
  expect(input().value).toBe("next exact draft");
  expect(screen.getByRole("button", { name: "remove attachment" })).toBeTruthy();
  expect(fetch).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Retry earlier instruction" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("alert").textContent).toBe(firstCopy);
  expect(fetch).toHaveBeenCalledTimes(2);
  const bodies = vi.mocked(fetch).mock.calls.map(([, init]) => JSON.parse(init!.body as string));
  expect(bodies[1]).toEqual(bodies[0]);
  expect(bodies[1].command_id).toBe(envelope?.command_id);
});
it("an installed stale dismiss callback cannot erase a newer identical failure", async () => {
  render(composer());
  send("/test unchanged arguments");
  await screen.findByRole("alert");
  const stale = installedDismiss();
  const copy = screen.getByRole("alert").textContent;
  fireEvent.click(screen.getByRole("button", { name: "send" }));
  await screen.findByRole("alert");
  expect(screen.getByRole("alert").textContent).toBe(copy);
  act(() => stale());
  expect(screen.getByRole("alert").textContent).toBe(copy);
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.queryByRole("alert")).toBeNull();
});
it("rejects a stale dismiss across route changes and authentication remount", async () => {
  const view = render(composer());
  send("/test old account");
  await screen.findByRole("alert");
  const stale = installedDismiss();
  view.rerender(composer("other-test"));
  send("/test new route");
  await screen.findByRole("alert");
  act(() => stale());
  expect(screen.getByRole("alert")).toBeTruthy();
  // The real privacy transition clears storage then reloads (api.ts), so its
  // mounted React tree cannot survive. Model that boundary, not a fake account
  // prop that production Composer never receives.
  view.unmount();
  clearPrivateSessionStorage();
  render(composer());
  send("/test new account");
  await screen.findByRole("alert");
  act(() => stale());
  expect(screen.getByRole("alert")).toBeTruthy();
});
it("keeps empty, populated and sending states free of dismissal chrome; closing another error never cancels a request", async () => {
  render(composer());
  expect(screen.queryByRole("button", { name: "Dismiss" })).toBeNull();
  expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.change(input(), { target: { value: "/test pending" } });
  expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(false);
  let resolve!: (response: Response) => void;
  vi.mocked(fetch).mockImplementation(() => new Promise(r => { resolve = r; }));
  fireEvent.click(screen.getByRole("button", { name: "send" }));
  expect(screen.queryByRole("button", { name: "Dismiss" })).toBeNull();
  // A bad dropped attachment is independent of the still-in-flight slash send.
  fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [new File(["bad"], "bad.txt", { type: "text/plain" })] } });
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(fetch).toHaveBeenCalledTimes(1);
  await act(async () => resolve(new Response(JSON.stringify({ ok: true, detail: "sent" }))));
  await waitFor(() => expect(input().value).toBe(""));
  expect(fetch).toHaveBeenCalledTimes(1);
});
