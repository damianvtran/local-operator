import type { SnapshotRef } from "./state";

export interface AXValue { value?: unknown }
export interface AXNode {
  nodeId: string;
  backendDOMNodeId?: number;
  ignored?: boolean;
  role?: AXValue;
  name?: AXValue;
  childIds?: string[];
  properties?: Array<{ name: string; value: AXValue }>;
}

const LANDMARKS = new Set(["banner", "main", "navigation", "complementary", "contentinfo", "form", "region"]);
const INTERACTIVE = new Set(["button", "link", "textbox", "checkbox", "radio", "combobox", "menuitem", "tab", "switch", "slider", "spinbutton"]);

export function compactAX(nodes: AXNode[], epoch: number): { snapshot: string; refs: Record<string, SnapshotRef> } {
  const byId = new Map(nodes.map((node) => [node.nodeId, node]));
  const refs: Record<string, SnapshotRef> = {};
  const lines: string[] = [];
  let sequence = 0;
  // Guard against cyclic or duplicated childIds in a malformed AX payload:
  // the walk trusts protocol data, and without this a cycle would recurse
  // forever inside the extension's service worker (review round 1, MINOR-1).
  const visited = new Set<string>();
  function visit(node: AXNode, depth: number): void {
    if (visited.has(node.nodeId)) return;
    visited.add(node.nodeId);
    // An ignored node is excluded from the accessible tree but its subtree is
    // NOT: Chrome wraps every real page in ignored generic containers (html and
    // body surface as role "none", ignored: true) sitting directly under the
    // RootWebArea. Returning early here therefore pruned the ENTIRE page and
    // produced the live one-line snapshots ('- RootWebArea "…" [e1]') that
    // survived the Accessibility.enable fix — confirmed against real headful
    // Chrome 151 and Chrome for Testing 145, where getFullAXTree returns a
    // full tree (48 nodes on the repro page) whose nodes 2..3 are ignored
    // wrappers. Treat ignored nodes as transparent: skip the line, keep the
    // depth, and walk through to their children. This is deliberate even for
    // an ignored node carrying `focusable` (e.g. inside aria-hidden): ignored
    // means excluded from the accessible tree, so it emits no line and no ref
    // despite focusable being a ref trigger for rendered nodes below.
    if (node.ignored) {
      for (const child of node.childIds ?? []) {
        const found = byId.get(child);
        if (found) visit(found, depth);
      }
      return;
    }
    const role = String(node.role?.value ?? "");
    const name = String(node.name?.value ?? "").trim();
    const focusable = node.properties?.some((property) => property.name === "focusable" && property.value.value === true);
    const interesting = INTERACTIVE.has(role) || LANDMARKS.has(role) || focusable || Boolean(name && role !== "StaticText");
    if (interesting) {
      let ref = "";
      if ((INTERACTIVE.has(role) || focusable) && node.backendDOMNodeId !== undefined) {
        ref = `e${++sequence}`;
        refs[ref] = { backendNodeId: node.backendDOMNodeId, epoch };
      }
      lines.push(`${"  ".repeat(depth)}- ${role || "node"}${name ? ` ${JSON.stringify(name)}` : ""}${ref ? ` [${ref}]` : ""}`);
    }
    for (const child of node.childIds ?? []) {
      const found = byId.get(child);
      if (found) visit(found, interesting ? depth + 1 : depth);
    }
  }
  const childIds = new Set(nodes.flatMap((node) => node.childIds ?? []));
  for (const root of nodes.filter((node) => !childIds.has(node.nodeId))) visit(root, 0);
  return { snapshot: lines.join("\n"), refs };
}
