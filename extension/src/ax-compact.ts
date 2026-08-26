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
  function visit(node: AXNode, depth: number): void {
    if (node.ignored) return;
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
