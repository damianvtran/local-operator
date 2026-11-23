/**
 * Which fenced code blocks the transcript draws as diagrams.
 *
 * Only a CLOSED ```mermaid fence with something inside it. An open fence is a
 * reply that is still streaming in: drawing it would flash a parse error on
 * every token until the closing line arrives, so it stays the plain code block
 * it has always been. The desktop UI draws on the same rule.
 */
export interface FenceSummary {
	/** The word after the opening backticks, if any. */
	lang: string | undefined;
	/** The closing line was seen. An open fence reaches the end of the text. */
	closed: boolean;
	/** The fence body, lines joined with "\n". */
	source: string;
}

export function shouldDrawDiagram(fence: FenceSummary): boolean {
	return fence.closed && fence.lang?.toLowerCase() === "mermaid" && fence.source.trim() !== "";
}

/**
 * A stable React key for one diagram: its source plus how many identical
 * sources came before it in the same message. Ordinary Markdown blocks use
 * their source offset, which would tear a diagram down when preceding prose
 * is corrected without changing the fence.
 */
export function diagramKey(source: string, occurrence: number): string {
	return `diagram:${occurrence}:${source}`;
}
