export type StoredVerdict = "allow" | "deny";

export function safeHttpUrl(raw: unknown): URL {
  if (typeof raw !== "string") throw new Error("URL is required");
  const parsed = new URL(raw);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("only http:// and https:// can be opened");
  }
  return parsed;
}

export function storedOriginAllowed(
  origins: Record<string, StoredVerdict>,
  url: URL,
): boolean {
  return origins[url.origin] === "allow";
}
