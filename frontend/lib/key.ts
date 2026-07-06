/**
 * key.ts — visitor-supplied OpenRouter key (demo-mode deployments).
 *
 * The deployed backend ships with no server-side key; visitors paste their
 * own, which lives ONLY in their browser's localStorage and is sent
 * per-request as the `api_key` query param the backend already accepts.
 */

const KEY = "llm-eval:api-key";

export function getVisitorKey(): string {
  if (typeof window === "undefined") return "";
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}

export function setVisitorKey(value: string): void {
  try {
    if (value.trim()) localStorage.setItem(KEY, value.trim());
    else localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

/** Append the visitor key to a stream URL when one is set. */
export function withVisitorKey(url: string): string {
  const k = getVisitorKey();
  return k ? `${url}&api_key=${encodeURIComponent(k)}` : url;
}
