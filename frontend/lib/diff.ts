/**
 * diff.ts — word-level LCS diff for the side-by-side response viewer.
 */

export type DiffPart = { text: string; kind: "same" | "added" | "removed" };

/** Diff two texts by word; returns parts for each side. */
export function wordDiff(a: string, b: string): { left: DiffPart[]; right: DiffPart[] } {
  const aw = a.split(/(\s+)/).filter((w) => w !== "");
  const bw = b.split(/(\s+)/).filter((w) => w !== "");

  // LCS table (capped — very long responses fall back to no highlighting)
  if (aw.length * bw.length > 400_000) {
    return {
      left: [{ text: a, kind: "same" }],
      right: [{ text: b, kind: "same" }],
    };
  }
  const n = aw.length, m = bw.length;
  const dp: Uint32Array[] = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = aw[i] === bw[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const left: DiffPart[] = [];
  const right: DiffPart[] = [];
  const push = (arr: DiffPart[], text: string, kind: DiffPart["kind"]) => {
    const last = arr[arr.length - 1];
    if (last && last.kind === kind) last.text += text;
    else arr.push({ text, kind });
  };
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (aw[i] === bw[j]) {
      push(left, aw[i], "same");
      push(right, bw[j], "same");
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      push(left, aw[i], "removed");
      i++;
    } else {
      push(right, bw[j], "added");
      j++;
    }
  }
  while (i < n) { push(left, aw[i], "removed"); i++; }
  while (j < m) { push(right, bw[j], "added"); j++; }
  return { left, right };
}
