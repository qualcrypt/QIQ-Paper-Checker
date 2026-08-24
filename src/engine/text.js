/**
 * Text primitives shared by retrieval, claim extraction and highlight anchoring.
 *
 * The normalisation here is the same one the existing UI already relies on to
 * pin highlights, so anchoring behaviour is preserved exactly: punctuation and
 * whitespace collapse to single spaces while `map` keeps every normalised
 * character pointing back at its original offset.
 */

/** Words too common to carry retrieval signal. */
const STOPWORDS = new Set(
  ("a an the and or but if then than that this these those of in on at to for from by with without " +
    "is are was were be been being it its as such into over under about above below can could should " +
    "would will shall may might must do does did done have has had not no nor so very also which who " +
    "whom whose what when where why how there here their they them he she his her you your we our i")
    .split(/\s+/)
);

export const isStopword = (w) => STOPWORDS.has(w);

/**
 * Lowercase, punctuation-to-space, whitespace-collapsed form plus an index map
 * back into the original string.
 * @param {string} text
 * @returns {{norm: string, map: number[]}}
 */
export function normalize(text) {
  let norm = "";
  const map = [];
  let prevSpace = true;
  const s = String(text || "");
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (/[a-z0-9]/i.test(ch)) {
      norm += ch.toLowerCase();
      map.push(i);
      prevSpace = false;
    } else if (!prevSpace) {
      norm += " ";
      map.push(i);
      prevSpace = true;
    }
  }
  if (norm.endsWith(" ")) {
    norm = norm.slice(0, -1);
    map.pop();
  }
  return { norm, map };
}

export const normalizeText = (s) => normalize(s).norm.trim();

/** Content tokens with stopwords removed and a crude suffix trim. */
export function tokenize(text, { keepStopwords = false } = {}) {
  const words = normalizeText(text).split(" ").filter(Boolean);
  const out = [];
  for (const w of words) {
    if (!keepStopwords && isStopword(w)) continue;
    if (w.length < 2) continue;
    out.push(stem(w));
  }
  return out;
}

/**
 * Deliberately conservative stemmer. A full Porter implementation would be more
 * aggressive than this corpus needs and risks collapsing distinct science terms.
 */
export function stem(word) {
  let w = word;
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && w.endsWith("sses")) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("es") && !/[aeiou]es$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
  if (w.length > 5 && w.endsWith("ed")) return w.slice(0, -2);
  return w;
}

/**
 * Split prose into sentences while keeping the offsets of each one, so a claim
 * can always be traced back to the exact span of the student's answer.
 * @param {string} text
 * @returns {{text: string, start: number, end: number}[]}
 */
export function splitSentences(text) {
  const s = String(text || "");
  const out = [];
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    const isTerminator = ch === "." || ch === "!" || ch === "?" || ch === "\n";
    if (!isTerminator) continue;

    // "6CO2 + 6H2O -> C6H12O6." and "e.g." must not split mid-token.
    if (ch === "." && /\d/.test(s[i - 1] || "") && /\d/.test(s[i + 1] || "")) continue;
    if (ch === "." && /\b[a-z]$/i.test(s.slice(Math.max(0, i - 2), i))) {
      const after = s.slice(i + 1, i + 3);
      if (/^\s*[a-z]/.test(after)) continue; // likely an abbreviation
    }

    let end = i + 1;
    while (end < s.length && /[\s")\]]/.test(s[end]) && s[end] !== "\n") end++;
    const chunk = s.slice(start, end);
    if (chunk.trim().length > 0) out.push({ text: chunk.trim(), start, end });
    start = end;
  }
  const tail = s.slice(start);
  if (tail.trim().length > 0) out.push({ text: tail.trim(), start, end: s.length });

  return out.filter((x) => normalizeText(x.text).length > 0);
}

/**
 * Locate a phrase inside a text tolerantly: exact first, then normalised, then
 * progressively shorter word-prefixes. Returns original-string offsets.
 * @returns {{start: number, end: number, exact: boolean}|null}
 */
export function findSpan(haystack, needle, { from = 0, taken = [] } = {}) {
  const hay = String(haystack || "");
  const raw = String(needle || "");
  if (!raw.trim()) return null;

  const overlaps = (a, b) => taken.some((r) => a < r.end && b > r.start);

  const exactAt = hay.indexOf(raw, from);
  if (exactAt !== -1 && !overlaps(exactAt, exactAt + raw.length)) {
    return { start: exactAt, end: exactAt + raw.length, exact: true };
  }

  const { norm, map } = normalize(hay);
  const target = normalizeText(raw);
  if (target.length < 3) return null;

  for (const cand of prefixCandidates(target)) {
    let cursor = 0;
    for (;;) {
      const hit = norm.indexOf(cand, cursor);
      if (hit === -1) break;
      const start = map[hit];
      const end = map[hit + cand.length - 1] + 1;
      if (start >= from && !overlaps(start, end)) return { start, end, exact: false };
      cursor = hit + 1;
    }
  }
  return null;
}

function prefixCandidates(needle) {
  const words = needle.split(" ").filter(Boolean);
  const out = [needle];
  const floor = Math.max(3, Math.ceil(words.length * 0.4));
  for (let n = words.length - 1; n >= floor; n--) {
    const c = words.slice(0, n).join(" ");
    if (c.length >= 10) out.push(c);
  }
  return out;
}

/** Jaccard overlap of content tokens — a cheap, dependency-free similarity. */
export function tokenOverlap(a, b) {
  const A = new Set(tokenize(a));
  const B = new Set(tokenize(b));
  if (A.size === 0 || B.size === 0) return 0;
  let shared = 0;
  for (const t of A) if (B.has(t)) shared++;
  return shared / (A.size + B.size - shared);
}

/** Deterministic non-cryptographic hash, used for cache keys. */
export function hashString(str) {
  let h = 5381;
  const s = String(str || "");
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}
