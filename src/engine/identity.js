/**
 * Which question is this?
 *
 * Every stage of the pipeline has to answer that: the answer sheet says "3(a)",
 * the question paper says "Q.3 (a)", the model answers say "Ans 3a". Until this
 * module existed each stage answered it on its own, with its own regular
 * expression, and agreement between them was a coincidence of formatting. A
 * paper labelled any other way — "Question Five", "1.1", "Q1 (i)", a Roman
 * "IV" — parsed differently in each place and silently failed to line up.
 *
 * The fix is to stop parsing labels in the abstract. The question paper is the
 * schema: it states the exact set of questions that exist. Every other document
 * is *aligned to that set* rather than parsed independently and then compared as
 * strings. Deciding "which of these twelve known questions is this block" is a
 * constrained problem with a right answer; parsing arbitrary label syntax is
 * not.
 *
 * Alignment runs in layers, strongest first, and each layer declares what it is
 * worth. Nothing here guesses silently: a block no layer can place comes back
 * unplaced, which the pipeline already knows how to report.
 */

import { normalizeText, tokenOverlap } from "./text.js";

/** What each route is worth, on the 0-100 scale the whole app already uses. */
export const IDENTITY = {
  LABEL: 95, // the labels agree once written the same way
  LOOSE: 88, // they agree after word/roman numerals and separators are resolved
  UNIQUE: 76, // one exam question, and only one, could carry this label
  CONTENT: 62, // no usable label; the text itself matches one question best
  NONE: 0,
};

/** Spelled-out numbers, shared with the rubric parser in choice.js. */
export const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
  seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8,
  ninth: 9, tenth: 10,
};

const ROMAN = {
  i: 1, ii: 2, iii: 3, iv: 4, v: 5, vi: 6, vii: 7, viii: 8, ix: 9, x: 10,
  xi: 11, xii: 12, xiii: 13, xiv: 14, xv: 15, xvi: 16, xvii: 17, xviii: 18, xix: 19, xx: 20,
};

/** Words that introduce a label rather than being one. */
/* The lookahead is what makes "Q1" work as well as "Q. 1": the keyword may be
   followed immediately by the number, but it must not be the start of a longer
   word ("part" is stripped, "particle" is not). */
const LEAD = /^(?:q(?:ue?s(?:tion)?)?|ans(?:wer)?|sol(?:ution)?|part|sec(?:tion)?)(?=[^a-z]|$)[\s.\-:)]*/i;

/**
 * A label reduced to the trail of levels it names.
 * "Q.3 (a) (ii)" -> ["3","a","ii"];  "Question Five" -> ["5"];  "1.1" -> ["1","1"]
 *
 * Returning the trail rather than one string is what lets a paper number its
 * parts one way and a key number them another and still line up.
 *
 * @param {string} raw
 * @returns {string[]}
 */
export function labelParts(raw) {
  let s = String(raw || "").trim().toLowerCase();
  if (!s) return [];

  // Strip every leading "Q"/"Ans"/"Section" word, however many are stacked up.
  let before;
  do {
    before = s;
    s = s.replace(LEAD, "");
  } while (s !== before && s);

  // A spelled-out number is a number.
  s = s.replace(/\b([a-z]+)\b/g, (w) => (NUMBER_WORDS[w] !== undefined ? String(NUMBER_WORDS[w]) : w));

  // Anything that is not a level marker is a separator.
  const parts = s.split(/[^a-z0-9]+/).filter(Boolean);

  const out = [];
  for (const part of parts) {
    // "3a" and "3ii" are two levels written without a separator between them.
    const split = /^(\d+)([a-z]+)$/.exec(part);
    if (split) {
      out.push(split[1], split[2]);
      continue;
    }
    out.push(part);
  }
  /* A level marker is a number, a Roman numeral or a single letter. Anything
     else is a word that happened to sit next to the label — "Qn", a section
     name, the first word of the question itself — and treating it as a level
     would invent a question number out of prose. */
  return out.filter((p) => /^\d+$/.test(p) || ROMAN[p] !== undefined || /^[a-z]$/.test(p));
}

/** Level tokens, as they may appear before being normalised. */
const TOKEN = /^(\d{1,3}|[ivx]{1,6}|[a-z])(?![a-z])/i;
/* A spelled-out number is only a level where a lead word announced one —
   "Answer Two:" is a label, the "Second" in "ii. Second point" is a word. */
const TOKEN_WORD = new RegExp(`^(${Object.keys(NUMBER_WORDS).join("|")})(?![a-z])`, "i");
/** "2a", "3ii" — a level and its sub-level written without a separator. */
const TOKEN_PAIR = /^(\d{1,3})([ivx]{1,6}|[a-z])(?![a-z])/i;

/**
 * Read a question label off the front of a line, if there is one.
 *
 * This is the only place that knows what a written label looks like. Detection
 * used to live in segment.js as a fixed list of patterns, so a paper that wrote
 * "Ans 1." or "Answer Two:" produced no head at all and the answer under it was
 * invisible to every later stage. A scanner is used rather than one large
 * regular expression because the shapes compose — a lead word, then levels,
 * bracketed or dotted, to any depth.
 *
 * `known` closes the one genuinely ambiguous case. "1.1" and "3.5 kg of water"
 * are the same shape, so an unpunctuated numeric label cannot be told from a
 * decimal quantity in isolation — but it can be told in context: the exam says
 * which questions exist, and a paper that actually asks question 1.1 makes
 * "1.1" a label. Detection is ambiguous; detection against a known set is not.
 *
 * @param {string} line
 * @param {{known?: Set<string>}} [opts]  label keys the paper is known to have
 * @returns {{label: string, rest: string, bracketed: boolean, lead: boolean}|null}
 */
export function readLabel(line, opts = {}) {
  const known = opts.known instanceof Set ? opts.known : null;
  const s = String(line || "");
  let i = 0;

  const take = (re) => {
    const m = re.exec(s.slice(i));
    if (!m) return null;
    i += m[0].length;
    return m;
  };

  take(/^\s+/);
  /* Lead words stack: "Q. No. 7" and "Answer to Question 3" both open with two
     of them before the number arrives. */
  let lead = false;
  while (take(/^(?:q(?:ue?s(?:tion)?)?|ans(?:wer)?|sol(?:ution)?|part|sec(?:tion)?|n(?:o|um(?:ber)?)?|to)(?=[^a-z]|$)\s*[.\-:)]?\s*/i)) {
    lead = true;
  }

  const levels = [];
  let bracketed = false;
  let terminated = false;

  for (let depth = 0; depth < 4; depth++) {
    const open = take(/^\s*[([]\s*/);
    if (open) bracketed = true;

    const pair = depth === 0 ? take(TOKEN_PAIR) : null;
    if (pair) {
      levels.push(pair[1], pair[2]);
    } else {
      const tok = (depth === 0 && lead ? take(TOKEN_WORD) : null) || take(TOKEN);
      if (!tok) break;
      levels.push(tok[1]);
    }

    const close = take(/^\s*[)\]]/);
    if (close) {
      bracketed = true;
      terminated = true;
      take(/^\s*/);
      if (/^[([]/.test(s.slice(i))) continue; // another bracketed level follows
      break;
    }

    /* "3." / "3:" / "3 -" ends the label; "3.1" and "3 a)" continue it, so a
       separator followed by another token is not a terminator yet. */
    const sep = take(/^\s*[.:\-]\s*/);
    if (sep) {
      terminated = true;
      /* "1.1" and "3.(a)" carry on; "2. k = 3" and "1. A cell divides" do not.
         A bare letter after a full stop is the start of the answer far more
         often than it is a sub-level, and reading it as one both mangles the
         label and eats the first word of the student's writing. */
      if (/^\d/.test(s.slice(i)) || /^[([]/.test(s.slice(i))) {
        terminated = false;
        continue;
      }
      break;
    }

    if (/^\s*[([]/.test(s.slice(i))) continue; // "3 (a)"
    // "2 a)" — a space, then a sub-level that closes properly.
    if (take(/^\s+(?=(?:[a-z]|[ivx]{1,6})\s*[).\]])/i)) continue;
    break;
  }

  if (levels.length === 0) return null;

  const rest = s.slice(i).trim();
  const marked = lead || bracketed || terminated;
  const allNumeric = levels.every(
    (l) => /^\d+$/.test(l) || NUMBER_WORDS[l.toLowerCase()] !== undefined
  );

  const label = s.slice(0, i).trim();

  /* Prose defends itself here. "A cell divides…" and "Five plants were grown…"
     open exactly like a label, so a token with nothing marking it as one — no
     lead word, no bracket, no terminator — is not a label.
     "3a" survives that test because a digit followed by a letter is not a way
     anyone writes a quantity; "1.1" does not, because "3.5 kg of water" is.
     Unless, that is, the paper being marked actually asks a question 1.1 — then
     the shape is no longer ambiguous and the guard would be discarding a real
     head. That is what `known` is for. */
  const recognised = known ? known.has(labelKey(label)) : false;
  if (!marked && !recognised && !(levels.length > 1 && !allNumeric)) return null;
  if (!rest && !marked && !recognised) return null;

  return { label, rest, bracketed, lead };
}

/**
 * One comparable form of a label. Two labels naming the same question produce
 * the same key whichever notation each of them used.
 *
 * Roman numerals and letters are deliberately not merged: "(i)" and "(a)" are
 * different sub-parts on papers that use both.
 */
export function labelKey(raw) {
  const parts = labelParts(raw);
  if (parts.length === 0) return "";
  return parts
    .map((p) => {
      if (/^\d+$/.test(p)) return String(Number(p));
      if (ROMAN[p] !== undefined) return `r${ROMAN[p]}`; // a roman level
      return p; // a letter level
    })
    .join(".");
}

/**
 * The historical display form, kept because it is what stored evaluations and
 * the report card already speak: "1", "2(a)", "3(ii)".
 */
export function canonicalLabel(raw) {
  const parts = labelParts(raw);
  if (parts.length === 0) return "";
  const [head, ...rest] = parts;
  const norm = (p) => (/^\d+$/.test(p) ? String(Number(p)) : p);
  if (rest.length === 0) return norm(head);
  return `${norm(head)}${rest.map((p) => `(${norm(p)})`).join("")}`;
}

/** Two level markers naming the same thing: "2" and "two", "ii" and "2". */
function samePart(a, b) {
  if (a === b) return true;
  const na = ROMAN[a] !== undefined ? ROMAN[a] : /^\d+$/.test(a) ? Number(a) : null;
  const nb = ROMAN[b] !== undefined ? ROMAN[b] : /^\d+$/.test(b) ? Number(b) : null;
  /* A Roman "ii" and a digit "2" name the same sub-part; a letter "b" does not,
     even though it is the second letter — papers use letters and numbers as
     distinct levels, and merging them invents sub-parts nobody wrote. */
  if (na !== null && nb !== null) return na === nb;
  return false;
}

/**
 * Align labelled blocks to the questions the paper actually has.
 *
 * @param {{label?: string, text?: string}[]} blocks  what some document split into
 * @param {{id: string, number: string, text?: string}[]} questions  the exam: the schema
 * @param {{minContent?: number, contentMargin?: number}} [opts]
 * @returns {{placed: Map<string, {index: number, confidence: number, method: string}>,
 *            unplaced: number[]}}
 *   `placed` is keyed by question id; `unplaced` holds the indexes of blocks
 *   nothing could claim.
 */
export function alignToQuestions(blocks, questions, opts = {}) {
  const minContent = opts.minContent === undefined ? 0.12 : opts.minContent;
  const contentMargin = opts.contentMargin === undefined ? 1.4 : opts.contentMargin;

  const placed = new Map();
  const taken = new Set();

  const qKeys = questions.map((q) => labelKey(q.number));
  const qParts = questions.map((q) => labelParts(q.number));

  const claim = (qi, bi, confidence, method) => {
    const q = questions[qi];
    if (!q || placed.has(q.id) || taken.has(bi)) return false;
    placed.set(q.id, { index: bi, confidence, method });
    taken.add(bi);
    return true;
  };

  const free = (qi) => !placed.has(questions[qi].id);

  /* ---- layer 1: the labels agree ---- */
  blocks.forEach((b, bi) => {
    const key = labelKey(b.label);
    if (!key) return;
    const qi = qKeys.indexOf(key);
    if (qi >= 0) claim(qi, bi, IDENTITY.LABEL, "label");
  });

  /* ---- layer 2: they agree once the notation is resolved ----
     Depth is where documents disagree most: a key files "5" where the paper
     asks "5(a)" because that question has only one part. */
  blocks.forEach((b, bi) => {
    if (taken.has(bi)) return;
    const parts = labelParts(b.label);
    if (parts.length === 0) return;

    const candidates = questions
      .map((_, qi) => qi)
      .filter(free)
      .filter((qi) => {
        const qp = qParts[qi];
        if (qp.length === 0) return false;
        const short = qp.length < parts.length ? qp : parts;
        const long = qp.length < parts.length ? parts : qp;
        return short.every((p, i) => samePart(p, long[i]));
      });

    if (candidates.length === 1) claim(candidates[0], bi, IDENTITY.LOOSE, "label-loose");
  });

  /* ---- layer 3: only one question could carry this label at all ---- */
  blocks.forEach((b, bi) => {
    if (taken.has(bi)) return;
    const parts = labelParts(b.label);
    if (parts.length === 0) return;

    const candidates = questions
      .map((_, qi) => qi)
      .filter(free)
      .filter((qi) => qParts[qi][0] !== undefined && samePart(qParts[qi][0], parts[0]));

    if (candidates.length === 1) claim(candidates[0], bi, IDENTITY.UNIQUE, "label-partial");
  });

  /* ---- layer 4: no usable label, so the content decides ----
     Only when one question is clearly the best fit. A block that suits two
     questions equally well is left unplaced: "we could not find this answer" is
     true, and picking one of them is not. */
  blocks.forEach((b, bi) => {
    if (taken.has(bi)) return;
    const text = normalizeText(b.text || "");
    if (text.length < 20) return;

    const scored = questions
      .map((q, qi) => ({ qi, score: free(qi) ? tokenOverlap(q.text || "", text) : -1 }))
      .filter((x) => x.score > 0)
      .sort((a, c) => c.score - a.score);

    if (scored.length === 0) return;
    const best = scored[0];
    const runnerUp = scored[1] ? scored[1].score : 0;
    if (best.score < minContent) return;
    if (runnerUp > 0 && best.score < runnerUp * contentMargin) return;

    claim(best.qi, bi, IDENTITY.CONTENT, "content");
  });

  return {
    placed,
    unplaced: blocks.map((_, i) => i).filter((i) => !taken.has(i)),
  };
}
