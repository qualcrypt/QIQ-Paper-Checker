/**
 * Question detection and answer segmentation.
 *
 * A paper is not one answer. This splits the OCR document into questions and
 * their answer regions, structurally first and with the LLM only where the
 * structure is genuinely ambiguous — and even then the model's output is
 * validated back against real line ids, so it can never invent a region.
 */

import { linesToText } from "./ocr.js";
import { validate } from "./json.js";
import { normalizeText } from "./text.js";

/**
 * Ordered by specificity. Anchored to line start because a "1." mid-sentence is
 * not a question number.
 */
const PATTERNS = [
  { re: /^\s*(?:Q(?:uestion)?\s*[.\-:]?\s*)(\d+)\s*[.):\-]?\s*/i, kind: "main" },
  { re: /^\s*(\d{1,2})\s*[.)]\s+/, kind: "main" },
  { re: /^\s*\(\s*([a-z])\s*\)\s*/i, kind: "sub" },
  { re: /^\s*\(\s*((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)+)\s*\)\s*/i, kind: "roman" },
  { re: /^\s*([a-z])\s*[.)]\s+/i, kind: "sub" },
];

/** "(5 marks)", "[5]", "- 5 marks", "5 M" */
const MARKS_RE = /[\[(]\s*(\d{1,3})\s*(?:marks?|m)?\s*[\])]|\b(\d{1,3})\s*marks?\b/i;

/**
 * Structural pass over the lines.
 * @param {import("./types.js").OCRLine[]} lines
 * @returns {{questions: import("./types.js").ParsedQuestion[], ambiguous: boolean, reason: string}}
 */
export function detectQuestionsStructural(lines) {
  const heads = [];

  for (const line of lines) {
    for (const { re, kind } of PATTERNS) {
      const m = re.exec(line.text);
      if (!m) continue;

      const label = (m[1] || "").trim();
      // A roman numeral like "i" also matches the single-letter rule; the more
      // specific pattern wins because PATTERNS is ordered.
      heads.push({
        lineId: line.id,
        index: line.index,
        page: line.page,
        kind: kind === "roman" ? "sub" : kind,
        label,
        rest: line.text.slice(m[0].length).trim(),
      });
      break;
    }
  }

  if (heads.length === 0) {
    return { questions: [], ambiguous: true, reason: "no_question_markers" };
  }

  const byIndex = new Map(lines.map((l) => [l.index, l]));
  const questions = [];
  let lastMain = null;

  heads.forEach((head, i) => {
    const next = heads[i + 1];
    const startIdx = head.index;
    const endIdx = next ? next.index - 1 : lines[lines.length - 1].index;

    const bodyIds = [];
    for (let idx = startIdx; idx <= endIdx; idx++) {
      const l = byIndex.get(idx);
      if (l) bodyIds.push(l.id);
    }

    const headLine = byIndex.get(startIdx);
    const marks = extractMarks(head.rest) ?? extractMarks(headLine ? headLine.text : "");

    // The head line often carries the question itself; the rest is the answer.
    const questionText = looksLikeQuestion(head.rest) ? stripMarks(head.rest) : "";
    const answerIds = questionText ? bodyIds.slice(1) : bodyIds;
    const answerTextRaw = linesToText(lines, answerIds);
    const answerText = questionText ? answerTextRaw : stripLeadingLabel(answerTextRaw, head.label);

    const isSub = head.kind === "sub" && lastMain;
    const number = isSub ? `${lastMain.questionNumber}(${head.label})` : head.label;

    const q = {
      id: `q${questions.length + 1}`,
      questionNumber: number,
      questionText,
      answerText,
      answerLineIds: answerIds,
      maxMarks: marks ?? undefined,
      pageStart: head.page,
      pageEnd: byIndex.get(endIdx) ? byIndex.get(endIdx).page : head.page,
      parentId: isSub ? lastMain.id : undefined,
    };
    questions.push(q);
    if (!isSub) lastMain = q;
  });

  // A parent that only introduces sub-parts carries no answer of its own.
  const withChildren = new Set(questions.filter((q) => q.parentId).map((q) => q.parentId));
  const cleaned = questions.filter(
    (q) => !(withChildren.has(q.id) && normalizeText(q.answerText).length < 12)
  );

  const ambiguous = cleaned.length === 0;
  return {
    questions: cleaned,
    ambiguous,
    reason: ambiguous ? "markers_without_answers" : "",
  };
}

function looksLikeQuestion(text) {
  const t = String(text || "").trim();
  if (t.length < 8) return false;
  if (t.endsWith("?")) return true;
  return /^(explain|describe|define|what|why|how|discuss|compare|differentiate|state|list|write|analyse|analyze|evaluate|justify|give|calculate|find|prove|derive|draw)\b/i.test(
    t
  );
}

function stripMarks(text) {
  return String(text || "").replace(MARKS_RE, "").replace(/\s{2,}/g, " ").trim();
}

function stripLeadingLabel(text, label) {
  if (!label) return text;
  return String(text || "").replace(new RegExp(`^\\s*\\(?${escapeRe(label)}\\)?\\s*[.)-]?\\s*`, "i"), "");
}

const escapeRe = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function extractMarks(text) {
  const m = MARKS_RE.exec(String(text || ""));
  if (!m) return null;
  const n = Number(m[1] ?? m[2]);
  return Number.isFinite(n) && n > 0 && n <= 200 ? n : null;
}

/* ----------------------------------------------------------- LLM assist -- */

const SEGMENT_SCHEMA = {
  type: "object",
  required: ["questions"],
  props: {
    questions: {
      type: "array",
      of: {
        type: "object",
        required: ["questionNumber", "answerLineIds"],
        props: {
          questionNumber: { type: "string", max: 20 },
          questionText: { type: "string", max: 600, default: "" },
          answerLineIds: { type: "array", of: { type: "string", max: 12 }, min: 1 },
          maxMarks: { type: "number", min: 0, max: 200, optional: true },
        },
      },
      min: 1,
    },
  },
};

const SEGMENT_SYSTEM =
  "You split scanned exam papers into questions. You only ever reference line ids that were given " +
  "to you. You never invent text. You reply with JSON only.";

/**
 * Ask the model to segment, then rebuild every region from real lines. Any id
 * the model hallucinates is discarded rather than trusted.
 * @param {import("./types.js").OCRLine[]} lines
 * @param {{callJson: Function}} llm
 */
export async function detectQuestionsWithLlm(lines, llm) {
  const listing = lines.map((l) => `${l.id}\t${l.text}`).join("\n");

  const user =
    `Here are the numbered lines of a scanned answer paper.\n\n${listing}\n\n` +
    `Group them into questions. For each question give:\n` +
    `- "questionNumber": the label as printed (e.g. "1", "2(a)", "3(ii)")\n` +
    `- "questionText": the question wording if the paper contains it, else ""\n` +
    `- "answerLineIds": the ids of the lines forming the STUDENT'S ANSWER only\n` +
    `- "maxMarks": the marks printed for the question, omit if not printed\n\n` +
    `Use only ids from the list. If the whole page is one answer, return one question.\n` +
    `Return JSON: {"questions":[...]}`;

  const raw = await llm.callJson({
    stage: "segment",
    system: SEGMENT_SYSTEM,
    user,
    schema: SEGMENT_SCHEMA,
    maxTokens: 1500,
  });

  const known = new Map(lines.map((l) => [l.id, l]));
  const questions = [];

  raw.questions.forEach((q, i) => {
    const ids = q.answerLineIds.filter((id) => known.has(id));
    if (ids.length === 0) return; // entirely hallucinated region

    const ordered = ids
      .map((id) => known.get(id))
      .sort((a, b) => a.index - b.index)
      .map((l) => l.id);

    questions.push({
      id: `q${questions.length + 1}`,
      questionNumber: q.questionNumber || String(i + 1),
      questionText: q.questionText || "",
      answerText: linesToText(lines, ordered),
      answerLineIds: ordered,
      maxMarks: q.maxMarks,
      pageStart: known.get(ordered[0]).page,
      pageEnd: known.get(ordered[ordered.length - 1]).page,
    });
  });

  return questions;
}

/**
 * Full segmentation with graceful degradation:
 *   structural pass → LLM only if ambiguous → whole paper as one question.
 * The paper is always segmented into at least one question; this never throws.
 *
 * @param {import("./types.js").OCRDocument} doc
 * @param {{llm?: {callJson: Function}, totalMarks?: number, warnings?: string[]}} opts
 * @returns {Promise<import("./types.js").ParsedQuestion[]>}
 */
export async function segmentPaper(doc, { llm, totalMarks, warnings = [] } = {}) {
  const structural = detectQuestionsStructural(doc.lines);

  let questions = structural.questions;

  if (structural.ambiguous && llm) {
    try {
      const viaLlm = await detectQuestionsWithLlm(doc.lines, llm);
      if (viaLlm.length > 0) questions = viaLlm;
    } catch (e) {
      warnings.push(`Question segmentation fell back to single-question mode (${e.message}).`);
    }
  }

  if (questions.length === 0) {
    questions = [
      {
        id: "q1",
        questionNumber: "1",
        questionText: "",
        answerText: doc.text,
        answerLineIds: doc.lines.map((l) => l.id),
        maxMarks: totalMarks,
        pageStart: 1,
        pageEnd: doc.pageCount,
      },
    ];
  }

  return allocateMarks(questions, totalMarks);
}

/**
 * Make per-question marks add up to the paper total exactly.
 *
 * Printed marks are authoritative where present. Anything unlabelled shares the
 * remainder, and a largest-remainder pass removes the rounding drift so the
 * paper total is never off by one.
 */
export function allocateMarks(questions, totalMarks) {
  const total = Number(totalMarks);
  if (!Number.isFinite(total) || total <= 0) {
    return questions.map((q) => ({ ...q, maxMarks: q.maxMarks ?? 10 }));
  }

  const known = questions.filter((q) => Number.isFinite(q.maxMarks) && q.maxMarks > 0);
  const unknown = questions.filter((q) => !(Number.isFinite(q.maxMarks) && q.maxMarks > 0));
  const knownSum = known.reduce((s, q) => s + q.maxMarks, 0);

  // Every question was labelled: rescale only if the labels disagree with the total.
  if (unknown.length === 0) {
    if (knownSum === total || knownSum === 0) return questions.map((q) => ({ ...q }));
    const factor = total / knownSum;
    const scaled = questions.map((q) => ({ ...q, maxMarks: q.maxMarks * factor }));
    return roundToTotal(scaled, total);
  }

  const remaining = Math.max(0, total - knownSum);
  const share = remaining / unknown.length;
  const filled = questions.map((q) =>
    Number.isFinite(q.maxMarks) && q.maxMarks > 0 ? { ...q } : { ...q, maxMarks: share }
  );
  return roundToTotal(filled, total);
}

/** Largest-remainder rounding so integers still sum to `total`. */
function roundToTotal(questions, total) {
  const floors = questions.map((q) => Math.max(1, Math.floor(q.maxMarks)));
  let sum = floors.reduce((a, b) => a + b, 0);

  const remainders = questions
    .map((q, i) => ({ i, frac: q.maxMarks - Math.floor(q.maxMarks) }))
    .sort((a, b) => b.frac - a.frac);

  let k = 0;
  while (sum < total && remainders.length) {
    floors[remainders[k % remainders.length].i] += 1;
    sum++;
    k++;
  }
  while (sum > total) {
    let reduced = false;
    for (let i = floors.length - 1; i >= 0; i--) {
      if (floors[i] > 1) {
        floors[i] -= 1;
        sum--;
        reduced = true;
        if (sum === total) break;
      }
    }
    if (!reduced) break; // every question is already at the floor of 1
  }

  return questions.map((q, i) => ({ ...q, maxMarks: floors[i] }));
}
