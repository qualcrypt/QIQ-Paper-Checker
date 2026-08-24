/**
 * Which answer belongs to which question.
 *
 * Getting this wrong is worse than getting it uncertain: a confidently
 * misassigned answer marks a student down for something they answered
 * correctly somewhere else on the page. So every route to a match carries a
 * confidence, and the weak routes say so rather than smoothing it over.
 *
 * Three routes, strongest first:
 *   1. the student wrote the question number, and it matches the paper
 *   2. the model assigned the region, validated back against real line ids
 *   3. nothing matched — the question is recorded as unanswered
 *
 * A question with no answer is a real, common outcome. It is reported as
 * skipped and scores zero; it is never quietly given someone else's text.
 */

import { detectQuestionsStructural } from "./segment.js";
import { linesToText } from "./ocr.js";
import { normalizeText, tokenOverlap } from "./text.js";

/**
 * Reduce a printed or handwritten label to a comparable form.
 * "Q.1", "1)", "1." -> "1";  "2 (a)", "2a", "Q2(A)" -> "2(a)"
 */
export function canonicalNumber(raw) {
  let s = String(raw || "").trim().toLowerCase();
  s = s.replace(/^q(?:uestion)?\s*[.\-:]?\s*/i, "");
  s = s.replace(/[.)\]:]+\s*$/, "");
  s = s.replace(/\s+/g, "");

  // "2a" / "2(a)" / "2-a" -> "2(a)";  roman "3ii" -> "3(ii)"
  const m = /^(\d+)[-(]?\s*([a-z]+|[ivx]+)\)?$/.exec(s);
  if (m) return `${m[1]}(${m[2]})`;

  const bare = /^\(?([a-z]|[ivx]+)\)?$/.exec(s);
  if (bare) return `(${bare[1]})`;

  return s;
}

/** Confidence for a match, in the same 0-100 scale the UI already uses. */
const CONFIDENCE = { label: 95, llm: 70, weak: 40, none: 0 };

const MATCH_SCHEMA = {
  type: "object",
  required: ["matches"],
  props: {
    matches: {
      type: "array",
      of: {
        type: "object",
        required: ["questionNumber", "lineIds"],
        props: {
          questionNumber: { type: "string", max: 20 },
          lineIds: { type: "array", of: { type: "string", max: 12 }, min: 1 },
        },
      },
    },
  },
};

const MATCH_SYSTEM =
  "You assign regions of a student's answer sheet to the questions of an exam paper. You only ever " +
  "reference line ids that were given to you, and you leave a question out entirely rather than " +
  "guess at it. You reply with JSON only.";

/**
 * Match the student's answer document against the exam definition.
 *
 * @param {import("./types.js").OCRDocument} doc  the answer sheet
 * @param {{questions: {id,number,text,maxMarks,type}[]}} exam
 * @param {{llm?: {callJson: Function}, warnings?: string[]}} opts
 * @returns {Promise<{answers: object[], unassignedLineIds: string[]}>}
 */
export async function matchAnswers(doc, exam, { llm, warnings = [] } = {}) {
  const byId = new Map(doc.lines.map((l) => [l.id, l]));
  const claimed = new Set();

  /** @type {Map<string, {lineIds: string[], confidence: number, method: string}>} */
  const assigned = new Map();

  /* ---- route 1: the student labelled their answers ---- */
  const { questions: blocks } = detectQuestionsStructural(doc.lines);
  const examByNumber = new Map(exam.questions.map((q) => [canonicalNumber(q.number), q]));

  /* A bare "(a)" continues the last numbered question, which is how students
     actually write sub-parts. */
  let lastMain = null;
  for (const block of blocks) {
    let key = canonicalNumber(block.questionNumber);
    const bare = /^\(([a-z]+|[ivx]+)\)$/.exec(key);
    if (bare && lastMain) key = `${lastMain}(${bare[1]})`;
    else if (/^\d+$/.test(key)) lastMain = key;

    const q = examByNumber.get(key);
    if (!q || assigned.has(q.id)) continue;

    const ids = block.answerLineIds.filter((id) => byId.has(id));
    if (ids.length === 0) continue;

    assigned.set(q.id, { lineIds: ids, confidence: CONFIDENCE.label, method: "label" });
    ids.forEach((id) => claimed.add(id));
  }

  /* ---- route 2: ask the model about what is left ---- */
  const leftover = doc.lines.filter((l) => !claimed.has(l.id));
  const unanswered = exam.questions.filter((q) => !assigned.has(q.id));

  if (llm && unanswered.length > 0 && leftover.length > 0) {
    try {
      const listing = leftover.map((l) => `${l.id}\t${l.text}`).join("\n");
      const questionList = unanswered
        .map((q) => `${q.number}. ${q.text}`.slice(0, 240))
        .join("\n");

      const raw = await llm.callJson({
        stage: "answer matching",
        system: MATCH_SYSTEM,
        user:
          `EXAM QUESTIONS STILL UNMATCHED:\n${questionList}\n\n` +
          `UNASSIGNED LINES FROM THE STUDENT'S ANSWER SHEET:\n${listing}\n\n` +
          `Assign lines to questions they answer. Rules:\n` +
          `- use only ids from the list above\n` +
          `- a question the student did not answer must be OMITTED, not guessed\n` +
          `- lines that answer nothing should be left out\n` +
          `- an answer may span many lines and cross pages\n` +
          `Return JSON: {"matches":[{"questionNumber":"2(a)","lineIds":["L12","L13"]}]}`,
        schema: MATCH_SCHEMA,
        maxTokens: 1800,
      });

      for (const m of raw.matches || []) {
        const q = examByNumber.get(canonicalNumber(m.questionNumber));
        if (!q || assigned.has(q.id)) continue;

        // Only ids that exist and are still free — a hallucinated id is dropped.
        const ids = m.lineIds.filter((id) => byId.has(id) && !claimed.has(id));
        if (ids.length === 0) continue;

        const ordered = ids
          .map((id) => byId.get(id))
          .sort((a, b) => a.index - b.index)
          .map((l) => l.id);

        assigned.set(q.id, { lineIds: ordered, confidence: CONFIDENCE.llm, method: "llm" });
        ordered.forEach((id) => claimed.add(id));
      }
    } catch (e) {
      warnings.push(`Answer matching fell back to labels only (${e.message}).`);
    }
  }

  /* ---- assemble, and cross-check every match against the question text ---- */
  const answers = exam.questions.map((q) => {
    const hit = assigned.get(q.id);

    if (!hit) {
      return {
        questionId: q.id,
        number: q.number,
        questionText: q.text,
        maxMarks: q.maxMarks,
        type: q.type,
        lineIds: [],
        answerText: "",
        skipped: true,
        confidence: CONFIDENCE.none,
        method: "none",
      };
    }

    const answerText = linesToText(doc.lines, hit.lineIds);
    const lines = hit.lineIds.map((id) => byId.get(id)).filter(Boolean);

    /* Lexical agreement between question and answer. It cannot confirm a match
       on its own — a good answer reuses few of the question's words — but a
       near-zero overlap on a model-assigned region is worth doubting. */
    const overlap = tokenOverlap(q.text, answerText);
    let confidence = hit.confidence;
    if (hit.method === "llm" && overlap < 0.04) confidence = CONFIDENCE.weak;

    // Barely any writing is not really an answer.
    const tooShort = normalizeText(answerText).length < 10;
    if (tooShort) confidence = Math.min(confidence, CONFIDENCE.weak);

    return {
      questionId: q.id,
      number: q.number,
      questionText: q.text,
      maxMarks: q.maxMarks,
      type: q.type,
      lineIds: hit.lineIds,
      answerText,
      skipped: tooShort,
      confidence,
      method: hit.method,
      overlap,
      pageStart: lines.length ? lines[0].page : undefined,
      pageEnd: lines.length ? lines[lines.length - 1].page : undefined,
    };
  });

  const skipped = answers.filter((a) => a.skipped);
  if (skipped.length) {
    warnings.push(
      `No answer was found for ${skipped.map((a) => "Q" + a.number).join(", ")}. ` +
        `These score zero — check the answer sheet if that looks wrong.`
    );
  }

  return {
    answers,
    unassignedLineIds: doc.lines.filter((l) => !claimed.has(l.id)).map((l) => l.id),
  };
}
