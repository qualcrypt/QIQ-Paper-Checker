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
import { alignToQuestions, labelKey, canonicalLabel } from "./identity.js";
import { linesToText } from "./ocr.js";
import { normalizeText, tokenOverlap } from "./text.js";

/**
 * Reduce a printed or handwritten label to a comparable form.
 * "Q.1", "1)", "1." -> "1";  "2 (a)", "2a", "Q2(A)" -> "2(a)"
 */
export function canonicalNumber(raw) {
  return canonicalLabel(raw);
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

/* Per-call listing budgets, in characters (~4 chars a token). Sized so prompt
   plus completion budget stays well inside one free-tier minute even when the
   answer sheet runs to many pages. */
const MATCH_LINE_CHARS = 6000;
const MATCH_QUESTION_CHARS = 2000;

/** Split lines into sequential windows that each fit the listing budget. */
export function chunkLines(lines, budget = MATCH_LINE_CHARS) {
  if (lines.length === 0) return [];

  const windows = [];
  let current = [];
  let chars = 0;

  for (const line of lines) {
    const cost = (line.text || "").length + 12; // id + tab + newline
    if (current.length && chars + cost > budget) {
      windows.push(current);
      /* An answer that straddles the cut must still be findable, so each window
         re-shows the tail of the one before it. */
      const overlap = current.slice(-Math.max(1, Math.round(current.length * 0.1)));
      current = overlap.slice();
      chars = overlap.reduce((n, l) => n + (l.text || "").length + 12, 0);
    }
    current.push(line);
    chars += cost;
  }
  if (current.length) windows.push(current);
  return windows;
}

/** Split questions into batches that each fit the listing budget. */
export function chunkQuestions(questions, budget = MATCH_QUESTION_CHARS) {
  const batches = [];
  let current = [];
  let chars = 0;

  for (const q of questions) {
    const cost = Math.min(240, `${q.number}. ${q.text}`.length) + 1;
    if (current.length && (chars + cost > budget || current.length >= 8)) {
      batches.push(current);
      current = [];
      chars = 0;
    }
    current.push(q);
    chars += cost;
  }
  if (current.length) batches.push(current);
  return batches;
}

/**
 * Runs of consecutive unmatched questions, each carrying the line range it can
 * legitimately occupy: after the last line already given to an earlier
 * question, before the first line given to a later one.
 */
export function unmatchedRuns(questions, assigned, indexOf = null) {
  const bound = (q, pick) => {
    const hit = assigned.get(q.id);
    if (!hit || hit.lineIds.length === 0) return null;
    const idx = hit.lineIds.map((id) => (indexOf ? indexOf(id) : null)).filter((n) => n !== null);
    return idx.length ? pick(idx) : null;
  };

  const runs = [];
  let current = null;

  questions.forEach((q, i) => {
    if (assigned.has(q.id)) {
      current = null;
      return;
    }
    if (!current) {
      // The nearest matched question before this one fixes the lower bound.
      let after = -1;
      for (let j = i - 1; j >= 0; j--) {
        const end = bound(questions[j], (a) => Math.max(...a));
        if (end !== null) { after = end; break; }
      }
      current = { questions: [], after, before: Infinity };
      runs.push(current);
    }
    current.questions.push(q);
  });

  // The upper bound is the first matched question that follows the run.
  runs.forEach((run) => {
    const last = questions.indexOf(run.questions[run.questions.length - 1]);
    for (let j = last + 1; j < questions.length; j++) {
      const start = bound(questions[j], (a) => Math.min(...a));
      if (start !== null) { run.before = start; return; }
    }
  });

  return runs;
}

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
  const examByKey = new Map(exam.questions.map((q) => [labelKey(q.number), q]));
  const claimed = new Set();

  /** @type {Map<string, {lineIds: string[], confidence: number, method: string}>} */
  const assigned = new Map();

  /* ---- route 1: align what the student labelled against the paper ----
     The exam is the schema. Rather than parse the student's labels in the
     abstract and hope they come out in the same notation the paper used, every
     detected block is aligned against the known set of questions — so "Ans 3a",
     "Q.3 (a)" and "3 a)" all reach question 3(a), and each route reports what
     it is worth instead of a fixed number. */
  /* The exam's own numbering, handed to detection so a head that is only a head
     on this paper ("1.1") is recognised as one. */
  const known = new Set(exam.questions.map((q) => labelKey(q.number)).filter(Boolean));
  const { questions: blocks } = detectQuestionsStructural(doc.lines, { known });
  const { placed } = alignToQuestions(
    blocks.map((b) => ({ label: b.questionNumber, text: b.answerText })),
    exam.questions
  );

  for (const [questionId, hit] of placed) {
    const block = blocks[hit.index];
    const ids = block.answerLineIds.filter((id) => byId.has(id));
    if (ids.length === 0) continue;

    assigned.set(questionId, { lineIds: ids, confidence: hit.confidence, method: hit.method });
    ids.forEach((id) => claimed.add(id));
  }

  /* ---- route 2: ask the model about what is left ---- */

  /* One call per gap in the paper, never one call for the paper.

     The token ceiling counts prompt *and* completion budget up front, so a
     whole answer sheet listed in a single request is refused outright (413)
     before the model ever sees it — and that one refusal used to leave every
     unlabelled answer unmatched. On a long paper that reads to the examiner as
     "only half of it was checked", which is exactly the failure this pipeline
     exists to prevent. Small calls also fail small: a batch that errors costs
     its own questions, not the paper. */
  const applyMatches = (raw, allowedIds) => {
    let matched = 0;
    for (const m of (raw && raw.matches) || []) {
      const q = examByKey.get(labelKey(m.questionNumber));
      if (!q || assigned.has(q.id) || !allowedIds.has(q.id)) continue;

      // Only ids that exist and are still free — a hallucinated id is dropped.
      const ids = (m.lineIds || []).filter((id) => byId.has(id) && !claimed.has(id));
      if (ids.length === 0) continue;

      const ordered = ids
        .map((id) => byId.get(id))
        .sort((a, b) => a.index - b.index)
        .map((l) => l.id);

      assigned.set(q.id, { lineIds: ordered, confidence: CONFIDENCE.llm, method: "llm" });
      ordered.forEach((id) => claimed.add(id));
      matched++;
    }
    return matched;
  };

  const askAbout = async (batch, lines) => {
    const allowedIds = new Set(batch.map((q) => q.id));
    const questionList = batch.map((q) => `${q.number}. ${q.text}`.slice(0, 240)).join("\n");

    for (const window of chunkLines(lines, MATCH_LINE_CHARS)) {
      const pending = batch.filter((q) => !assigned.has(q.id));
      if (pending.length === 0) return;

      const listing = window.map((l) => `${l.id}\t${l.text}`).join("\n");
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
        maxTokens: Math.min(1800, Math.max(500, 160 * batch.length)),
      });

      applyMatches(raw, allowedIds);
    }
  };

  if (llm) {
    /* Questions are asked in runs, and each run is searched in the stretch of
       the sheet between the answers around it: an unlabelled answer sits where
       the paper's order says it should. When that stretch holds nothing, the
       run is searched against everything still unclaimed rather than being
       written off. */
    const indexOf = (id) => (byId.has(id) ? byId.get(id).index : null);

    for (const run of unmatchedRuns(exam.questions, assigned, indexOf)) {
      const pending = run.questions.filter((q) => !assigned.has(q.id));
      if (pending.length === 0) continue;

      const free = doc.lines.filter((l) => !claimed.has(l.id));
      if (free.length === 0) break;

      const gap = free.filter((l) => l.index > run.after && l.index < run.before);
      const candidates = gap.length ? gap : free;

      for (const batch of chunkQuestions(pending, MATCH_QUESTION_CHARS)) {
        try {
          await askAbout(batch, candidates);
        } catch (e) {
          warnings.push(
            `Answer matching failed for ${batch.map((q) => "Q" + q.number).join(", ")} ` +
              `(${e.message}). They are reported as undetected rather than guessed.`
          );
        }
      }
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

    /* How short is "too short to be an answer"? It depends entirely on what was
       asked. A one-mark fill-in-the-blank is answered in a word — "42", "k = 3",
       an option letter — and a flat ten-character floor called those blank,
       scored them 0 and told the examiner no answer was detected, with the
       student's correct answer sitting right there on the page. The floor now
       follows the marks the question carries. */
    const length = normalizeText(answerText).length;
    const floor = !Number.isFinite(q.maxMarks) || q.maxMarks <= 1 ? 1 : q.maxMarks <= 2 ? 4 : 10;
    const tooShort = length < floor;

    /* Lexical agreement between question and answer. It cannot confirm a match
       on its own — a good answer reuses few of the question's words — but a
       near-zero overlap on a model-assigned region is worth doubting. It is only
       evidence where there are enough words for it to be evidence: a two-word
       answer shares nothing with its question by nature, and downgrading it for
       that was punishing brevity, not detecting a bad match. */
    const overlap = tokenOverlap(q.text, answerText);
    let confidence = hit.confidence;
    if (hit.method === "llm" && length >= 40 && overlap < 0.04) confidence = CONFIDENCE.weak;

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
      `No answer could be confidently detected for ${skipped.map((a) => "Q" + a.number).join(", ")}. ` +
        `This may mean they were left unanswered — or that the writing was missed. Check the answer sheet.`
    );
  }

  return {
    answers,
    unassignedLineIds: doc.lines.filter((l) => !claimed.has(l.id)).map((l) => l.id),
  };
}
