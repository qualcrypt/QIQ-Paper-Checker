/**
 * The question paper, understood.
 *
 * Marking starts here rather than at the student's answer: the paper is what
 * says how many questions exist, what each one asks, and what each is worth.
 * Everything downstream — matching, criteria, mark ceilings, the report total —
 * is derived from this structure.
 *
 * Two passes, and the deterministic one is the cross-check rather than the
 * fallback: the LLM reads the paper (handwritten or typeset, numbering styles
 * vary far too much for regex alone), and segment.js's patterns confirm the
 * numbering and printed marks independently. Disagreements become warnings, not
 * silent corrections.
 */

import { detectQuestionsStructural, extractMarks } from "./segment.js";
import { normalizeText } from "./text.js";

/**
 * What each kind of question is marked on. Deliberately shallow — this steers
 * the examiner's attention, it is not a scoring framework.
 */
export const CRITERIA = {
  definition: ["key concept", "accuracy"],
  short_answer: ["required points", "accuracy"],
  explain: ["key concepts", "explanation", "completeness", "accuracy"],
  describe: ["key concepts", "detail", "completeness", "accuracy"],
  compare: ["required comparison points", "similarities", "differences", "accuracy"],
  differentiate: ["required comparison points", "differences", "accuracy"],
  numerical: ["method", "working", "final answer", "accuracy"],
  diagram: ["required elements", "labelling", "accuracy"],
  evaluate: ["concept coverage", "reasoning", "judgement", "accuracy"],
  justify: ["claim", "supporting reasons", "accuracy"],
  discuss: ["concept coverage", "reasoning", "balance", "accuracy"],
  analyze: ["concept coverage", "reasoning", "accuracy"],
  cause_effect: ["causes", "effects", "linkage", "accuracy"],
  essay: ["concept coverage", "accuracy", "reasoning", "structure", "examples", "conclusion"],
  long_answer: ["concept coverage", "accuracy", "reasoning", "structure", "examples"],
};

/* Read the instruction verb, which is what actually sets the expected shape of
   an answer. Ordered: the more specific verb wins. */
const TYPE_RULES = [
  [/\b(compare|contrast|distinguish|differentiate|difference between)\b/i, "compare"],
  [/\b(define|definition|what is meant by)\b/i, "definition"],
  [/\b(derive|prove|calculate|compute|solve|find the value)\b/i, "numerical"],
  [/\b(draw|sketch|label(?:led)?\s+diagram|diagram)\b/i, "diagram"],
  [/\b(justify|comment on)\b/i, "justify"],
  [/\b(critically|evaluate|assess)\b/i, "evaluate"],
  [/\b(analyse|analyze)\b/i, "analyze"],
  [/\b(discuss)\b/i, "discuss"],
  [/\b(why|reason for|causes? of|effects? of)\b/i, "cause_effect"],
  [/\b(describe)\b/i, "describe"],
  [/\b(explain|elaborate|account for)\b/i, "explain"],
  [/\b(list|state|name|mention|enumerate|write down)\b/i, "short_answer"],
];

/**
 * Classify a question. Falls back to the mark allocation, which is the other
 * honest signal of how much answer is expected.
 * @returns {string} a QuestionType from types.js
 */
export function classifyQuestion(text, maxMarks) {
  const t = String(text || "");
  for (const [re, type] of TYPE_RULES) if (re.test(t)) return type;

  const m = Number(maxMarks);
  if (!Number.isFinite(m)) return "short_answer";
  if (m <= 2) return "definition";
  if (m <= 5) return "explain";
  if (m <= 10) return "long_answer";
  return "essay";
}

/** The criteria list for a question, always non-empty. */
export const criteriaFor = (type) => CRITERIA[type] || CRITERIA.explain;

/* ------------------------------------------------------------ extraction -- */

const EXAM_SCHEMA = {
  type: "object",
  required: ["questions"],
  props: {
    title: { type: "string", max: 200, default: "" },
    subject: { type: "string", max: 80, default: "" },
    totalMarks: { type: "number", min: 0, max: 2000, optional: true },
    questions: {
      type: "array",
      min: 1,
      max: 100,
      of: {
        type: "object",
        required: ["number", "text"],
        props: {
          number: { type: "string", max: 20 },
          text: { type: "string", max: 900 },
          maxMarks: { type: "number", min: 0, max: 200, optional: true },
        },
      },
    },
  },
};

const EXAM_SYSTEM =
  "You read examination question papers and return their structure as JSON. You copy question " +
  "wording verbatim. You never invent a question and you never invent a mark that is not printed " +
  "on the paper. You reply with JSON only.";

/**
 * Ask the model to read the question paper.
 *
 * `maxMarks` is deliberately optional in the schema: a paper that does not print
 * per-question marks must come back with them absent, so validateExam can say so.
 * Inventing them here would hide the problem behind a plausible number.
 *
 * @param {string} paperText  OCR text of the question paper.
 * @param {{callJson: Function}} llm
 */
export async function extractExamWithLlm(paperText, llm) {
  const user =
    `Here is the text of an examination question paper.\n\n${paperText}\n\n` +
    `Return its structure as JSON:\n` +
    `- "title": the paper's title if printed, else ""\n` +
    `- "subject": the subject if printed, else ""\n` +
    `- "totalMarks": the total printed on the paper, omit if not printed\n` +
    `- "questions": one entry per question OR sub-question that a student answers separately\n` +
    `    - "number": the label exactly as printed, e.g. "1", "2(a)", "3(ii)"\n` +
    `    - "text": the question wording, copied verbatim\n` +
    `    - "maxMarks": the marks printed for that question — OMIT THIS FIELD if the paper does not print it\n\n` +
    `If a question has lettered or numbered parts that are answered separately, list the parts, ` +
    `not the parent. Do not include instructions, headers or rubric text as questions.\n` +
    `Return JSON: {"title":"","subject":"","totalMarks":0,"questions":[...]}`;

  return llm.callJson({
    stage: "question paper",
    system: EXAM_SYSTEM,
    user,
    schema: EXAM_SCHEMA,
    maxTokens: 3000,
  });
}

/**
 * Independent structural read of the same paper, used to cross-check the model.
 * Returns a map of question number -> printed marks.
 */
export function structuralMarks(lines) {
  const found = new Map();
  const { questions } = detectQuestionsStructural(lines);

  for (const q of questions) {
    const marks = Number.isFinite(q.maxMarks)
      ? q.maxMarks
      : extractMarks(q.questionText) ?? extractMarks(q.answerText);
    if (Number.isFinite(marks) && marks > 0) found.set(String(q.questionNumber), marks);
  }
  return found;
}

/* ------------------------------------------------------------ validation -- */

/**
 * Check the paper adds up, and say so plainly when it does not.
 *
 * Nothing is repaired here. A missing mark stays missing and a mismatched total
 * stays mismatched, because both mean the operator needs to look at the paper —
 * and a silently invented number would be marked against for the rest of the run.
 *
 * @returns {{questions: object[], totalMarks: number, declaredTotal: number|null,
 *            warnings: string[], blocking: boolean}}
 */
export function validateExam(raw, { structural = new Map() } = {}) {
  const warnings = [];

  const questions = (raw.questions || [])
    .filter((q) => normalizeText(q.text).length > 0 || String(q.number || "").trim())
    .map((q, i) => {
      const number = String(q.number || i + 1).trim();
      const printed = structural.get(number);
      let maxMarks = Number.isFinite(q.maxMarks) && q.maxMarks > 0 ? q.maxMarks : null;

      /* The structural pass read the marks straight off the page. Where the two
         disagree, the printed value wins and the disagreement is reported. */
      if (Number.isFinite(printed) && printed > 0) {
        if (maxMarks === null) maxMarks = printed;
        else if (maxMarks !== printed) {
          warnings.push(
            `Q${number}: the marks were read as ${maxMarks} but “${printed}” is printed on the paper. Using ${printed}.`
          );
          maxMarks = printed;
        }
      }

      return {
        id: `q${i + 1}`,
        number,
        text: String(q.text || "").trim(),
        maxMarks,
        type: classifyQuestion(q.text, maxMarks),
      };
    });

  if (questions.length === 0) {
    return {
      questions: [],
      totalMarks: 0,
      declaredTotal: null,
      warnings: ["No questions could be read from the question paper."],
      blocking: true,
    };
  }

  const duplicates = questions
    .map((q) => q.number)
    .filter((n, i, all) => all.indexOf(n) !== i);
  if (duplicates.length) {
    warnings.push(`Duplicate question numbers were read: ${[...new Set(duplicates)].join(", ")}.`);
  }

  const declaredTotal =
    Number.isFinite(raw.totalMarks) && raw.totalMarks > 0 ? raw.totalMarks : null;

  return {
    title: String(raw.title || "").trim(),
    subject: String(raw.subject || "").trim(),
    declaredTotal,
    /* Warnings about how the paper was *read* — these are facts about the
       extraction and never change afterwards. Anything about the marks
       themselves is derived below, because the operator can edit those. */
    baseWarnings: warnings,
    ...deriveMarks(questions, declaredTotal, warnings),
  };
}

/**
 * Everything that depends on the current mark values.
 *
 * Kept separate from validateExam so it can be recomputed whenever a mark is
 * edited. Previously the "these questions had no printed marks" warning was
 * generated once and then stood forever — it was still on screen, telling the
 * teacher to enter marks, after they had entered every one of them.
 *
 * @returns {{questions: object[], totalMarks: number, warnings: string[], blocking: boolean}}
 */
export function deriveMarks(questions, declaredTotal, baseWarnings = []) {
  const warnings = baseWarnings.slice();
  const missing = questions.filter((q) => q.maxMarks === null);

  if (missing.length) {
    warnings.push(
      `${missing.length} question${missing.length === 1 ? "" : "s"} (${missing
        .map((q) => "Q" + q.number)
        .join(", ")}) had no printed marks. Enter them before grading — they have not been guessed.`
    );
  }

  const sum = questions.reduce((s, q) => s + (q.maxMarks || 0), 0);

  if (declaredTotal !== null && missing.length === 0 && sum !== declaredTotal) {
    warnings.push(
      `The questions add up to ${sum} marks, but the paper states a total of ${declaredTotal}. ` +
        `Please check the question paper.`
    );
  }

  return {
    questions,
    totalMarks: sum,
    warnings,
    // Grading cannot start while a question has no ceiling to clamp against.
    blocking: missing.length > 0,
  };
}
