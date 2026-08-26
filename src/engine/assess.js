/**
 * Marking one question at a time.
 *
 * The model is asked to judge — what is correct, what is missing, what is wrong
 * — and it is asked to recommend a mark. It is not allowed to decide one. Every
 * number it returns is clamped to the ceiling the question paper set, and the
 * paper total is summed here, in code, from the clamped values. A model that
 * returns 8/5, or whose parts do not add up to its own total, cannot corrupt
 * the result.
 *
 * The same distrust applies to grounding. The model labels each point
 * REFERENCE_SUPPORTED, GENERAL_KNOWLEDGE or INSUFFICIENT_REFERENCE, and every
 * REFERENCE_SUPPORTED claim is checked against the evidence ids actually
 * retrieved. A claim that cites nothing real is downgraded rather than believed
 * — otherwise "the reference says so" becomes the easiest sentence to
 * hallucinate.
 */

import { criteriaFor } from "./exam.js";
import { WEAK_COVERAGE, chunkDocument, createRetriever } from "./reference.js";
import { detectQuestionsStructural } from "./segment.js";
import { applyChoice, describeChoice } from "./choice.js";
import { structureOcr } from "./ocr.js";
import { alignToQuestions, labelKey } from "./identity.js";

export const GROUNDING = {
  REFERENCE: "REFERENCE_SUPPORTED",
  GENERAL: "GENERAL_KNOWLEDGE",
  INSUFFICIENT: "INSUFFICIENT_REFERENCE",
};

/* --------------------------------------------------------- answer status -- */

/**
 * What the system honestly knows about an answer's existence.
 *
 * "unanswered" is a claim about the STUDENT; "not_detected" is a claim about
 * the PIPELINE. Blurring them is how a technical failure becomes a student's
 * zero, so the two are kept apart everywhere downstream of this function.
 */
export const ANSWER_STATUS = {
  DETECTED: "detected",
  UNCERTAIN: "uncertain",
  NOT_DETECTED: "not_detected",
  UNANSWERED: "unanswered",
  FAILED: "evaluation_failed",
};

/**
 * Classify one marked question.
 *
 * "unanswered" requires the strong case: nothing was matched AND there is no
 * meaningful unassigned writing that could be the missing answer. When such
 * writing exists the honest answer is "we could not link it", not "the
 * student did not write it".
 *
 * @param {object} q  one entry of PaperAssessment.questions
 * @param {{hasUnassignedWriting?: boolean, lowConfidence?: number}} opts
 */
export function answerStatus(q, { hasUnassignedWriting = false, lowConfidence = 60 } = {}) {
  if (q.failed) return ANSWER_STATUS.FAILED;
  if (q.skipped || !String(q.answerText || "").trim()) {
    return hasUnassignedWriting ? ANSWER_STATUS.NOT_DETECTED : ANSWER_STATUS.UNANSWERED;
  }
  if (Number.isFinite(q.confidence) && q.confidence < lowConfidence) return ANSWER_STATUS.UNCERTAIN;
  return ANSWER_STATUS.DETECTED;
}

/* A key short enough to read whole is sent whole; anything longer is cut to the
   part that answers *this* question. */
const KEY_WHOLE_CHARS = 1500;
const KEY_SECTION_CHARS = 1800;

/* Parsing the key is pure local work, but it is the same key for every question
   on the paper, so the last parse is kept rather than repeated a dozen times. */
let keyCache = { text: null, sections: null, retriever: null };

/**
 * Split any numbered document — an answer key, a set of model answers, a
 * solutions chapter — into its sections, keyed by question number.
 *
 * This is the same structural detector the student's answer sheet goes through,
 * so "Q3", "3.", "3(a)" and "iii)" mean the same thing on both sides. That is
 * what makes question 3 of the reference and question 3 of the paper pairable
 * at all.
 *
 * @param {string} text
 * @returns {Map<string, string>} canonical question number -> that section
 */
export function numberedSections(text, opts = {}) {
  const body = String(text || "").trim();
  if (!body) return [];

  const doc = structureOcr([body]);
  const { questions } = detectQuestionsStructural(doc.lines, opts);

  const out = [];
  for (const q of questions) {
    const label = String(q.questionNumber || "").trim();
    const section = `${q.questionText}\n${q.answerText}`.trim();
    if (!label || !section) continue;
    out.push({ label, text: section });
  }
  return out;
}

/**
 * The reference's model answers, filed against the questions of *this* paper.
 *
 * The pairing goes through the same alignment the student's answers go through,
 * so a key that writes "Ans 3a" still reaches the question the paper calls
 * "Q.3 (a)". Nothing is matched by string equality between two documents that
 * never agreed on a notation.
 *
 * @param {string} text  the reference material, as read
 * @param {{id: string, number: string, text?: string}[]} questions  the exam
 * @returns {Map<string, {label: string, text: string, confidence: number, method: string}>}
 *          keyed by question id
 */
export function pairReferenceAnswers(text, questions) {
  const known = new Set(
    (Array.isArray(questions) ? questions : []).map((q) => labelKey(q.number)).filter(Boolean)
  );
  const sections = numberedSections(text, { known });
  const out = new Map();
  if (sections.length === 0 || !Array.isArray(questions) || questions.length === 0) return out;

  /* Labels only. A model answer placed by content similarity alone would be a
     guess about which question it answers, and marking a student against the
     wrong model answer is worse than marking them without one. */
  const { placed } = alignToQuestions(sections, questions, { minContent: 2 });
  for (const [questionId, hit] of placed) {
    out.set(questionId, { ...sections[hit.index], confidence: hit.confidence, method: hit.method });
  }
  return out;
}

function parseKey(key) {
  if (keyCache.text === key) return keyCache;

  const sections = new Map();
  for (const section of numberedSections(key)) {
    const k = labelKey(section.label);
    /* A number that appears twice is ambiguous, and picking one of them is a
       guess. The first is kept and the rest ignored rather than concatenated
       into an answer nobody wrote. */
    if (k && !sections.has(k)) sections.set(k, section.text);
  }

  let retriever = null;
  try {
    retriever = createRetriever(chunkDocument(key, { source: "answer key" }));
  } catch {
    retriever = null; // an unchunkable key just loses the fallback, not the run
  }

  keyCache = { text: key, sections, retriever };
  return keyCache;
}

/**
 * The part of the teacher's answer key that belongs to one question.
 *
 * A long key used to be truncated at a fixed 4000 characters and handed to
 * every question alike, which meant the later half of a paper was marked
 * without its key at all — the answers were there, past the cut, and nobody was
 * told. The key is now addressed by question number, and falls back on
 * retrieval when the key carries no numbering to address.
 *
 * @param {string} answerKey  the teacher's key, as typed
 * @param {string} number     the question's number on the paper
 * @param {string} questionText  used only by the retrieval fallback
 * @returns {{text: string, scope: "whole"|"section"|"retrieved"|"none"}}
 */
export function answerKeySection(answerKey, number, questionText = "") {
  const key = String(answerKey || "").trim();
  if (!key) return { text: "", scope: "none" };
  if (key.length <= KEY_WHOLE_CHARS) return { text: key, scope: "whole" };

  const { sections, retriever } = parseKey(key);
  const want = labelKey(number);
  const hit = want ? sections.get(want) : null;
  if (hit && hit.length > 0) return { text: hit.slice(0, KEY_SECTION_CHARS), scope: "section" };

  if (retriever) {
    const { evidence } = retriever.search({ query: `${number} ${questionText}`.slice(0, 800), topK: 2 });
    const text = evidence.map((e) => e.text).join("\n\n").slice(0, KEY_SECTION_CHARS);
    if (text.trim()) return { text, scope: "retrieved" };
  }

  return { text: key.slice(0, KEY_SECTION_CHARS), scope: "whole" };
}

const ASSESS_SYSTEM =
  "You are an experienced school examiner. You mark one question at a time, strictly against the " +
  "marks available and the reference material supplied. You quote the student verbatim when you " +
  "annotate. You never award more than the maximum. You reply with JSON only.";

/** Built per question so the validator itself enforces the mark ceiling. */
function schemaFor(maxMarks) {
  return {
    type: "object",
    required: ["marksAwarded", "annotations"],
    props: {
      marksAwarded: { type: "number", min: 0, max: maxMarks, default: 0 },
      grounding: {
        type: "string",
        enum: [GROUNDING.REFERENCE, GROUNDING.GENERAL, GROUNDING.INSUFFICIENT],
        default: GROUNDING.GENERAL,
      },
      relevant: { type: "boolean", default: true },
      /* The mark's defence, written for the human examiner who reviews it.
         The assistant proposes; the examiner decides — and cannot decide on a
         bare number. */
      rationale: { type: "string", max: 500, default: "" },
      correctPoints: { type: "array", max: 12, of: { type: "string", max: 300 }, default: [] },
      missingPoints: { type: "array", max: 12, of: { type: "string", max: 300 }, default: [] },
      incorrectPoints: { type: "array", max: 12, of: { type: "string", max: 300 }, default: [] },
      feedback: { type: "string", max: 900, default: "" },
      annotations: {
        type: "array",
        max: 20,
        default: [],
        of: {
          type: "object",
          required: ["text", "type"],
          props: {
            text: { type: "string", max: 400 },
            type: {
              type: "string",
              enum: ["correct", "partial", "wrong", "missing"],
              default: "partial",
            },
            comment: { type: "string", max: 400, default: "" },
            /* No default: a model that says nothing about this annotation's
               marks has said nothing, and a defaulted 0 turned that silence
               into a "+0" badge printed beside correct work. */
            marks: { type: "number", min: -maxMarks, max: maxMarks, optional: true },
            /* No default, for the same reason marks has none: a defaulted 70 is
               indistinguishable from a reported 70, and every question on every
               paper came back at exactly 70% because the model rarely fills this
               in. A number nobody produced is not a measurement. */
            confidence: { type: "number", min: 0, max: 100, integer: true, optional: true },
            evidenceId: { type: "string", max: 120, optional: true },
          },
        },
      },
    },
  };
}

/**
 * Mark a single question.
 *
 * @param {object} args
 * @param {object} args.answer     one entry from matchAnswers()
 * @param {import("./types.js").RetrievedEvidence[]} args.evidence
 * @param {number} args.coverage   0..1, how well the reference covers the question
 * @param {{callJson: Function}} args.llm
 */
export async function assessQuestion({
  answer,
  evidence = [],
  coverage = 0,
  llm,
  answerKey = "",
  pairedReference = null,
}) {
  const maxMarks = Number(answer.maxMarks) || 0;

  /* A question with no answer needs no model call. Skipping it saves a request
     against the rate limit and removes any chance of marks appearing from
     nowhere. The wording deliberately avoids "not attempted": a match failure
     is a fact about the pipeline, not about the student. */
  if (answer.skipped || !answer.answerText.trim()) {
    return {
      ...answer,
      marksAwarded: 0,
      maxMarks,
      grounding: GROUNDING.INSUFFICIENT,
      rationale:
        "No answer could be confidently linked to this question. This is a detection result, " +
        "not proof the question was left unanswered — check the answer sheet before treating it as a zero.",
      correctPoints: [],
      missingPoints: ["No answer was confidently detected for this question."],
      incorrectPoints: [],
      annotations: [],
      feedback:
        "No answer was confidently detected for this question. If the student did answer it, " +
        "the OCR may have missed the writing — check the raw text.",
      confidence: answer.confidence,
      evidence,
    };
  }

  const criteria = criteriaFor(answer.type);

  /* The teacher's own key, when there is one, outranks everything else.
     Measured on a real CBSE matrices paper: with no key the examiner has to
     solve the mathematics itself, and it marked objectively correct MCQ answers
     wrong — scoring "k = 3, p = n" as incorrect when that was the right option.
     Papers exported from question banks routinely carry empty "Answer Key:"
     fields, so the key has to be something the teacher can supply. */
  const key = answerKeySection(answerKey, answer.number, answer.questionText);
  const keyBlock = key.text
    ? "AUTHORITATIVE ANSWER KEY / MARKING SCHEME — this outranks your own working. " +
      "If it settles this question, mark against it and do not re-derive the answer yourself" +
      (key.scope === "whole"
        ? ":\n"
        : `. This is the part of the key that addresses question ${answer.number}:\n`) +
      key.text +
      "\n\n"
    : "";
  const refBlock = evidence.length
    ? evidence
        .map((e) => {
          const src = e.metadata && e.metadata.source ? ` — ${e.metadata.source}` : "";
          return `[${e.id}]${src}\n${e.text}`;
        })
        .join("\n\n")
    : "(no reference material matched this question)";

  /* The strongest case there is: the reference material carries a model answer
     filed under this exact question number, so the two can be compared point
     by point instead of by topic. It is still a comparison of meaning — a
     student who says the same thing in their own words has said it. */
  const pairBlock = pairedReference
    ? `The reference material includes the model answer filed under question ${answer.number} (cited above as [${pairedReference.id}]). Mark this student against it point by point: for each point the model answer makes, decide whether the student made it. Wording does not have to match — the same idea in the student's own words earns the mark, and a point the model answer does not make cannot cost one.\n\n`
    : "";

  const user =
    keyBlock +
    pairBlock +
    `QUESTION ${answer.number} (${maxMarks} marks, type: ${answer.type})\n${answer.questionText}\n\n` +
    `MARK THIS ANSWER ON: ${criteria.join(", ")}\n\n` +
    `REFERENCE MATERIAL — prefer this over your own knowledge:\n${refBlock}\n\n` +
    `STUDENT'S ANSWER (from OCR of handwriting, so spelling may be imperfect):\n${answer.answerText}\n\n` +
    `Mark it as a teacher would. Award partial credit where deserved. Judge the answer on its ` +
    `substance, not its handwriting or spelling.\n\n` +
    /* Students label their answers with the question number, and multiple-choice
       options are numbered too. Without this the examiner reads a leading "2)"
       as "chose option 2" and marks a correct answer wrong — observed on a real
       paper where the student's "2) k = 3, p = n" was in fact option 1. */
    `The student's answer may begin with the question number they wrote (e.g. "${answer.number})"). ` +
    `That is a label, NOT their choice of option. Judge only the content after it. For a ` +
    `multiple-choice question, compare the student's written content against the options and ` +
    `decide which option that content matches.\n\n` +
    `Rules you must follow:\n` +
    `- "marksAwarded" must be between 0 and ${maxMarks}.\n` +
    `- Every "text" in "annotations" MUST be copied character-for-character from the student's ` +
    `answer above, so it can be highlighted on their page. Keep each to a phrase or one sentence.\n` +
    `- Use type "missing" for expected content that is absent; put the missing idea in "text" ` +
    `(it will be shown as a margin note, not a highlight).\n` +
    `- "grounding": "${GROUNDING.REFERENCE}" only if the reference material above actually ` +
    `settles this answer — and then set "evidenceId" on the annotations it supports, using the ` +
    `bracketed id exactly. Use "${GROUNDING.GENERAL}" if you judged from your own knowledge, or ` +
    `"${GROUNDING.INSUFFICIENT}" if the reference material was not enough and you are unsure.\n` +
    `- "marks" on EVERY annotation: how many marks that specific point earned, or cost. The positive ones must add up to exactly "marksAwarded" — this is the breakdown the teacher and the student see beside the highlight, and an annotation with no marks shows them nothing.\n` +
    `- "confidence" on EVERY annotation: 0-100, your own honest certainty that this mark is right. Do not leave it out and do not put the same number on every one. Go low when the handwriting was ambiguous, the phrasing unclear, or the marking a judgement call; go high when the answer plainly matches the reference.\n` +
    `- "rationale": 2-3 sentences for the teacher who will review your marking — say exactly why ` +
    `this mark and not more or less: what earned marks, what cost marks. Name the content, do not ` +
    `write "the answer was good".\n` +
    `- "feedback": tell the student what was right, what was missing and how to improve, ` +
    `referring to the actual content — not generic advice.\n\n` +
    `Return ONLY JSON:\n` +
    `{"marksAwarded":0,"grounding":"...","relevant":true,"rationale":"...","correctPoints":[],"missingPoints":[],` +
    `"incorrectPoints":[],"feedback":"","annotations":[{"text":"","type":"correct","comment":"",` +
    `"marks":0,"confidence":0,"evidenceId":""}]}`;

  const raw = await llm.callJson({
    stage: `Q${answer.number}`,
    system: ASSESS_SYSTEM,
    user,
    schema: schemaFor(maxMarks),
    maxTokens: 2200,
  });

  /* ---- grounding, verified rather than accepted ---- */
  const validIds = new Set(evidence.map((e) => e.id));
  const annotations = (raw.annotations || []).map((a) => ({
    ...a,
    questionId: answer.questionId,
    questionNumber: answer.number,
    // A citation that does not name real retrieved evidence is not a citation.
    evidenceId: a.evidenceId && validIds.has(a.evidenceId) ? a.evidenceId : undefined,
  }));

  let grounding = raw.grounding;
  const citesRealEvidence = annotations.some((a) => a.evidenceId);

  if (grounding === GROUNDING.REFERENCE && !citesRealEvidence) {
    // It said the reference settled it but pointed at nothing that exists.
    grounding = evidence.length ? GROUNDING.GENERAL : GROUNDING.INSUFFICIENT;
  }
  if (evidence.length === 0) grounding = GROUNDING.INSUFFICIENT;
  else if (coverage < WEAK_COVERAGE && grounding === GROUNDING.REFERENCE && !citesRealEvidence) {
    grounding = GROUNDING.INSUFFICIENT;
  }

  /* ---- marks: clamped again here, independently of the schema ---- */
  const marksAwarded = clamp(raw.marksAwarded, 0, maxMarks);

  /* ---- per-annotation marks: kept only when they can be true ----
     They are the line-by-line account of the mark awarded. When they add up to
     more than was actually given, they contradict it, and a "+2" beside a
     sentence that earned nothing is a claim the student will read and believe.
     In that case the numbers go and the highlight keeps its verdict alone. */
  const claimed = annotations.reduce(
    (n, a) => n + (Number.isFinite(a.marks) && a.marks > 0 ? a.marks : 0),
    0
  );
  const reportsMarks = annotations.some((a) => Number.isFinite(a.marks));

  if (claimed > marksAwarded + 1e-9) {
    for (const a of annotations) delete a.marks;
  } else if (!reportsMarks && marksAwarded > 0) {
    /* One highlight and one mark can only be allocated one way, so allocating it
       states nothing the result does not already say. Two highlights and one
       mark can be allocated several ways, and choosing one would be inventing a
       breakdown — those stay unmarked until the marker gives one. */
    const earners = annotations.filter((a) => a.type === "correct" || a.type === "partial");
    if (earners.length === 1) earners[0].marks = marksAwarded;
  }

  /* ---- confidence: the weakest link in the chain wins ----
     Only annotations that actually reported a confidence count towards it. When
     none did, there is no model confidence to average, and the honest number is
     the one the detection stage measured — not a constant dressed up as one. */
  const reported = annotations
    .map((a) => Number(a.confidence))
    .filter((n) => Number.isFinite(n));
  const detection = Number.isFinite(answer.confidence) ? answer.confidence : 100;
  const modelConfidence = reported.length
    ? Math.round(reported.reduce((s, n) => s + n, 0) / reported.length)
    : null;
  let confidence = modelConfidence === null ? detection : Math.min(modelConfidence, detection);
  if (grounding === GROUNDING.INSUFFICIENT) confidence = Math.min(confidence, 65);
  if (raw.relevant === false) confidence = Math.min(confidence, 55);

  return {
    ...answer,
    maxMarks,
    marksAwarded,
    grounding,
    relevant: raw.relevant !== false,
    rationale: raw.rationale || "",
    correctPoints: raw.correctPoints || [],
    missingPoints: raw.missingPoints || [],
    incorrectPoints: raw.incorrectPoints || [],
    feedback: raw.feedback || "",
    annotations,
    confidence,
    evidence,
    coverage,
  };
}

/**
 * One closing remark in the teacher's voice.
 *
 * A single call for the whole paper, kept small. The per-question feedback is
 * where the grounded detail lives; this exists because the report card reads it
 * aloud and types it out, and a remark assembled from string fragments sounds
 * like a form letter.
 */
export async function summarisePaper({ paper, llm }) {
  const lines = paper.questions
    .map(
      (q) =>
        `Q${q.number} (${q.marksAwarded}/${q.maxMarks}): ` +
        (q.skipped
          ? "not attempted"
          : `correct: ${(q.correctPoints || []).join("; ") || "—"}. missing: ${
              (q.missingPoints || []).join("; ") || "—"
            }`)
    )
    .join("\n")
    .slice(0, 2500);

  try {
    const raw = await llm.callJson({
      stage: "summary",
      system:
        "You are an experienced teacher writing the closing remark on a marked paper. Encouraging " +
        "but honest, specific to what this student actually did. You reply with JSON only.",
      user:
        `Marked paper — ${paper.totalMarks} out of ${paper.maximumMarks} (${Math.round(
          paper.percentage
        )}%), grade ${paper.grade}.\n\n${lines}\n\n` +
        `Write the teacher's final remark: 3-4 sentences, naming real strengths and the specific ` +
        `topics to work on. Do not invent anything not shown above.\n` +
        `Return JSON: {"remark":"..."}`,
      schema: {
        type: "object",
        required: ["remark"],
        props: { remark: { type: "string", max: 900 } },
      },
      maxTokens: 900,
    });
    return raw.remark;
  } catch {
    /* The remark is a flourish, not a result. If it fails, the marks and the
       per-question feedback are all still correct. */
    return `Scored ${paper.totalMarks} out of ${paper.maximumMarks} (${Math.round(
      paper.percentage
    )}%), grade ${paper.grade}. See the per-question feedback for detail.`;
  }
}

/**
 * Present a marked paper in the shape the existing UI already reads.
 *
 * The report card, the annotated views, the score arc, the history log and the
 * print stylesheet all consume `{annotations, keyPoints, totalMarksAwarded,
 * totalMarks, grade, overallRemark, ...}`. Rather than rewrite those, the new
 * per-question assessment is projected onto that shape — each question becomes
 * one key point, and every question's annotations flatten into the single list
 * the highlighter expects. Extra fields ride along for the views that want them
 * and are ignored by the ones that do not.
 */
export function toEvaluation(paper, remark) {
  return {
    annotations: paper.questions.flatMap((q) => q.annotations || []),
    keyPoints: paper.questions.map((q) => ({
      point: `Q${q.number}. ${q.questionText}`.trim(),
      covered: q.marksAwarded > 0,
      quality:
        q.maxMarks > 0 && q.marksAwarded >= q.maxMarks
          ? "well"
          : q.marksAwarded > 0
          ? "partially"
          : "not",
      marksAwarded: q.marksAwarded,
      marksTotal: q.maxMarks,
      teacherNote: q.feedback,
      // Carried through for the per-question detail the new pipeline provides.
      // questionId and the page range are the identity the "view on page"
      // navigation keys off — one id governs text, marks, annotations and page.
      questionId: q.questionId,
      questionNumber: q.number,
      pageStart: q.pageStart,
      pageEnd: q.pageEnd,
      rationale: q.rationale || "",
      grounding: q.grounding,
      confidence: q.confidence,
      /* The status vocabulary (detected / not detected / unanswered) is decided
         by answerStatus() from these three fields. Dropping answerText here
         made every question look unanswered in the views. */
      answerText: q.answerText || "",
      skipped: !!q.skipped,
      failed: !!q.failed,
      /* False when the paper's choice means this attempt does not count towards
         the total. It is still marked, still shown, and still not a zero. */
      counted: q.counted !== false,
      correctPoints: q.correctPoints || [],
      missingPoints: q.missingPoints || [],
      incorrectPoints: q.incorrectPoints || [],
      evidence: (q.evidence || []).map((e) => ({ id: e.id, source: e.metadata && e.metadata.source })),
    })),
    totalMarksAwarded: paper.totalMarks,
    totalMarks: paper.maximumMarks,
    printedMarks: paper.printedMarks,
    choice: paper.choice || [],
    dropped: paper.dropped || [],
    grade: paper.grade,
    overallRemark: remark,
    thingsWellDone: collectPoints(paper.questions, "correctPoints"),
    improvementAreas: collectPoints(paper.questions, "missingPoints"),
    // The new pipeline's own record, for callers that want the full detail.
    paper,
  };
}

/** Distinct, non-empty, capped — for the report card's two lists. */
export function collectPoints(questions, key, limit = 8) {
  const seen = new Set();
  const out = [];
  for (const q of questions) {
    for (const p of q[key] || []) {
      const clean = String(p || "").trim();
      const dedupe = clean.toLowerCase();
      if (!clean || seen.has(dedupe)) continue;
      seen.add(dedupe);
      out.push(clean);
      if (out.length >= limit) return out;
    }
  }
  return out;
}

const clamp = (n, lo, hi) => {
  const v = Number(n);
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
};

export function gradeFor(pct) {
  if (pct >= 90) return "A+";
  if (pct >= 75) return "A";
  if (pct >= 60) return "B";
  if (pct >= 45) return "C";
  if (pct >= 33) return "D";
  return "F";
}

/**
 * Mark the whole paper, question by question.
 *
 * Sequential on purpose. The free Groq tier meters ~8000 tokens a minute across
 * all models and charges the completion budget up front, so firing every
 * question at once just trips the limit and burns the retry budget. One at a
 * time with the client's own back-off is slower but finishes.
 *
 * @param {object} args
 * @param {object} args.exam
 * @param {object[]} args.answers    from matchAnswers()
 * @param {object} [args.retriever]  from createRetriever()
 * @param {{callJson: Function}} args.llm
 * @param {(done: number, total: number, label: string, question?: object) => void} [args.onProgress]
 *        The finished question is passed on so a caller can report what was
 *        actually decided, not just how far along it is.
 * @returns {Promise<import("./types.js").PaperAssessment>}
 */
/**
 * The reference's model answer for one question, as a piece of evidence.
 *
 * Keyed by question id, not by number: the pairing was already decided by
 * alignment, so nothing here re-compares two label strings that were never
 * written in the same notation.
 *
 * @param {Map<string,{text: string}>|null} index  from pairReferenceAnswers()
 * @param {object} question  the answer being marked (needs questionId, number)
 * @returns {import("./types.js").RetrievedEvidence|null}
 */
export function pairFor(index, question) {
  if (!index || typeof index.get !== "function" || !question) return null;
  const hit = index.get(question.questionId);
  const text = hit && hit.text;
  if (!text || !String(text).trim()) return null;

  return {
    id: `refq-${labelKey(question.number) || question.questionId}`,
    text: String(text).slice(0, 2000),
    score: 1,
    lexicalScore: 1,
    metadata: { source: `reference answer for question ${question.number}` },
  };
}

export async function assessPaper({
  exam,
  answers,
  retriever,
  llm,
  onProgress,
  warnings = [],
  concurrency = 1,
  answerKey = "",
  /* Model answers from the reference material, already aligned to this paper's
     questions — see pairReferenceAnswers(). When the paper's question 5 and the
     reference's question 5 are both known, they are marked against each other
     directly instead of through a topic search that may land anywhere. */
  referenceAnswers = null,
}) {
  const questions = new Array(answers.length);
  let done = 0;
  let cursor = 0;

  const markOne = async (i) => {
    const answer = answers[i];

    let evidence = [];
    let coverage = 0;
    if (retriever && retriever.size > 0) {
      // Retrieve on the question plus what the student actually wrote: the
      // answer often names the concept the question only implies.
      const found = retriever.search({
        query: `${answer.questionText} ${answer.answerText}`.slice(0, 1200),
        topK: 3,
      });
      evidence = found.evidence;
      coverage = found.coverage;
    }

    /* An exact numbered pairing outranks anything retrieval found. It goes in
       as real evidence with a real id, so the grounding check still has to be
       satisfied by a citation — being the right section does not exempt it. */
    const paired = pairFor(referenceAnswers, answer);
    if (paired) {
      evidence = [paired].concat(evidence.filter((e) => e.id !== paired.id)).slice(0, 4);
      coverage = 1;
    }

    try {
      questions[i] = await assessQuestion({
        answer,
        evidence,
        coverage,
        llm,
        answerKey,
        pairedReference: paired,
      });
    } catch (e) {
      /* One failed question must not lose the other nine. It is recorded as
         unmarked, at zero, and called out — never silently given full marks. */
      warnings.push(
        `Q${answer.number} could not be marked (${e.message}). It is left unmarked — enter a mark by hand.`
      );
      questions[i] = {
        ...answer,
        maxMarks: Number(answer.maxMarks) || 0,
        marksAwarded: 0,
        grounding: GROUNDING.INSUFFICIENT,
        rationale: `Automatic marking failed (${e.message}). This zero is a placeholder, not a judgement — mark it by hand.`,
        correctPoints: [],
        missingPoints: [],
        incorrectPoints: [],
        annotations: [],
        feedback: `This question could not be marked automatically: ${e.message}`,
        confidence: 0,
        evidence,
        failed: true,
      };
    }

    done++;
    if (onProgress) onProgress(done - 1, answers.length, `Q${answer.number}`, questions[i]);
  };

  /* One worker per available API key. The proxy schedules each request onto a
     key with budget to spare, so N keys really do give N questions in flight —
     but only N: going wider just queues behind the same token ceiling and turns
     a clean run into a storm of 429s. Results are written by index, so the
     report keeps the question paper's order regardless of finishing order. */
  const workers = Math.max(1, Math.min(concurrency, answers.length));
  await Promise.all(
    Array.from({ length: workers }, async () => {
      for (;;) {
        const i = cursor++;
        if (i >= answers.length) return;
        await markOne(i);
      }
    })
  );

  if (onProgress) onProgress(answers.length, answers.length, "");

  /* ---- the total is computed here, never read from a model ---- */
  /* A paper that said "answer any 3 of the following 5" is scored out of three
     questions. The best attempts fill the slots; the extras are marked and shown
     but do not count, and are not held against the student either. */
  for (const q of questions) q.marksAwarded = clamp(q.marksAwarded, 0, q.maxMarks);
  const choice = Array.isArray(exam.choice) ? exam.choice : [];
  const chosen = applyChoice(questions, choice);
  for (const q of questions) q.counted = chosen.counted.has(q.questionId);

  const totalMarks = chosen.totalMarks;
  const maximumMarks = chosen.maximumMarks;
  const percentage = maximumMarks > 0 ? (totalMarks / maximumMarks) * 100 : 0;

  if (chosen.dropped.length) {
    warnings.push(
      `${describeChoice(choice)}: ` +
        chosen.dropped.map((d) => `Q${d.number} does not count (${d.reason})`).join("; ") +
        `. The total is out of ${maximumMarks}.`
    );
  }

  const marked = questions.filter((q) => !q.skipped && !q.failed && q.counted !== false);
  const overallConfidence = marked.length
    ? Math.round(marked.reduce((s, q) => s + (q.confidence || 0), 0) / marked.length)
    : 0;

  const usedReference = questions.filter((q) => q.grounding === GROUNDING.REFERENCE).length;
  if (retriever && retriever.size > 0 && usedReference === 0) {
    warnings.push(
      "No question was settled by the reference material — the marking fell back to the model's " +
        "own knowledge. Check that the reference PDFs cover this paper."
    );
  }

  return {
    paperId: `paper-${Date.now()}`,
    subject: exam.subject || "",
    questions,
    totalMarks,
    maximumMarks,
    /* What the paper printed, kept beside what it is worth, so a report can say
       "24 / 30, from a paper printing 50" rather than looking like it lost 20. */
    printedMarks: questions.reduce((s, q) => s + (Number(q.maxMarks) || 0), 0),
    choice,
    dropped: chosen.dropped,
    percentage,
    grade: gradeFor(percentage),
    passed: maximumMarks > 0 && percentage >= 33,
    overallConfidence,
    warnings,
  };
}
