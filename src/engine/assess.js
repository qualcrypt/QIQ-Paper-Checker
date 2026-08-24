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
import { WEAK_COVERAGE } from "./reference.js";

export const GROUNDING = {
  REFERENCE: "REFERENCE_SUPPORTED",
  GENERAL: "GENERAL_KNOWLEDGE",
  INSUFFICIENT: "INSUFFICIENT_REFERENCE",
};

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
            marks: { type: "number", min: -maxMarks, max: maxMarks, default: 0 },
            confidence: { type: "number", min: 0, max: 100, integer: true, default: 70 },
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
export async function assessQuestion({ answer, evidence = [], coverage = 0, llm, answerKey = "" }) {
  const maxMarks = Number(answer.maxMarks) || 0;

  /* A question with no answer needs no model call. Skipping it saves a request
     against the rate limit and removes any chance of marks appearing from
     nowhere. */
  if (answer.skipped || !answer.answerText.trim()) {
    return {
      ...answer,
      marksAwarded: 0,
      maxMarks,
      grounding: GROUNDING.INSUFFICIENT,
      correctPoints: [],
      missingPoints: ["The question was not attempted."],
      incorrectPoints: [],
      annotations: [],
      feedback: "No answer was found for this question on the answer sheet.",
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
  const keyBlock = String(answerKey || "").trim()
    ? "AUTHORITATIVE ANSWER KEY / MARKING SCHEME — this outranks your own working. " +
      "If it settles this question, mark against it and do not re-derive the answer yourself:\n" +
      String(answerKey).trim().slice(0, 4000) +
      "\n\n"
    : "";
  const refBlock = evidence.length
    ? evidence.map((e, i) => `[${e.id}]\n${e.text}`).join("\n\n")
    : "(no reference material matched this question)";

  const user =
    keyBlock +
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
    `- "confidence" per annotation: 0-100, honest. Go low when the handwriting was ambiguous or ` +
    `the marking was a judgement call.\n` +
    `- "feedback": tell the student what was right, what was missing and how to improve, ` +
    `referring to the actual content — not generic advice.\n\n` +
    `Return ONLY JSON:\n` +
    `{"marksAwarded":0,"grounding":"...","relevant":true,"correctPoints":[],"missingPoints":[],` +
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

  /* ---- confidence: the weakest link in the chain wins ---- */
  const modelConfidence = annotations.length
    ? Math.round(annotations.reduce((s, a) => s + (Number(a.confidence) || 0), 0) / annotations.length)
    : 60;
  let confidence = Math.min(modelConfidence, answer.confidence || 100);
  if (grounding === GROUNDING.INSUFFICIENT) confidence = Math.min(confidence, 65);
  if (raw.relevant === false) confidence = Math.min(confidence, 55);

  return {
    ...answer,
    maxMarks,
    marksAwarded,
    grounding,
    relevant: raw.relevant !== false,
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
      questionNumber: q.number,
      grounding: q.grounding,
      confidence: q.confidence,
      skipped: !!q.skipped,
      missingPoints: q.missingPoints || [],
      incorrectPoints: q.incorrectPoints || [],
      evidence: (q.evidence || []).map((e) => ({ id: e.id, source: e.metadata && e.metadata.source })),
    })),
    totalMarksAwarded: paper.totalMarks,
    totalMarks: paper.maximumMarks,
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
 * @param {(done: number, total: number, label: string) => void} [args.onProgress]
 * @returns {Promise<import("./types.js").PaperAssessment>}
 */
export async function assessPaper({
  exam,
  answers,
  retriever,
  llm,
  onProgress,
  warnings = [],
  concurrency = 1,
  answerKey = "",
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

    try {
      questions[i] = await assessQuestion({ answer, evidence, coverage, llm, answerKey });
    } catch (e) {
      /* One failed question must not lose the other nine. It is recorded as
         unmarked, at zero, and called out — never silently given full marks. */
      warnings.push(`Q${answer.number} could not be marked (${e.message}). It is scored 0.`);
      questions[i] = {
        ...answer,
        maxMarks: Number(answer.maxMarks) || 0,
        marksAwarded: 0,
        grounding: GROUNDING.INSUFFICIENT,
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
    if (onProgress) onProgress(done - 1, answers.length, `Q${answer.number}`);
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
  const totalMarks = questions.reduce((s, q) => s + clamp(q.marksAwarded, 0, q.maxMarks), 0);
  const maximumMarks = questions.reduce((s, q) => s + (Number(q.maxMarks) || 0), 0);
  const percentage = maximumMarks > 0 ? (totalMarks / maximumMarks) * 100 : 0;

  const marked = questions.filter((q) => !q.skipped && !q.failed);
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
    percentage,
    grade: gradeFor(percentage),
    passed: maximumMarks > 0 && percentage >= 33,
    overallConfidence,
    warnings,
  };
}
