/**
 * The question-paper-first pipeline: exam validation, reference retrieval,
 * answer matching and mark control. A stub LLM stands in for Groq so the
 * deterministic guards — the parts that must hold no matter what the model
 * says — are what actually gets tested.
 */
import { validateExam, classifyQuestion, criteriaFor, deriveMarks } from "../src/engine/exam.js";
import { chunkDocument, createRetriever, WEAK_COVERAGE } from "../src/engine/reference.js";
import { matchAnswers, canonicalNumber } from "../src/engine/match.js";
import { assessPaper, assessQuestion, gradeFor, GROUNDING } from "../src/engine/assess.js";
import { structureOcr } from "../src/engine/ocr.js";

let failures = 0;
const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) failures++;
};
const section = (s) => console.log("\n" + s);

/* ------------------------------------------------------ exam validation -- */
section("exam structure and mark validation (Phase 2/3)");

t("Q1 style", canonicalNumber("Q1") === "1");
t("Q.1 style", canonicalNumber("Q.1") === "1");
t("1. style", canonicalNumber("1.") === "1");
t("1) style", canonicalNumber("1)") === "1");
t("1(a) style", canonicalNumber("1(a)") === "1(a)");
t("Q2(A) normalises case", canonicalNumber("Q2(A)") === "2(a)");
t("2a style", canonicalNumber("2a") === "2(a)");
t("bare (a)", canonicalNumber("(a)") === "(a)");
t("roman 3(ii)", canonicalNumber("3(ii)") === "3(ii)");

const good = validateExam({
  title: "Class 12 Biology", subject: "Biology", totalMarks: 12,
  questions: [
    { number: "1", text: "Define photosynthesis.", maxMarks: 2 },
    { number: "2", text: "Explain the process of photosynthesis.", maxMarks: 5 },
    { number: "3", text: "Compare mitosis and meiosis.", maxMarks: 5 },
  ],
});
t("well-formed paper produces no warnings", good.warnings.length === 0);
t("total is summed from the questions", good.totalMarks === 12);
t("not blocking", good.blocking === false);

const mismatch = validateExam({
  totalMarks: 20,
  questions: [
    { number: "1", text: "Define photosynthesis.", maxMarks: 2 },
    { number: "2", text: "Explain photosynthesis.", maxMarks: 5 },
    { number: "3", text: "Compare mitosis and meiosis.", maxMarks: 8 },
  ],
});
t("sum 15 vs declared 20 is reported", mismatch.warnings.some((w) => w.includes("15") && w.includes("20")));
t("marks are NOT silently adjusted to fit", mismatch.totalMarks === 15);

const missing = validateExam({
  totalMarks: 10,
  questions: [
    { number: "1", text: "Define photosynthesis.", maxMarks: 2 },
    { number: "2", text: "Explain photosynthesis." },
  ],
});
t("missing marks are reported", missing.warnings.some((w) => w.includes("Q2")));
t("missing marks are NOT invented", missing.questions[1].maxMarks === null);
t("missing marks block grading", missing.blocking === true);

const dup = validateExam({ questions: [
  { number: "1", text: "A question here.", maxMarks: 2 },
  { number: "1", text: "Another question.", maxMarks: 3 },
]});
t("duplicate numbers reported", dup.warnings.some((w) => w.toLowerCase().includes("duplicate")));

t("printed marks beat model-read marks", (() => {
  const v = validateExam(
    { questions: [{ number: "1", text: "Explain photosynthesis.", maxMarks: 9 }] },
    { structural: new Map([["1", 5]]) }
  );
  return v.questions[0].maxMarks === 5 && v.warnings.some((w) => w.includes("printed"));
})());

section("mark warnings clear once the marks are entered");
{
  /* A paper that prints no marks at all. The notice telling the teacher to enter
     them must not still be on screen after they have entered every one. */
  let e = validateExam({
    title: "Important GK Questions and Answers",
    questions: Array.from({ length: 9 }, (_, i) => ({ number: String(i + 1), text: "GK question " + (i + 1) })),
  });
  t("warns while marks are missing", e.warnings.some((w) => w.includes("no printed marks")) && e.blocking);

  for (const q of e.questions) {
    const qs = e.questions.map((x) => (x.id === q.id ? { ...x, maxMarks: 1 } : x));
    e = { ...e, ...deriveMarks(qs, e.declaredTotal, e.baseWarnings) };
  }
  t("warning clears once every mark is entered", !e.warnings.some((w) => w.includes("no printed marks")));
  t("total and blocking follow the edits", e.totalMarks === 9 && e.blocking === false);

  let c = validateExam(
    { questions: [{ number: "1", text: "a question", maxMarks: 9 }] },
    { structural: new Map([["1", 5]]) }
  );
  c = { ...c, ...deriveMarks(c.questions, c.declaredTotal, c.baseWarnings) };
  t("read-time warnings survive mark edits", c.warnings.some((w) => w.includes("printed")));
}

section("question typing and criteria (Phase 9)");
t("define -> definition", classifyQuestion("Define photosynthesis.", 2) === "definition");
t("compare -> compare", classifyQuestion("Compare mitosis and meiosis.", 5) === "compare");
t("explain -> explain", classifyQuestion("Explain the process.", 5) === "explain");
t("difference between -> compare", classifyQuestion("State the difference between X and Y.", 5) === "compare");
t("unknown verb falls back on marks", classifyQuestion("Photosynthesis.", 12) === "essay");
t("compare criteria mention differences", criteriaFor("compare").join().includes("differences"));
t("definition criteria stay short", criteriaFor("definition").length === 2);

/* ---------------------------------------------------------- retrieval --- */
section("reference retrieval (Phase 4)");

const BOOK = `Photosynthesis is the process by which green plants and certain other organisms
transform light energy into chemical energy. During photosynthesis in green plants, light energy is
captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic
compounds. Chlorophyll is the green pigment found in the chloroplasts of plant leaves. It absorbs
light most strongly in the blue and red regions of the spectrum. The light-dependent reactions take
place in the thylakoid membranes of the chloroplast. The light-independent reactions, also called the
Calvin cycle, take place in the stroma of the chloroplast. Mitosis is a type of cell division that
results in two daughter cells each having the same number of chromosomes as the parent nucleus.
Meiosis is a type of cell division that reduces the chromosome number by half, producing four
haploid gametes. Mitosis produces two diploid cells whereas meiosis produces four haploid cells.`;

const chunks = chunkDocument(BOOK, { source: "biology-textbook.pdf" });
t("document splits into chunks", chunks.length >= 2);
t("chunks carry a source", chunks.every((c) => c.metadata.source === "biology-textbook.pdf"));
t("chunks carry unique ids", new Set(chunks.map((c) => c.id)).size === chunks.length);

const retriever = createRetriever(chunks);
const photo = retriever.search({ query: "Explain the role of chlorophyll in photosynthesis", topK: 3 });
t("retrieves something for a covered topic", photo.evidence.length > 0);
t("top chunk actually mentions chlorophyll", /chlorophyll/i.test(photo.evidence[0].text));
t("coverage is high for a covered topic", photo.coverage >= WEAK_COVERAGE);

const offTopic = retriever.search({ query: "Describe the Treaty of Versailles and its economic consequences", topK: 3 });
t("coverage is low for an uncovered topic", offTopic.coverage < WEAK_COVERAGE);
t("  (covered=" + photo.coverage.toFixed(2) + " vs uncovered=" + offTopic.coverage.toFixed(2) + ")", true);

const mit = retriever.search({ query: "Compare mitosis and meiosis", topK: 2 });
t("retrieval discriminates between topics", /mitosis|meiosis/i.test(mit.evidence[0].text));

/* ----------------------------------------------------- answer matching -- */
section("answer matching (Phase 7)");

const exam = validateExam({
  totalMarks: 12,
  questions: [
    { number: "1", text: "Define photosynthesis.", maxMarks: 2 },
    { number: "2", text: "Explain the role of chlorophyll.", maxMarks: 5 },
    { number: "3", text: "Compare mitosis and meiosis.", maxMarks: 5 },
  ],
});

// Student answered 1 and 3, labelled, and skipped 2 entirely.
const answerDoc = structureOcr([
  [
    "1. Photosynthesis is the process by which green plants make their own food",
    "using sunlight, water and carbon dioxide.",
    "3. Mitosis produces two diploid cells while meiosis produces four haploid",
    "cells. Mitosis is for growth and meiosis is for gamete formation.",
  ].join("\n"),
]);

const matched = await matchAnswers(answerDoc, exam, { llm: null });
const byNum = Object.fromEntries(matched.answers.map((a) => [a.number, a]));
t("every exam question appears in the result", matched.answers.length === 3);
t("Q1 matched by its written label", byNum["1"].method === "label" && !byNum["1"].skipped);
t("Q1 answer text is the student's own words", /green plants make their own food/.test(byNum["1"].answerText));
t("Q3 matched by its written label", byNum["3"].method === "label" && !byNum["3"].skipped);
t("Q3 got the mitosis text, not Q1's", /diploid/.test(byNum["3"].answerText) && !/sunlight/.test(byNum["3"].answerText));
t("skipped Q2 is reported skipped, not guessed", byNum["2"].skipped === true && byNum["2"].answerText === "");
t("skipped Q2 has zero confidence", byNum["2"].confidence === 0);
t("labelled matches carry high confidence", byNum["1"].confidence >= 90);

/* ------------------------------------------------- marks are controlled -- */
section("mark control (Phase 10)");

// A deliberately badly-behaved model: over-awards, cites fake evidence,
// and claims the reference supported it.
const rogueLlm = {
  callJson: async () => ({
    marksAwarded: 999,
    grounding: GROUNDING.REFERENCE,
    relevant: true,
    correctPoints: ["said something"],
    missingPoints: [],
    incorrectPoints: [],
    feedback: "ok",
    annotations: [
      { text: "green plants make their own food", type: "correct", comment: "Right.",
        marks: 500, confidence: 99, evidenceId: "totally-made-up-id" },
    ],
  }),
};

const rogue = await assessQuestion({
  answer: byNum["1"], evidence: photo.evidence, coverage: photo.coverage, llm: rogueLlm,
});
t("over-awarded marks clamp to the question maximum", rogue.marksAwarded === 2);
t("fabricated evidence id is stripped", rogue.annotations[0].evidenceId === undefined);
t("unsupported REFERENCE_SUPPORTED claim is downgraded", rogue.grounding !== GROUNDING.REFERENCE);

const honestLlm = {
  callJson: async ({ user }) => {
    const max = Number(/\((\d+) marks/.exec(user)[1]);
    return {
      marksAwarded: max, grounding: GROUNDING.REFERENCE, relevant: true,
      correctPoints: ["correct"], missingPoints: [], incorrectPoints: [], feedback: "Good.",
      annotations: [{ text: "green plants", type: "correct", comment: "Yes.", marks: max,
                      confidence: 90, evidenceId: photo.evidence[0].id }],
    };
  },
};

const paper = await assessPaper({ exam, answers: matched.answers, retriever, llm: honestLlm });
t("paper total is summed in code, not taken from the model", paper.totalMarks === 2 + 0 + 5);
t("maximum is summed from the question paper", paper.maximumMarks === 12);
t("skipped question contributes zero", paper.questions.find((q) => q.number === "2").marksAwarded === 0);
t("percentage derived from the two", Math.round(paper.percentage) === Math.round((7 / 12) * 100));
t("grade computed from percentage", paper.grade === gradeFor(paper.percentage));
t("no question exceeds its own maximum", paper.questions.every((q) => q.marksAwarded <= q.maxMarks));
t("valid evidence id survives", paper.questions[0].annotations[0].evidenceId === photo.evidence[0].id);
t("real citation keeps REFERENCE_SUPPORTED", paper.questions[0].grounding === GROUNDING.REFERENCE);

section("grounding without reference material (Phase 4 fallback)");
const noRefPaper = await assessPaper({ exam, answers: matched.answers, retriever: null, llm: honestLlm });
t("no reference -> INSUFFICIENT_REFERENCE", noRefPaper.questions[0].grounding === GROUNDING.INSUFFICIENT);
t("marks still awarded from general knowledge", noRefPaper.totalMarks === 7);
t("confidence is capped when reference is insufficient", noRefPaper.questions[0].confidence <= 65);

section("a failing question does not lose the paper");
let calls = 0;
const flakyLlm = {
  callJson: async (req) => {
    calls++;
    if (req.stage === "Q3") throw new Error("rate limit exhausted");
    const max = Number(/\((\d+) marks/.exec(req.user)[1]);
    return { marksAwarded: max, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
             missingPoints: [], incorrectPoints: [], feedback: "", annotations: [] };
  },
};
const warns = [];
const flaky = await assessPaper({ exam, answers: matched.answers, retriever, llm: flakyLlm, warnings: warns });
t("the other questions still marked", flaky.questions.find((q) => q.number === "1").marksAwarded === 2);
t("failed question scores zero, not full marks", flaky.questions.find((q) => q.number === "3").marksAwarded === 0);
t("failure is surfaced as a warning", warns.some((w) => w.includes("Q3")));

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall pipeline assertions passed");
process.exitCode = failures ? 1 : 0;
