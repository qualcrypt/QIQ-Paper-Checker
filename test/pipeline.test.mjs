/**
 * The question-paper-first pipeline: exam validation, reference retrieval,
 * answer matching and mark control. A stub LLM stands in for Groq so the
 * deterministic guards — the parts that must hold no matter what the model
 * says — are what actually gets tested.
 */
import { validateExam, classifyQuestion, criteriaFor, deriveMarks } from "../src/engine/exam.js";
import { chunkDocument, createRetriever, WEAK_COVERAGE } from "../src/engine/reference.js";
import { matchAnswers, canonicalNumber, chunkLines, chunkQuestions } from "../src/engine/match.js";
import {
  assessPaper, assessQuestion, gradeFor, GROUNDING,
  toEvaluation, answerStatus, ANSWER_STATUS, answerKeySection,
  numberedSections, pairFor, pairReferenceAnswers,
} from "../src/engine/assess.js";
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
/* A bare sub-label no longer carries brackets of its own. Where it belongs is
   decided by aligning it against the exam, not by the shape of the string —
   see identity.js. */
t("bare (a)", canonicalNumber("(a)") === "a");
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

/* ------------------------------------- a short answer is still an answer --
   A one-mark question is answered in a word. A flat character floor called
   those blank, scored them zero and reported "no answer detected" with the
   student's correct answer sitting on the page. */
section("brevity is not absence");

const shortExam = validateExam({
  totalMarks: 7,
  questions: [
    { number: "1", text: "The powerhouse of the cell is ____.", maxMarks: 1 },
    { number: "2", text: "State the value of k.", maxMarks: 1 },
    { number: "3", text: "Explain the process of photosynthesis in detail.", maxMarks: 5 },
  ],
});
const shortDoc = structureOcr([
  ["1. Mitochondria", "2. k = 3", "3. ok"].join("\n"),
]);
const shortMatch = await matchAnswers(shortDoc, shortExam, { llm: null });
const shortBy = Object.fromEntries(shortMatch.answers.map((a) => [a.number, a]));

t("a one-word answer to a one-mark question is an answer", !shortBy["1"].skipped);
t("and keeps the confidence its label earned", shortBy["1"].confidence >= 90);
t("\"k = 3\" is an answer too", !shortBy["2"].skipped && shortBy["2"].confidence >= 90);
t("but two characters against five marks is still not one", shortBy["3"].skipped === true);
t("an unanswerably short answer is still reported honestly",
  answerStatus(shortBy["3"]) !== ANSWER_STATUS.DETECTED);

/* ---------------------------------- confidence is measured, not defaulted --
   Every question on every paper came back at exactly 70% because the schema
   filled in 70 whenever the model said nothing. */
section("confidence comes from the marker or from the detection, never from a default");

const noConfidenceLlm = {
  callJson: async () => ({
    marksAwarded: 1, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [{ text: "Mitochondria", type: "correct", comment: "Right." }],
  }),
};
const unreported = await assessQuestion({
  answer: shortBy["1"], evidence: [], coverage: 0, llm: noConfidenceLlm,
});
/* The detection's own number carries through, capped only by the rule that a
   mark made without usable reference material cannot claim high confidence. */
t("with nothing reported, the detection's own confidence stands",
  unreported.confidence === Math.min(shortBy["1"].confidence, 65));
t("and it is not the old constant", unreported.confidence !== 70);

const confidentLlm = {
  callJson: async () => ({
    marksAwarded: 1, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [
      { text: "Mitochondria", type: "correct", comment: "Right.", confidence: 40 },
      { text: "Mitochondria", type: "correct", comment: "Also.", confidence: 60 },
    ],
  }),
};
const measured = await assessQuestion({
  answer: shortBy["1"], evidence: [], coverage: 0, llm: confidentLlm,
});
t("what the marker did report is what is used", measured.confidence === 50);

/* ------------------------------ pairing the reference to the paper by number --
   Both sides of the marking carry question numbers: the paper asks question 5,
   and a set of model answers answers question 5. Searching the reference by
   topic can land on the wrong passage; matching the number cannot. */
section("model answers are paired to questions by their number");

/* Deliberately written in a different notation from the exam: this is the case
   that used to fail, and the whole point of aligning rather than string-matching. */
const modelAnswerText =
  "Ans 1. Photosynthesis is how green plants build food from sunlight, water and carbon dioxide.\n" +
  "Answer Two: chlorophyll is the pigment that absorbs the light energy the plant uses.\n" +
  "Q.3 Mitosis yields two diploid cells; meiosis yields four haploid cells.\n" +
  "Sol. 9 A question this paper never asked.";

const sections = numberedSections(modelAnswerText);
t("every numbered section is found", sections.length === 4);
t("sections keep the label as written", sections[0].label.toLowerCase().includes("1"));
t("and sections do not bleed into each other",
  /carbon dioxide/.test(sections[0].text) && !/chlorophyll/.test(sections[0].text));

const refIndex = pairReferenceAnswers(modelAnswerText, exam.questions);
t("a key written as \"Ans 1.\" reaches the paper's question 1",
  /sunlight/.test(refIndex.get(exam.questions[0].id).text));
t("a key written as \"Answer Two\" reaches question 2",
  /pigment/.test(refIndex.get(exam.questions[1].id).text));
t("a key written as \"Q.3\" reaches question 3",
  /haploid/.test(refIndex.get(exam.questions[2].id).text));
t("an answer to a question this paper never asked is left out", refIndex.size === 3);

t("pairing produces citable evidence",
  pairFor(refIndex, { questionId: exam.questions[1].id, number: "2" }).id === "refq-2");
t("it says where it came from",
  /question 2/.test(pairFor(refIndex, { questionId: exam.questions[1].id, number: "2" }).metadata.source));
t("a question with no model answer pairs with nothing",
  pairFor(refIndex, { questionId: "nope", number: "9" }) === null);
t("no index at all is not an error", pairFor(null, { questionId: "a", number: "1" }) === null);

const prompts = [];
const capturingLlm = {
  callJson: async ({ user }) => {
    prompts.push(user);
    return {
      marksAwarded: 1, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
      missingPoints: [], incorrectPoints: [], feedback: "", annotations: [],
    };
  },
};

const pairedPaper = await assessPaper({
  exam,
  answers: matched.answers,
  retriever: null,
  llm: capturingLlm,
  referenceAnswers: refIndex,
});

const q1 = pairedPaper.questions.find((q) => q.number === "1");
t("the paired answer is attached to the question as evidence",
  q1.evidence.some((e) => e.id === "refq-1"));
t("the marker is told to compare it point by point",
  prompts.some((u) => /model answer filed under question 1/.test(u)));
t("and told that different wording still earns the mark",
  prompts.some((u) => /same idea in the student's own words earns the mark/.test(u)));
t("a question the reference does not cover is still marked",
  pairedPaper.questions.find((q) => q.number === "3").marksAwarded >= 0);

/* ------------------------------------ a mark badge states only what is true --
   Every annotation used to default to 0 marks, so a correct highlight was
   stamped "+0" — a zero the student never earned, printed beside work that was
   right. Silence stays silence now, and numbers that contradict the mark
   awarded are dropped rather than shown. */
section("per-annotation marks are kept only when they can be true");

const silentLlm = {
  callJson: async () => ({
    marksAwarded: 2, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [{ text: "green plants make their own food", type: "correct", comment: "Right." }],
  }),
};
const silent = await assessQuestion({ answer: byNum["1"], evidence: [], coverage: 0, llm: silentLlm });
/* One highlight and one mark can only be allocated one way, so it is allocated:
   the examiner sees the breakdown rather than a bare verdict. */
t("the only highlight there is carries the mark", silent.annotations[0].marks === 2);
t("the question's own mark is unaffected", silent.marksAwarded === 2);

const twoSilentLlm = {
  callJson: async () => ({
    marksAwarded: 3, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [
      { text: "green plants", type: "correct", comment: "a" },
      { text: "own food", type: "correct", comment: "b" },
    ],
  }),
};
const twoSilent = await assessQuestion({
  answer: { ...byNum["1"], maxMarks: 4 }, evidence: [], coverage: 0, llm: twoSilentLlm,
});
t("two highlights and one total are not split by guesswork",
  twoSilent.annotations.every((a) => a.marks === undefined));

const overclaimLlm = {
  callJson: async () => ({
    marksAwarded: 1, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [
      { text: "green plants", type: "correct", comment: "a", marks: 2 },
      { text: "own food", type: "correct", comment: "b", marks: 2 },
    ],
  }),
};
const overclaim = await assessQuestion({ answer: byNum["1"], evidence: [], coverage: 0, llm: overclaimLlm });
t("marks that add up to more than was awarded are dropped",
  overclaim.annotations.every((a) => a.marks === undefined));
t("the highlights themselves survive", overclaim.annotations.length === 2);

const honestLlmMarks = {
  callJson: async () => ({
    marksAwarded: 2, grounding: GROUNDING.GENERAL, relevant: true, correctPoints: [],
    missingPoints: [], incorrectPoints: [], feedback: "",
    annotations: [{ text: "green plants", type: "correct", comment: "a", marks: 2 }],
  }),
};
const honestMarks = await assessQuestion({ answer: byNum["1"], evidence: [], coverage: 0, llm: honestLlmMarks });
t("marks that can be true are kept", honestMarks.annotations[0].marks === 2);

/* --------------------------------------- the answer key reaches every question --
   The key used to be truncated at a fixed character count and handed to every
   question alike, so on a long paper the last questions were marked without
   their key — silently, with the answers sitting just past the cut. */
section("a long answer key is addressed by question, not truncated");

const longKey = Array.from({ length: 14 }, (_, i) => {
  const n = i + 1;
  return `${n}. The expected answer for this one names marker${n} explicitly. ` + "detail ".repeat(45);
}).join("\n");

t("the key is long enough that a fixed cut would lose its tail", longKey.length > 4000);
t("an early question still gets its own section",
  /marker1\b/.test(answerKeySection(longKey, "1").text));
t("a late question gets its own section too — this is what the cut used to lose",
  /marker14\b/.test(answerKeySection(longKey, "14").text));
t("and not some other question's answer", !/marker3\b/.test(answerKeySection(longKey, "14").text));
t("the section is reported as a section", answerKeySection(longKey, "9").scope === "section");
t("a short key is still sent whole", answerKeySection("Photosynthesis needs light.", "1").scope === "whole");
t("no key stays no key", answerKeySection("", "1").scope === "none");
t("sub-parts are addressed too",
  /marker2\b/.test(answerKeySection(longKey, "2").text));

/* A key with no numbering at all cannot be addressed by number — it falls back
   on retrieval rather than on the first N characters. */
const proseKey =
  "Photosynthesis converts light energy into chemical energy in chloroplasts. " .repeat(20) +
  "Mitosis produces two identical diploid cells and is used for growth and repair. ".repeat(20);
const prose = answerKeySection(proseKey, "3", "Compare mitosis and meiosis.");
t("an unnumbered key falls back on retrieval", prose.scope === "retrieved" || prose.scope === "whole");
t("and the fallback finds the relevant half", /[Mm]itosis/.test(prose.text));

/* ------------------------------------------- a long paper is matched whole --
   The token ceiling counts prompt plus completion budget up front, so one
   request carrying a whole answer sheet is refused before the model sees it.
   That refusal used to take every unlabelled answer with it, and the examiner
   was shown a half-checked paper with no obvious sign of why. */
section("a long paper is matched in batches, not in one oversized call");

const bigExam = validateExam({
  totalMarks: 24,
  questions: Array.from({ length: 12 }, (_, i) => ({
    number: String(i + 1),
    text: `Explain topic${i + 1} and give one example of it.`,
    maxMarks: 2,
  })),
});

/* Unlabelled answers, in the paper's order — the case route 1 cannot help
   with. Long enough that the listing cannot fit one call. */
const bigPages = Array.from({ length: 4 }, (_, page) =>
  Array.from({ length: 3 }, (_, j) => {
    const n = page * 3 + j + 1;
    return Array.from(
      { length: 5 },
      (_, l) =>
        `Regarding topic${n}, the student writes a fairly long sentence number ${l} that carries ` +
        `enough characters to make the listing realistic for a real answer sheet.`
    ).join("\n");
  }).join("\n")
);
const bigDoc = structureOcr(bigPages);

const matchCalls = [];
const batchLlm = {
  callJson: async ({ user }) => {
    matchCalls.push(user);
    const asked = [...user.matchAll(/^(\d+)\. Explain/gm)].map((m) => m[1]);
    const listing = user.slice(user.indexOf("UNASSIGNED LINES"));
    const matches = [];
    for (const n of asked) {
      const lineIds = [...listing.matchAll(/^(L\d+)\t(.*)$/gm)]
        .filter(([, , text]) => new RegExp(`topic${n},`).test(text))
        .map(([, id]) => id);
      if (lineIds.length) matches.push({ questionNumber: n, lineIds });
    }
    return { matches };
  },
};

const bigWarns = [];
const bigMatch = await matchAnswers(bigDoc, bigExam, { llm: batchLlm, warnings: bigWarns });
t("the whole paper is matched, not just what fits one call",
  bigMatch.answers.every((a) => !a.skipped && a.answerText.includes("topic" + a.number + ",")));
t("it took more than one call to do it", matchCalls.length > 1);
t("no single call carries an oversized listing", matchCalls.every((u) => u.length < 12000));
t("matching raised no warnings when it worked", bigWarns.length === 0);

t("line windows respect the budget", chunkLines(bigDoc.lines, 2000).every((w, i, all) =>
  i === all.length - 1 || w.reduce((n, l) => n + l.text.length + 12, 0) <= 2000 + 200));
t("line windows overlap so a straddling answer is still visible",
  chunkLines(bigDoc.lines, 2000).slice(1).every((w, i) => {
    const prev = chunkLines(bigDoc.lines, 2000)[i];
    return prev.some((l) => w[0].id === l.id);
  }));
t("every line appears in some window",
  new Set(chunkLines(bigDoc.lines, 2000).flat().map((l) => l.id)).size === bigDoc.lines.length);
t("question batches are capped", chunkQuestions(bigExam.questions).every((b) => b.length <= 8));
t("no question is dropped from the batches",
  chunkQuestions(bigExam.questions).flat().length === bigExam.questions.length);

section("one failed matching call does not lose the rest of the paper");
let nth = 0;
const flakyMatchLlm = {
  callJson: async (req) => {
    if (++nth === 1) throw new Error("request too large");
    return batchLlm.callJson(req);
  },
};
const partialWarns = [];
const partial = await matchAnswers(bigDoc, bigExam, { llm: flakyMatchLlm, warnings: partialWarns });
t("the questions in the other batches are still matched",
  partial.answers.filter((a) => !a.skipped).length > 0);
t("the failure is reported against the questions it cost",
  partialWarns.some((w) => /Answer matching failed for Q/.test(w)));
t("and says they were not guessed", partialWarns.some((w) => /rather than guessed/.test(w)));

/* ---------------------------------------- what the views are told (Phase 11) --
   The UI classifies every question through answerStatus() over the keyPoints
   that toEvaluation() produces. If that projection drops a field the classifier
   reads, every answered question silently reports as unanswered — which is the
   one claim about a student the app must never make by accident. */
section("key points carry what the status vocabulary needs");

const evalView = toEvaluation(flaky, "remark");
const kp = Object.fromEntries(evalView.keyPoints.map((k) => [k.questionNumber, k]));
const statusOf = (k, unassigned = false) => answerStatus(k, { hasUnassignedWriting: unassigned });

t("key points keep the matched answer text", /green plants make their own food/.test(kp["1"].answerText));
t("an answered question reads as detected", statusOf(kp["1"]) === ANSWER_STATUS.DETECTED);
t("  and never as unanswered", statusOf(kp["1"], true) !== ANSWER_STATUS.NOT_DETECTED);
t("a failed evaluation reads as failed, not as a zero", statusOf(kp["3"]) === ANSWER_STATUS.FAILED);
t("  even though its placeholder mark is 0", kp["3"].marksAwarded === 0 && kp["3"].failed === true);
t("an unmatched question with no stray writing is unanswered",
  statusOf(kp["2"]) === ANSWER_STATUS.UNANSWERED);
t("  but is only 'not detected' when stray writing exists",
  statusOf(kp["2"], true) === ANSWER_STATUS.NOT_DETECTED);
t("the failed question's marks are pending, not lost",
  kp["3"].marksTotal === 5 && evalView.totalMarks === 12);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall pipeline assertions passed");
process.exitCode = failures ? 1 : 0;
