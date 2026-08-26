/**
 * Optional questions: "Answer any 3 of the following 5".
 *
 * The rule this file holds: a student is scored out of what the paper required
 * them to attempt, never out of everything it printed. Scoring an any-3-of-5
 * paper out of five questions invents two questions' worth of failure and hands
 * it to the student.
 *
 * Nothing about a particular paper is configured anywhere. Every case below
 * reads its rule off the page, so a wording or a count this file does not list
 * is handled by the same code path.
 */
import { structureOcr } from "../src/engine/ocr.js";
import { validateExam } from "../src/engine/exam.js";
import { matchAnswers } from "../src/engine/match.js";
import { assessPaper, GROUNDING } from "../src/engine/assess.js";
import {
  parseAttemptRules,
  detectChoice,
  applyChoice,
  describeChoice,
  countFromWord,
  saysAllCompulsory,
} from "../src/engine/choice.js";

let failures = 0;
const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) failures++;
};
const section = (s) => console.log("\n" + s);

const linesOf = (text) => structureOcr([text]).lines;

/* ------------------------------------------------- reading the rubric ---- */
section("the rule is read off the paper, in whatever words it used");

t("digits", countFromWord("5") === 5);
t("words", countFromWord("five") === 5);
t("case", countFromWord("FIVE") === 5);
t("not a count", countFromWord("banana") === null);

const wordings = [
  ["Answer any 3 of the following 5 questions:", 3, 5],
  ["Attempt any five questions.", 5, null],
  ["Answer any THREE questions from Section B.", 3, null],
  ["Do any 2 of the following.", 2, null],
  ["Answer any 4 of the following 6 questions", 4, 6],
  ["Solve any two of the following three questions", 2, 3],
];
for (const [text, required, of_] of wordings) {
  const [rule] = parseAttemptRules(linesOf(text));
  t(`"${text.slice(0, 40)}" -> any ${required}`, !!rule && rule.required === required);
  if (of_ !== null) t(`  …of ${of_}`, !!rule && rule.of === of_);
}

t("a paper with no rule has no rule", parseAttemptRules(linesOf("1. Define photosynthesis.")).length === 0);
t("compulsory is recognised as its own statement",
  saysAllCompulsory(linesOf("All questions are compulsory.")));

/* ------------------------------------------------- scoping the rule ------ */
section("a rule governs the questions that follow it");

const twoSections = detectChoice(
  linesOf(
    [
      "Section A",
      "All questions are compulsory.",
      "1. Define photosynthesis. (2 marks)",
      "2. Explain chlorophyll. (3 marks)",
      "Section B",
      "Answer any 3 of the following 5 questions:",
      "3. Compare mitosis and meiosis. (5 marks)",
      "4. Describe the cell cycle. (5 marks)",
      "5. Explain osmosis. (5 marks)",
      "6. Explain diffusion. (5 marks)",
      "7. Describe the nucleus. (5 marks)",
    ].join("\n")
  )
);
t("one group is found", twoSections.length === 1);
t("it asks for 3", twoSections[0].required === 3);
t("it covers only Section B", twoSections[0].numbers.join(",") === "3,4,5,6,7");
t("the compulsory section is untouched", !twoSections[0].numbers.includes("1"));

const bounded = detectChoice(
  linesOf(
    ["Answer any 4 of the following 6 questions", ...Array.from({ length: 7 }, (_, i) => `${i + 1}. Question ${i + 1}. (3 marks)`)].join("\n")
  )
);
t('"of the following 6" bounds the group at 6', bounded[0].numbers.length === 6);

t("a rule asking for more than it governs is dropped",
  detectChoice(linesOf(["Answer any 5 questions", "1. Only one question. (2 marks)"].join("\n"))).length === 0);
t("a rule with no real choice is not a choice",
  detectChoice(linesOf(["Answer any 2 of the following 2", "1. a (2)", "2. b (2)"].join("\n"))).length === 0);

/* --------------------------------------------------- applying the rule --- */
section("the best attempts fill the slots");

const five = (awarded) =>
  awarded.map((m, i) => ({
    questionId: `q${i + 1}`,
    number: String(i + 1),
    maxMarks: 5,
    marksAwarded: m === null ? 0 : m,
    skipped: m === null,
  }));
const group = [{ required: 3, numbers: ["1", "2", "3", "4", "5"] }];

const best = applyChoice(five([5, 4, 3, 2, 1]), group);
t("the three best count", ["q1", "q2", "q3"].every((id) => best.counted.has(id)));
t("the extras do not", !best.counted.has("q4") && !best.counted.has("q5"));
t("the total is the best three", best.totalMarks === 12);
t("out of three questions, not five", best.maximumMarks === 15);

const outOfOrder = applyChoice(five([1, 2, 5, 4, 3]), group);
t("order on the page does not decide it",
  outOfOrder.counted.has("q3") && outOfOrder.counted.has("q4") && outOfOrder.counted.has("q5"));
t("and the weakest two are dropped", !outOfOrder.counted.has("q1") && !outOfOrder.counted.has("q2"));

section("attempting fewer than required loses the difference");
const short = applyChoice(five([5, 4, null, null, null]), group);
t("both attempts count", short.counted.has("q1") && short.counted.has("q2"));
t("an empty slot is still a slot", short.maximumMarks === 15);
t("and the student is scored on what they earned", short.totalMarks === 9);

section("attempting exactly the required number changes nothing");
const exact = applyChoice(five([5, 4, 3, null, null]), group);
t("all three attempts count", exact.totalMarks === 12 && exact.maximumMarks === 15);

section("a question that could not be evaluated is not treated as a bad attempt");
const withFailure = applyChoice(
  [
    { questionId: "q1", number: "1", maxMarks: 5, marksAwarded: 5, skipped: false },
    { questionId: "q2", number: "2", maxMarks: 5, marksAwarded: 4, skipped: false },
    { questionId: "q3", number: "3", maxMarks: 5, marksAwarded: 0, skipped: false, failed: true },
    { questionId: "q4", number: "4", maxMarks: 5, marksAwarded: 0, skipped: false },
  ],
  [{ required: 3, numbers: ["1", "2", "3", "4"] }]
);
t("the unevaluated attempt takes the last slot over a genuine zero",
  withFailure.counted.has("q3") && !withFailure.counted.has("q4"));

section("no rule, no change");
const noChoice = applyChoice(five([5, 4, 3, 2, 1]), []);
t("every question counts", noChoice.counted.size === 5);
t("and the paper is worth all of it", noChoice.maximumMarks === 25);

t("the rule is describable in one line",
  /any 3 of 5/.test(describeChoice(group)));

/* ------------------------------------------------- end to end ------------ */
section("a whole paper, scored out of what it asked for");

const paperText = [
  "Answer any 3 of the following 5 questions:",
  "1. Define photosynthesis. (5 marks)",
  "2. Explain the role of chlorophyll. (5 marks)",
  "3. Compare mitosis and meiosis. (5 marks)",
  "4. Describe osmosis. (5 marks)",
  "5. Explain diffusion. (5 marks)",
].join("\n");

const exam = validateExam(
  {
    totalMarks: 15,
    questions: [
      { number: "1", text: "Define photosynthesis.", maxMarks: 5 },
      { number: "2", text: "Explain the role of chlorophyll.", maxMarks: 5 },
      { number: "3", text: "Compare mitosis and meiosis.", maxMarks: 5 },
      { number: "4", text: "Describe osmosis.", maxMarks: 5 },
      { number: "5", text: "Explain diffusion.", maxMarks: 5 },
    ],
  },
  { lines: linesOf(paperText) }
);

t("the paper is worth 15, not 25", exam.totalMarks === 15);
t("what it printed is kept alongside", exam.printedMarks === 25);
t("the declared total no longer looks like a mismatch",
  !exam.warnings.some((w) => /add up to|is worth/.test(w)));
t("and the choice is stated for the examiner",
  exam.warnings.some((w) => /offers a choice/.test(w)));

const answers = structureOcr([
  [
    "1. Photosynthesis is how green plants make their own food from sunlight.",
    "2. Chlorophyll is the pigment that absorbs the light energy for the plant.",
    "3. Mitosis makes two diploid cells and meiosis makes four haploid cells.",
    "4. Osmosis is the movement of water across a semi permeable membrane.",
  ].join("\n"),
]);
const matched = await matchAnswers(answers, exam, { llm: null });

let nth = 0;
const scores = [5, 4, 3, 2];
const llm = {
  callJson: async () => ({
    marksAwarded: scores[nth++ % scores.length],
    grounding: GROUNDING.GENERAL,
    relevant: true,
    correctPoints: [],
    missingPoints: [],
    incorrectPoints: [],
    feedback: "",
    annotations: [],
  }),
};

const paper = await assessPaper({ exam, answers: matched.answers, retriever: null, llm });
t("scored out of the three required questions", paper.maximumMarks === 15);
t("with the best three attempts", paper.totalMarks === 12);
t("the fourth attempt is marked but not counted",
  paper.questions.find((q) => q.number === "4").counted === false);
t("the unattempted fifth is not held against the student",
  paper.questions.find((q) => q.number === "5").counted === false);
t("and the examiner is told which is which",
  paper.dropped.length === 2 && paper.dropped.some((d) => /extra attempt/.test(d.reason)));
t("the percentage follows the real denominator",
  Math.round(paper.percentage) === Math.round((12 / 15) * 100));

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall choice assertions passed");
process.exitCode = failures ? 1 : 0;
