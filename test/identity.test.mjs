/**
 * Question identity: the one place that decides what a written label means and
 * which of the paper's questions a block of text belongs to.
 *
 * The rule this file exists to hold: a paper, an answer sheet and a set of model
 * answers may each number their questions differently, and the pipeline must
 * still line them up. Anything that only works for "1." is a bug.
 */
import { readLabel, labelKey, labelParts, canonicalLabel, alignToQuestions, IDENTITY } from "../src/engine/identity.js";

let failures = 0;
const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) failures++;
};
const section = (s) => console.log("\n" + s);

/* ------------------------------------------------------ reading a label -- */
section("a label is read off the line however it was written");

const reads = [
  ["1. Define photosynthesis", "1"],
  ["1) Define photosynthesis", "1"],
  ["Q1 Define photosynthesis", "1"],
  ["Q.1 Define photosynthesis", "1"],
  ["Q 1: Define photosynthesis", "1"],
  ["Ans 1. Photosynthesis is", "1"],
  ["Answer Two: chlorophyll", "2"],
  ["Sol. 9 the answer", "9"],
  ["Q. No. 7 Explain", "7"],
  ["No. 4 The answer", "4"],
  ["Question Five is about plants", "5"],
  ["12. Explain the role", "12"],
  ["3(a) Mitosis produces", "3.a"],
  ["3 (a) Mitosis produces", "3.a"],
  ["2 a) first part", "2.a"],
  ["2a Mitosis is", "2.a"],
  ["Question 12 (b) Compare", "12.b"],
  ["Q.3 (a) (ii) deepest", "3.a.r2"],
  ["(c) Both are division", "c"],
  ["iii) The third point", "r3"],
  ["ii. Second point", "r2"],
];
for (const [line, key] of reads) {
  const hit = readLabel(line);
  t(`"${line.slice(0, 26)}" -> ${key}`, !!hit && labelKey(hit.label) === key);
}

section("prose is not a label");
const notLabels = [
  "A cell divides into two daughter cells",
  "Five plants were grown in the same soil",
  "The mitochondria is the powerhouse",
  "3.5 kg of water was added to it",
  "Now the plant begins to grow",
  "Note that water is needed here",
  "",
];
for (const line of notLabels) t(`"${line.slice(0, 32)}" is not a label`, readLabel(line) === null);

/* "1.1" and "3.5 kg" are the same shape, so neither is a label on its own. The
   paper resolves it: a paper that asks question 1.1 makes "1.1" a head, and
   still does not make "3.5" one. */
section("the paper's own numbering settles the ambiguous shapes");
const known = new Set(["1.1", "1.2"]);
t("\"1.1\" is not a label with no paper to check against", readLabel("1.1 Sub numbered item") === null);
t("but is one when the paper asks question 1.1",
  (readLabel("1.1 Sub numbered item", { known }) || {}).label === "1.1");
t("and a decimal quantity is still not a label",
  readLabel("3.5 kg of water was added to it", { known }) === null);
t("the guard is not weakened for prose",
  readLabel("A cell divides into two daughter cells", { known }) === null);

/* -------------------------------------------------------- one key space -- */
section("notations that name the same question agree");

const same = [
  ["Q.3 (a)", "3a"],
  ["Q.3 (a)", "Ans 3 a)"],
  ["Question Five", "5."],
  ["Q2(A)", "2 (a)"],
  ["Sol. 7", "7)"],
];
for (const [a, b] of same) t(`${a} = ${b}`, labelKey(a) === labelKey(b) && labelKey(a) !== "");

t("a roman sub-part is not a letter sub-part", labelKey("3(ii)") !== labelKey("3(b)"));
t("levels come back in order", JSON.stringify(labelParts("Q.3 (a) (ii)")) === '["3","a","ii"]');
t("the display form is unchanged for the common case", canonicalLabel("Q.1") === "1");
t("the display form keeps sub-parts", canonicalLabel("Ans 3a") === "3(a)");

/* ---------------------------------------------------------- alignment ---- */
section("blocks are aligned to the questions the paper actually has");

const exam = [
  { id: "qa", number: "1", text: "Define photosynthesis." },
  { id: "qb", number: "2", text: "Explain the role of chlorophyll." },
  { id: "qc", number: "3(a)", text: "Compare mitosis and meiosis." },
];

const aligned = alignToQuestions(
  [
    { label: "Ans 1.", text: "plants build food from sunlight" },
    { label: "Question Two", text: "the pigment that absorbs light" },
    { label: "3 a)", text: "mitosis gives two diploid cells" },
  ],
  exam
);
t("every question is placed", aligned.placed.size === 3);
t("nothing is left over", aligned.unplaced.length === 0);
t("and each placement says it came from the label",
  [...aligned.placed.values()].every((p) => p.confidence === IDENTITY.LABEL));

section("a shallower label still reaches the only question it could mean");
const loose = alignToQuestions([{ label: "3", text: "mitosis gives two cells" }], exam);
t("\"3\" reaches \"3(a)\" when nothing else could be meant", loose.placed.has("qc"));
t("and says the notation had to be resolved", loose.placed.get("qc").confidence === IDENTITY.LOOSE);

section("an ambiguous label is not guessed at");
const twoParts = [
  { id: "qa", number: "3(a)", text: "Compare mitosis and meiosis." },
  { id: "qb", number: "3(b)", text: "Describe the cell cycle." },
];
const ambiguous = alignToQuestions([{ label: "3", text: "all the best sir" }], twoParts);
t("a bare \"3\" against 3(a) and 3(b) is not claimed by its label", ambiguous.placed.size === 0);
t("and the block is reported unplaced", ambiguous.unplaced.length === 1);

/* Content may break a label tie — but it says so, and the lower confidence is
   what the Evaluate tab turns into "uncertain, verify this". */
const broken = alignToQuestions(
  [{ label: "3", text: "mitosis makes two diploid cells and meiosis makes four haploid ones" }],
  twoParts
);
t("content can settle which sub-part it was", broken.placed.has("qa"));
t("and never claims the label settled it", broken.placed.get("qa").confidence === IDENTITY.CONTENT);

section("content places a block only when one question clearly fits");
const byContent = alignToQuestions(
  [{ label: "", text: "chlorophyll is the pigment that absorbs light energy for the plant" }],
  exam
);
t("an unlabelled block can still find its question", byContent.placed.has("qb"));
t("and admits it was placed by content", byContent.placed.get("qb").confidence === IDENTITY.CONTENT);

const noFit = alignToQuestions([{ label: "", text: "all the best sir thank you" }], exam);
t("writing that answers nothing is left unplaced", noFit.placed.size === 0);

section("one block, one question");
const duplicate = alignToQuestions(
  [
    { label: "1.", text: "first attempt at photosynthesis" },
    { label: "1.", text: "second attempt at photosynthesis" },
  ],
  exam
);
t("a question is claimed once", duplicate.placed.size === 1);
t("the loser is reported rather than merged", duplicate.unplaced.length === 1);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall identity assertions passed");
process.exitCode = failures ? 1 : 0;
