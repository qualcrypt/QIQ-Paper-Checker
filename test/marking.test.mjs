/**
 * The whole marking chain, end to end, on the real engine code:
 *   OCR page texts -> structured lines -> measured bands -> annotation boxes
 * Everything except the canvas pixel read (covered by geom-test.mjs).
 */
import { structureOcr } from "../src/engine/ocr.js";
import {
  anchorAnnotations, attachGeometry, resolveAnnotationBoxes, overallGeometry,
} from "../src/engine/marking.js";

const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) process.exitCode = 1;
};

/* A two-page paper, as runOcr() now returns it: one string per page. */
const pageTexts = [
  [
    "Q1. Explain the process of photosynthesis.",
    "Photosynthesis is the process by which green plants make their own food.",
    "Chlorophyll present in the leaves absorbs sunlight.",
    "The equation is 6CO2 + 6H2O -> C6H12O6 + 6O2.",
  ].join("\n"),
  [
    "The light reaction takes place in the thylakoid membrane.",
    "The dark reaction happens only at night.",
    "Oxygen is released as a by-product of the reaction.",
  ].join("\n"),
];

const doc = structureOcr(pageTexts);
console.log("structure");
t("7 lines across 2 pages", doc.lines.length === 7 && doc.pageCount === 2);
t("page attribution correct",
  doc.lines.filter((l) => l.page === 1).length === 4 && doc.lines.filter((l) => l.page === 2).length === 3);
t("every offset slices back to its own text",
  doc.lines.every((l) => doc.text.slice(l.start, l.end) === l.text));

/* Bands as detectLineBands would report them: evenly spaced, correct count. */
const bandsFor = (n, from = 0.1, step = 0.09) =>
  Array.from({ length: n }, (_, i) => ({
    top: from + i * step, bottom: from + i * step + 0.055, left: 0.09, right: 0.86,
  }));

const measured = { 1: { bands: bandsFor(4) }, 2: { bands: bandsFor(3) } };
const conf = attachGeometry(doc, measured);
console.log("geometry attachment");
t("both pages report high confidence", conf[1] === "high" && conf[2] === "high");
t("every line received a bbox", doc.lines.every((l) => l.bbox));
t("boxes are fractions in 0..1",
  doc.lines.every((l) => l.bbox.x >= 0 && l.bbox.y >= 0 && l.bbox.x + l.bbox.width <= 1 && l.bbox.y + l.bbox.height <= 1));
t("boxes descend the page in reading order", (() => {
  const p1 = doc.lines.filter((l) => l.page === 1).map((l) => l.bbox.y);
  return p1.every((v, i) => i === 0 || v > p1[i - 1]);
})());

/* Annotations exactly as the examiner returns them: verbatim quotes. */
const annotations = [
  { text: "Chlorophyll present in the leaves absorbs sunlight", type: "correct", marks: 3, comment: "Correct.", confidence: 92 },
  { text: "The dark reaction happens only at night", type: "wrong", marks: 0, comment: "It is light-independent, not nocturnal.", confidence: 88 },
  { text: "6CO2 + 6H2O -> C6H12O6 + 6O2", type: "correct", marks: 4, comment: "Balanced equation correct.", confidence: 95 },
  { text: "the role of ATP and NADPH", type: "missing", marks: 0, comment: "Not mentioned at all.", confidence: 70 },
];

const anchored = anchorAnnotations(doc.text, annotations);
const boxes = resolveAnnotationBoxes(doc, anchored.placed);

console.log("annotation placement");
t("3 of 4 anchored, the 'missing' one is not", anchored.anchoredCount === 3 && anchored.unanchored.length === 1);
t("the unanchored one is the 'missing' annotation", anchored.unanchored[0].idx === 3);
t("all 3 anchored annotations got page boxes", Object.keys(boxes).length === 3);

// The critical assertion: does each mark land on the line that actually says it?
const landedOn = (idx) => {
  const entry = boxes[idx];
  if (!entry) return null;
  const line = doc.lines.find(
    (l) => l.page === entry[0].page && Math.abs(l.bbox.y - entry[0].bbox.y) < 1e-9
  );
  return line ? line.text : null;
};
console.log("  ---- what each mark landed on ----");
[0, 1, 2].forEach((i) => console.log(`   [${i}] ${JSON.stringify(annotations[i].text.slice(0, 40))} -> ${JSON.stringify(landedOn(i))}`));

t("chlorophyll mark landed on the chlorophyll line",
  (landedOn(0) || "").startsWith("Chlorophyll present"));
t("dark-reaction mark landed on the dark-reaction line",
  (landedOn(1) || "").startsWith("The dark reaction"));
t("equation mark landed on the equation line",
  (landedOn(2) || "").includes("6CO2"));
t("page-2 annotation is reported on page 2", boxes[1][0].page === 2);
t("page-1 annotations are reported on page 1", boxes[0][0].page === 1 && boxes[2][0].page === 1);

/* Degradation: a page that could not be measured must lose its boxes, not fake them. */
console.log("degradation");
const doc2 = structureOcr(pageTexts);
const conf2 = attachGeometry(doc2, { 1: { bands: bandsFor(4) } }); // page 2 unmeasured
const a2 = anchorAnnotations(doc2.text, annotations);
const b2 = resolveAnnotationBoxes(doc2, a2.placed);
t("unmeasured page reports 'none'", conf2[2] === "none");
t("overall confidence is the weakest page", overallGeometry(conf2) === "none");
t("page-1 marks still placed", b2[0] && b2[0][0].page === 1);
t("page-2 mark produced no box at all", !b2[1]);

const conf3 = attachGeometry(structureOcr(pageTexts), { 1: { bands: bandsFor(2) }, 2: { bands: bandsFor(3) } });
t("band/line mismatch downgrades that page", conf3[1] === "low" && conf3[2] === "high");
t("overall takes the worst page", overallGeometry(conf3) === "low");

/* A span crossing the page break must yield one box per page. */
console.log("cross-page span");
const crossText = "6O2.\n\nThe light reaction";
const at = doc.text.indexOf(crossText);
const cross = resolveAnnotationBoxes(doc, [{ start: at, end: at + crossText.length, idx: 99 }]);
t("cross-page span found", at !== -1);
t("yields two boxes, one per page",
  cross[99] && cross[99].length === 2 && cross[99][0].page === 1 && cross[99][1].page === 2);

console.log(process.exitCode ? "\nFAILURES ABOVE" : "\nall chain assertions passed");
