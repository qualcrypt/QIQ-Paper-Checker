/**
 * Putting the marks back on the paper.
 *
 * This is the chain the whole product turns on: a verbatim phrase the examiner
 * quoted becomes a character span, the span becomes the physical lines that
 * carry it, and those lines carry boxes measured off the page image. Every hop
 * is reversible and every hop can fail cleanly — a mark that cannot be placed
 * becomes a margin note rather than a box in the wrong place.
 *
 * None of it trusts a coordinate from a model. See geometry.js for why.
 */

import { findSpan } from "./text.js";
import { linesForSpan } from "./ocr.js";
import { detectLineBands, alignBandsToLines, unionBoxes } from "./geometry.js";

/**
 * Locate each annotation inside the student's answer and slice the answer into
 * renderable segments. Annotations that cannot be pinned (typically "missing"
 * ones) come back unanchored and are shown in the margin only.
 *
 * The matching itself is text.js's findSpan: exact hit first, then the
 * normalised form, then progressively shorter word-prefixes, skipping any span
 * already claimed by an earlier annotation.
 */
export function anchorAnnotations(answer, annotations) {
  const taken = [];
  const placed = [];
  const unanchored = [];

  annotations.forEach((ann, idx) => {
    /* "missing" describes content the student never wrote, so there is nothing
       on the page to point at. The length floor matters too: findSpan will take
       an exact hit of any length, and a two-character quote would happily land
       inside an unrelated word. */
    const needle = String(ann.text || "").trim();
    const range =
      ann.type === "missing" || needle.length < 3 ? null : findSpan(answer, needle, { taken });

    if (range) {
      taken.push(range);
      placed.push({ start: range.start, end: range.end, ann, idx });
    } else {
      unanchored.push({ ann, idx });
    }
  });

  placed.sort((a, b) => a.start - b.start);

  const segments = [];
  let cursor = 0;
  for (const p of placed) {
    if (p.start > cursor) segments.push({ text: answer.slice(cursor, p.start) });
    segments.push({ text: answer.slice(p.start, p.end), ann: p.ann, idx: p.idx });
    cursor = p.end;
  }
  if (cursor < answer.length) segments.push({ text: answer.slice(cursor) });

  // `placed` carries the raw character spans. The text view only needs the
  // sliced segments, but the page overlay needs the spans themselves, to walk
  // them back through the OCR lines and onto the image.
  return { segments, anchoredCount: placed.length, unanchored, placed };
}

/* ======================================================== PAGE GEOMETRY == */

/**
 * Measure where the writing sits on every uploaded page.
 *
 * Runs once per upload and is cached, because it depends only on the images —
 * re-grading or correcting the OCR text does not change where the ink is.
 * A page that cannot be measured is skipped rather than faked; its annotations
 * fall back to margin notes.
 *
 * @returns {Promise<Object.<number, {bands: any[]}>>} keyed by 1-based page number
 */
export async function measurePages(pages) {
  const measured = {};
  for (let i = 0; i < pages.length; i++) {
    try {
      measured[i + 1] = await detectLineBands(pages[i].dataUrl);
    } catch {
      // Leave the page unmeasured; attachGeometry reports it as "none".
    }
  }
  return measured;
}

/**
 * Give every OCR line a box on its page, and report how much to trust it.
 * Mutates `doc.lines`, filling the `bbox` field that types.js has always
 * declared and that no producer previously filled.
 *
 * @returns {Object.<number, "high"|"medium"|"low"|"none">} confidence per page
 */
export function attachGeometry(doc, measured) {
  const byPage = {};

  for (let page = 1; page <= doc.pageCount; page++) {
    const pageLines = doc.lines.filter((l) => l.page === page);
    const detected = measured[page];

    if (!detected || pageLines.length === 0) {
      byPage[page] = "none";
      continue;
    }

    const { confidence, boxes } = alignBandsToLines(detected.bands, pageLines);
    byPage[page] = confidence;
    for (const line of pageLines) {
      if (boxes[line.id]) line.bbox = boxes[line.id];
    }
  }

  return byPage;
}

/**
 * Turn each anchored annotation into boxes on the page images.
 *
 * The chain is: verbatim phrase → character span (anchorAnnotations) → OCR
 * lines (linesForSpan) → the union of their measured boxes. A span that runs
 * across a page break yields one box per page rather than a single box spanning
 * nothing.
 *
 * @returns {Object.<number, {page: number, bbox: object}[]>} keyed by annotation index
 */
export function resolveAnnotationBoxes(doc, placed) {
  const out = {};
  if (!doc || !placed) return out;

  for (const p of placed) {
    const hits = linesForSpan(doc.lines, p.start, p.end).filter((l) => l.bbox);
    if (hits.length === 0) continue;

    const byPage = new Map();
    for (const line of hits) {
      if (!byPage.has(line.page)) byPage.set(line.page, []);
      byPage.get(line.page).push(line.bbox);
    }

    const boxes = [];
    for (const [page, list] of byPage) {
      const bbox = unionBoxes(list);
      if (bbox) boxes.push({ page, bbox });
    }
    if (boxes.length) out[p.idx] = boxes;
  }

  return out;
}

/** The weakest link across the pages that carry writing. */
export function overallGeometry(byPage) {
  const ranks = { high: 3, medium: 2, low: 1, none: 0 };
  const values = Object.values(byPage || {});
  if (values.length === 0) return "none";
  return values.reduce((worst, v) => (ranks[v] < ranks[worst] ? v : worst), "high");
}
