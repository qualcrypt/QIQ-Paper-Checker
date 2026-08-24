/**
 * Turns raw OCR output into a structured document.
 *
 * The vision model used by this project (qwen3.6-27b) returns plain text per
 * page with no geometry, so `bbox` is left undefined rather than fabricated —
 * the OCRLine contract carries it for providers that do report boxes. What we
 * *can* derive honestly is reading order, page numbers and a legibility
 * estimate, and the confidence engine consumes the latter as a real signal.
 */

import { normalizeText } from "./text.js";

/** Markers a transcriber leaves when the handwriting could not be read. */
const ILLEGIBLE = /\[(illegible|unclear|unreadable|\?+)\]|\?\?+|\bx{3,}\b|_{3,}/i;

/**
 * @param {string[]} pageTexts  One entry per page, in order.
 * @returns {import("./types.js").OCRDocument}
 */
export function structureOcr(pageTexts) {
  const pages = Array.isArray(pageTexts) ? pageTexts : [String(pageTexts || "")];
  /** @type {import("./types.js").OCRLine[]} */
  const lines = [];
  const chunks = [];
  let index = 0;
  let base = 0; // where the current page starts inside the joined `text`

  pages.forEach((pageText, p) => {
    const page = p + 1;
    const raw = String(pageText || "").replace(/\r\n?/g, "\n").trim();

    let cursor = 0;
    for (const piece of raw.split("\n")) {
      const lineStart = cursor;
      cursor += piece.length + 1; // the "\n" that split() consumed

      // Blank lines are dropped from the line index but preserved in `text`
      // below, so paragraph structure survives for the highlight layer.
      if (piece.trim() === "") continue;

      // `start`/`end` are offsets into `text`, not into the raw page. Without
      // them a character span cannot be resolved back to a line: `text` keeps
      // the blank lines that `lines` drops, so the two indexes drift apart.
      // Everything downstream — geometry, highlighting — needs this mapping to
      // be exact rather than reconstructed.
      const lead = piece.length - piece.trimStart().length;
      const trimmed = piece.trim();
      lines.push({
        id: `L${index + 1}`,
        page,
        index,
        text: trimmed,
        start: base + lineStart + lead,
        end: base + lineStart + lead + trimmed.length,
        confidence: lineConfidence(piece),
      });
      index++;
    }

    chunks.push(raw);
    base += raw.length + 2; // the "\n\n" joined between pages
  });

  const text = chunks.join("\n\n");

  return { lines, pageCount: pages.length, text, quality: assessQuality(lines) };
}

/**
 * Heuristic per-line legibility in 0..1. Deliberately simple and explainable:
 * the point is to detect pages the vision model struggled with, not to be a
 * calibrated probability.
 */
export function lineConfidence(line) {
  const s = String(line || "");
  const trimmed = s.trim();
  if (!trimmed) return 0;

  let score = 0.95;
  if (ILLEGIBLE.test(trimmed)) score -= 0.45;

  const letters = (trimmed.match(/[A-Za-z0-9]/g) || []).length;
  const ratio = letters / trimmed.length;
  if (ratio < 0.5) score -= 0.25; // mostly symbols — likely a bad read
  if (trimmed.length < 4) score -= 0.15;

  // Long runs without vowels rarely survive a clean transcription.
  if (/[bcdfghjklmnpqrstvwxz]{6,}/i.test(trimmed)) score -= 0.15;

  return Math.max(0.05, Math.min(1, score));
}

/** @returns {import("./types.js").OCRQuality} */
export function assessQuality(lines) {
  const flags = [];
  if (lines.length === 0) {
    return { score: 0, illegibleRatio: 1, shortLineRatio: 1, flags: ["no_text_extracted"] };
  }

  const illegible = lines.filter((l) => ILLEGIBLE.test(l.text)).length;
  const short = lines.filter((l) => normalizeText(l.text).length < 12).length;
  const illegibleRatio = illegible / lines.length;
  const shortLineRatio = short / lines.length;

  const mean = lines.reduce((s, l) => s + (l.confidence ?? 0.9), 0) / lines.length;
  let score = mean;
  if (illegibleRatio > 0.1) score -= 0.15;
  if (shortLineRatio > 0.6) score -= 0.1;
  score = Math.max(0, Math.min(1, score));

  if (illegibleRatio > 0.1) flags.push("illegible_regions");
  if (shortLineRatio > 0.6) flags.push("fragmented_lines");
  if (score < 0.6) flags.push("low_ocr_quality");
  if (lines.length < 3) flags.push("very_short_document");

  return { score, illegibleRatio, shortLineRatio, flags };
}

/**
 * The lines a character span of `doc.text` touches, in reading order.
 *
 * Relies on the `start`/`end` offsets recorded by structureOcr, so the answer is
 * exact. This is the hop from "the model quoted this phrase" to "these are the
 * physical lines that carry it" — and from there, via their bboxes, to pixels.
 */
export function linesForSpan(lines, start, end) {
  return lines
    .filter((l) => Number.isFinite(l.start) && start < l.end && end > l.start)
    .sort((a, b) => a.index - b.index);
}

/** Reconstruct text for a set of line ids, preserving reading order. */
export function linesToText(lines, ids) {
  const wanted = new Set(ids);
  return lines
    .filter((l) => wanted.has(l.id))
    .sort((a, b) => a.index - b.index)
    .map((l) => l.text)
    .join("\n");
}

/**
 * Map a character span inside an answer's reconstructed text back to line ids.
 * This is what lets an annotation say which physical line produced the mark.
 */
export function spanToLineIds(lines, ids, start, end) {
  const ordered = ids
    .map((id) => lines.find((l) => l.id === id))
    .filter(Boolean)
    .sort((a, b) => a.index - b.index);

  const hits = [];
  let cursor = 0;
  for (const line of ordered) {
    const lineStart = cursor;
    const lineEnd = cursor + line.text.length;
    if (start < lineEnd && end > lineStart) hits.push(line.id);
    cursor = lineEnd + 1; // the "\n" joined in linesToText
  }
  return hits;
}
