/**
 * Ink-projection accuracy against synthetic pages whose true line positions are
 * known. Paints RGBA directly — no canvas, no browser.
 */
import { bandsFromPixels, alignBandsToLines } from "../src/engine/geometry.js";

const W = 1000;
const H = 1400;

function blankPage({ paper = 245, noise = 0 } = {}) {
  const d = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const n = noise ? (Math.random() - 0.5) * noise : 0;
    const v = Math.max(0, Math.min(255, paper + n));
    d[i * 4] = v; d[i * 4 + 1] = v; d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
  }
  return d;
}

const px = (d, x, y, v) => {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const p = (y * W + x) * 4;
  d[p] = v; d[p + 1] = v; d[p + 2] = v;
};

/**
 * Paint one line of pseudo-handwriting: a dense x-height core, plus sparse
 * ascenders and descenders — the shape that makes real line separation hard.
 */
function writeLine(d, { yBase, xStart, xEnd, core = 22, ink = 40, skew = 0, density = 0.55 }) {
  // Ascenders and descenders scale with the writing, as real handwriting does.
  const asc = Math.round(core * 0.55);
  const desc = Math.round(core * 0.36);
  const top = yBase - core - asc;
  const bottom = yBase + desc;
  for (let x = xStart; x < xEnd; x++) {
    const dy = Math.round(skew * (x - xStart));
    // x-height core: most ink lives here
    for (let y = yBase - core; y <= yBase; y++) {
      if (Math.random() < density) px(d, x, y + dy, ink);
    }
    // ascenders / descenders: occasional tall strokes
    if (Math.random() < 0.13) {
      for (let y = yBase - core - asc; y < yBase - core; y++) px(d, x, y + dy, ink);
    }
    if (Math.random() < 0.09) {
      for (let y = yBase; y <= yBase + desc; y++) px(d, x, y + dy, ink);
    }
  }
  return { top: top / H, bottom: (bottom + 1) / H };
}

function makePage({ n = 8, gap = 62, yStart = 150, core = 22, ink = 40, paper = 245,
                    noise = 0, skew = 0, border = false, density = 0.55, xEnd = 880 } = {}) {
  const d = blankPage({ paper, noise });
  if (border) {
    for (let x = 0; x < W; x++) for (let y = 0; y < 6; y++) { px(d, x, y, 30); px(d, x, H - 1 - y, 30); }
    for (let y = 0; y < H; y++) for (let x = 0; x < 6; x++) { px(d, x, y, 30); px(d, W - 1 - x, y, 30); }
  }
  const truth = [];
  for (let i = 0; i < n; i++) {
    truth.push(writeLine(d, { yBase: yStart + i * gap, xStart: 90, xEnd, core, ink, skew, density }));
  }
  return { data: d, truth };
}

/** Fraction of detected bands that overlap their true line by >40% IoU. */
function score(bands, truth) {
  if (bands.length !== truth.length) return null;
  let ok = 0;
  for (let i = 0; i < truth.length; i++) {
    const b = bands[i], t = truth[i];
    const inter = Math.min(b.bottom, t.bottom) - Math.max(b.top, t.top);
    const union = Math.max(b.bottom, t.bottom) - Math.min(b.top, t.top);
    if (inter > 0 && inter / union > 0.4) ok++;
  }
  return ok / truth.length;
}

const cases = [
  ["clean scan, 8 lines",       {}],
  ["clean scan, 20 lines",      { n: 20, gap: 62, yStart: 120 }],
  ["tight line spacing",        { gap: 42 }],
  ["very tight (touching)",     { gap: 34 }],
  ["wide line spacing",         { gap: 100 }],
  ["small writing (12 lines)",  { core: 13, gap: 38, yStart: 100, n: 12 }],
  ["large writing (6 lines)",   { core: 34, gap: 92, yStart: 160, n: 6 }],
  ["faint pencil",              { ink: 150 }],
  ["grey paper",                { paper: 205, ink: 70 }],
  ["scan noise",                { noise: 45 }],
  ["dark scan border",          { border: true }],
  ["sparse strokes",            { density: 0.3 }],
  ["short lines (indented)",    { xEnd: 380 }],
  ["two lines only",            { n: 2 }],
  ["single line",               { n: 1 }],
  ["slight skew (~0.6deg)",     { skew: 0.011 }],
  ["moderate skew (~1.7deg)",   { skew: 0.030 }],
  ["heavy skew (~4deg)",        { skew: 0.070 }],
];

console.log("case".padEnd(26) + "bands/true  align    line-overlap");
console.log("-".repeat(66));

let exact = 0;
for (const [label, opts] of cases) {
  const { data, truth } = makePage(opts);
  const { bands } = bandsFromPixels(data, W, H);
  const acc = score(bands, truth);
  const align = alignBandsToLines(bands, truth.map((_, i) => ({ id: "L" + i })));
  if (bands.length === truth.length) exact++;
  console.log(
    label.padEnd(26) +
      (bands.length + "/" + truth.length).padEnd(12) +
      align.confidence.padEnd(9) +
      (acc === null ? "—" : (acc * 100).toFixed(0) + "%")
  );
}
console.log("-".repeat(66));
console.log(`exact line-count match: ${exact}/${cases.length}`);
