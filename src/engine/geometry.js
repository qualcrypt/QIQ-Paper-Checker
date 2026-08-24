/**
 * Where a line of handwriting physically sits on the page.
 *
 * The vision model returns text with no geometry (see the note at the top of
 * ocr.js), so the coordinates that let a mark land on the student's actual
 * handwriting have to be measured from the image itself. This does that
 * deterministically — no model call, no dependency, no invented numbers, which
 * keeps faith with the rule in types.js that a model-produced number is never
 * trusted as fact.
 *
 * The method is horizontal ink projection: binarise the page, count dark pixels
 * per row, and read the peaks as lines of writing. It is decades old, and it is
 * the right tool here precisely because it is boring — it fails predictably, and
 * we can measure how far to trust it.
 *
 * Known limits, surfaced rather than hidden (see alignBandsToLines):
 *   - assumes near-horizontal text; a tilted phone photo smears the projection
 *   - lines whose ascenders and descenders touch merge into one band
 *   - multi-column layouts read as single lines spanning both columns
 *
 * All coordinates are fractions of page width/height, so they survive the JPEG
 * re-encode in encodeCanvas() and any CSS scaling of the displayed page.
 */

/* Ignore a thin frame around the page: scanner shadow and dark photo edges are
   otherwise read as ink on every row, which flattens the whole profile. */
const EDGE_MARGIN = 0.015;

/* A band thinner than this fraction of the page is a speck, a ruled line or a
   staple mark rather than a line of writing. */
const MIN_BAND_HEIGHT = 0.006;

/* Gaps smaller than this fraction of the median band height are within-line —
   the dot of an "i", or a descender split off from its own line. */
const MERGE_GAP_RATIO = 0.3;

/* Analysis resolution. Line bands are coarse features; full resolution costs
   time and finds nothing extra. */
const ANALYSIS_EDGE = 1000;

/** Draw a data URL into a canvas, downscaled for analysis. */
function loadToCanvas(dataUrl, maxEdge = ANALYSIS_EDGE) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) return reject(new Error("The page image had no dimensions."));

      const scale = Math.min(1, maxEdge / Math.max(w, h));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(w * scale));
      canvas.height = Math.max(1, Math.round(h * scale));

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      resolve({ canvas, ctx, naturalWidth: w, naturalHeight: h });
    };
    img.onerror = () => reject(new Error("The page image could not be decoded for analysis."));
    img.src = dataUrl;
  });
}

/**
 * Otsu's method: the grey level that best separates the histogram into two
 * classes. Used instead of a fixed cutoff so that a dim phone photo and a bright
 * flatbed scan both binarise sensibly.
 *
 * @param {Uint32Array} hist  256 bins.
 * @param {number} total      Pixels counted into the histogram.
 */
export function otsuThreshold(hist, total) {
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];

  let sumB = 0;
  let wB = 0;
  let best = 0;
  let bestVar = -1;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > bestVar) {
      bestVar = between;
      best = t;
    }
  }
  return best;
}

/**
 * Ink pixels per row, plus a per-pixel ink mask, over the interior of the page.
 * Takes raw RGBA rather than a canvas so the analysis can be exercised without
 * a browser.
 *
 * @param {Uint8ClampedArray|Uint8Array} data RGBA, w*h*4
 */
function inkMap(data, w, h) {
  const x0 = Math.floor(w * EDGE_MARGIN);
  const x1 = Math.ceil(w * (1 - EDGE_MARGIN));
  const y0 = Math.floor(h * EDGE_MARGIN);
  const y1 = Math.ceil(h * (1 - EDGE_MARGIN));

  const gray = new Uint8Array(w * h);
  const hist = new Uint32Array(256);
  let counted = 0;

  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const p = (y * w + x) * 4;
      // Rec. 601 luma — cheap, and adequate for pen on paper.
      const g = (data[p] * 299 + data[p + 1] * 587 + data[p + 2] * 114) / 1000;
      const v = g < 0 ? 0 : g > 255 ? 255 : g | 0;
      gray[y * w + x] = v;
      hist[v]++;
      counted++;
    }
  }

  const threshold = otsuThreshold(hist, counted);
  const rows = new Uint32Array(h);
  const ink = new Uint8Array(w * h);

  for (let y = y0; y < y1; y++) {
    let n = 0;
    for (let x = x0; x < x1; x++) {
      if (gray[y * w + x] <= threshold) {
        ink[y * w + x] = 1;
        n++;
      }
    }
    rows[y] = n;
  }

  return { rows, ink, w, h, x0, x1, y0, y1, threshold };
}

/** Median of a numeric list. Returns 0 for an empty one. */
function median(values) {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
}

/**
 * Find the horizontal bands of writing on a page.
 *
 * @param {string} dataUrl
 * @returns {Promise<{width: number, height: number, threshold: number,
 *   bands: {top: number, bottom: number, left: number, right: number, ink: number}[]}>}
 *   Band coordinates are fractions of the page.
 */
export async function detectLineBands(dataUrl) {
  const { ctx, canvas, naturalWidth, naturalHeight } = await loadToCanvas(dataUrl);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { threshold, bands } = bandsFromPixels(data, canvas.width, canvas.height);
  return { width: naturalWidth, height: naturalHeight, threshold, bands };
}

/**
 * The measurement itself, on raw pixels. Separated from canvas loading so the
 * algorithm can be tested against synthetic pages with known line positions.
 *
 * @param {Uint8ClampedArray|Uint8Array} data RGBA, w*h*4
 * @returns {{threshold: number, bands: {top,bottom,left,right,ink}[]}}
 */
export function bandsFromPixels(data, w, h) {
  const map = inkMap(data, w, h);

  /* A page-relative floor: a row needs more than a trace of ink to count as part
     of a line. Taken from the median inked row, so a sparse page and a dense one
     are treated alike. */
  const inked = [];
  for (let y = map.y0; y < map.y1; y++) if (map.rows[y] > 0) inked.push(map.rows[y]);
  const rowFloor = Math.max(2, median(inked) * 0.3);

  // Contiguous runs of inked rows.
  const runs = [];
  let start = -1;
  for (let y = map.y0; y < map.y1; y++) {
    const on = map.rows[y] >= rowFloor;
    if (on && start === -1) start = y;
    else if (!on && start !== -1) {
      runs.push([start, y - 1]);
      start = -1;
    }
  }
  if (start !== -1) runs.push([start, map.y1 - 1]);

  // Merge runs separated only by a within-line gap.
  const medianHeight = median(runs.map(([a, b]) => b - a + 1)) || Math.max(1, h * 0.02);
  const maxGap = Math.max(1, medianHeight * MERGE_GAP_RATIO);

  const merged = [];
  for (const run of runs) {
    const last = merged[merged.length - 1];
    if (last && run[0] - last[1] - 1 <= maxGap) last[1] = run[1];
    else merged.push([run[0], run[1]]);
  }

  const minHeight = Math.max(1, h * MIN_BAND_HEIGHT);
  const bands = [];

  for (const [top, bottom] of merged) {
    if (bottom - top + 1 < minHeight) continue;

    // Horizontal extent of the writing inside this band.
    let left = map.x1;
    let right = map.x0;
    let total = 0;

    for (let y = top; y <= bottom; y++) {
      for (let x = map.x0; x < map.x1; x++) {
        if (!map.ink[y * w + x]) continue;
        if (x < left) left = x;
        if (x > right) right = x;
        total++;
      }
    }
    if (right < left) continue; // no ink survived

    bands.push({
      top: top / h,
      bottom: (bottom + 1) / h,
      left: left / w,
      right: (right + 1) / w,
      ink: total,
    });
  }

  return { threshold: map.threshold, bands };
}

/**
 * Attach a band to each OCR line of a page.
 *
 * Both sequences are in reading order, so an exact count match is a direct
 * index-to-index mapping and needs no cleverness. When the counts disagree — OCR
 * merged two lines, or a band split — the mapping is stretched proportionally,
 * which keeps it monotonic and roughly right rather than confidently wrong. The
 * disagreement comes back as `confidence` so the UI can admit when a placement
 * is a guess.
 *
 * @param {{top: number, bottom: number, left: number, right: number}[]} bands
 * @param {{id: string}[]} lines  The page's OCR lines, in reading order.
 * @returns {{confidence: "high"|"medium"|"low"|"none",
 *            boxes: Object.<string, {x: number, y: number, width: number, height: number}>}}
 */
export function alignBandsToLines(bands, lines) {
  const boxes = {};
  const B = bands.length;
  const L = lines.length;

  if (B === 0 || L === 0) return { confidence: "none", boxes };

  const drift = Math.abs(B - L) / Math.max(B, L);
  const confidence = B === L ? "high" : drift <= 0.2 ? "medium" : "low";

  lines.forEach((line, i) => {
    const band = B === L ? bands[i] : bands[Math.min(B - 1, Math.floor((i * B) / L))];
    boxes[line.id] = {
      x: band.left,
      y: band.top,
      width: Math.max(0, band.right - band.left),
      height: Math.max(0, band.bottom - band.top),
    };
  });

  return { confidence, boxes };
}

/** Smallest box containing all of them — spans an annotation across lines. */
export function unionBoxes(boxes) {
  const list = boxes.filter(Boolean);
  if (list.length === 0) return null;

  const x = Math.min(...list.map((b) => b.x));
  const y = Math.min(...list.map((b) => b.y));
  const right = Math.max(...list.map((b) => b.x + b.width));
  const bottom = Math.max(...list.map((b) => b.y + b.height));

  return { x, y, width: right - x, height: bottom - y };
}
