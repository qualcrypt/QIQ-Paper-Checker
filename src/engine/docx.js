/**
 * Word document handling.
 *
 * A .docx is a zip of XML; mammoth reads it in the browser and returns the raw
 * text. Like pdf.js it is pulled from a CDN at runtime rather than bundled, so
 * the app keeps its "no runtime dependencies" property.
 *
 * A digital document already carries its text, so extraction is free, exact,
 * and avoids spending vision tokens transcribing something that was never
 * handwritten — the same reasoning as the reference-PDF text layer.
 *
 * The legacy binary .doc format predates the Office Open XML standard and
 * mammoth cannot read it. That case is detected by the caller and reported,
 * rather than silently producing garbage.
 */

const MAMMOTH_SRC = "https://cdn.jsdelivr.net/npm/mammoth@1.9.0/mammoth.browser.min.js";

let mammothPromise = null;

export function loadMammoth() {
  if (typeof window !== "undefined" && window.mammoth) return Promise.resolve(window.mammoth);
  if (!mammothPromise) {
    mammothPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = MAMMOTH_SRC;
      s.onload = () => {
        if (!window.mammoth) return reject(new Error("The Word reader failed to initialise."));
        resolve(window.mammoth);
      };
      s.onerror = () => {
        mammothPromise = null;
        reject(new Error("Could not load the Word reader from the CDN. Upload a PDF or images instead."));
      };
      document.head.appendChild(s);
    });
  }
  return mammothPromise;
}

/** True for .docx files, by extension or MIME type. */
export function isDocx(file) {
  return (
    /\.docx$/i.test(file.name) ||
    file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  );
}

/** True for the legacy binary Word format, which mammoth cannot parse. */
export function isLegacyDoc(file) {
  return /\.doc$/i.test(file.name) || file.type === "application/msword";
}

export const LEGACY_DOC_MESSAGE = (name) =>
  `"${name}" is the old Word format (.doc). Open it in Word, save it as .docx, and upload again — ` +
  `or export it as a PDF.`;

/**
 * Read the text of a .docx file, preserving paragraph breaks.
 *
 * @param {File} file
 * @returns {Promise<string>}
 */
export async function extractDocxText(file) {
  const mammoth = await loadMammoth();
  const arrayBuffer = await file.arrayBuffer();
  const { value } = await mammoth.extractRawText({ arrayBuffer });
  const text = String(value || "").replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (text.length < 2) {
    throw new Error(`No readable text found in "${file.name}". If it is a scan, upload a PDF or photos instead.`);
  }
  return text;
}
