/**
 * PDF handling.
 *
 * pdf.js is already pulled in at runtime to rasterise answer sheets. Reference
 * material needs the other half of it: most textbooks, notes and syllabus
 * documents are digital PDFs with a real text layer, so their text can be read
 * directly. That is free, exact, and avoids spending vision tokens transcribing
 * something that was never handwritten.
 *
 * A scanned reference PDF has no text layer. That is detected and reported
 * rather than silently producing an empty index.
 */

const PDFJS_SRC = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

let pdfjsPromise = null;

/* Loaded from a CDN via a script tag rather than bundled, so the app keeps its
   "no runtime dependencies" property. */
export function loadPdfJs() {
  if (typeof window !== "undefined" && window.pdfjsLib) return Promise.resolve(window.pdfjsLib);
  if (!pdfjsPromise) {
    pdfjsPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PDFJS_SRC;
      s.onload = () => {
        if (!window.pdfjsLib) return reject(new Error("The PDF reader failed to initialise."));
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
        resolve(window.pdfjsLib);
      };
      s.onerror = () => {
        pdfjsPromise = null;
        reject(new Error("Could not load the PDF reader from the CDN. Upload page images instead."));
      };
      document.head.appendChild(s);
    });
  }
  return pdfjsPromise;
}

/**
 * Read the embedded text layer of a PDF, one entry per page.
 *
 * Items are joined with awareness of pdf.js's `hasEOL` flag so paragraphs keep
 * their line structure — the chunker splits on sentences, and run-together text
 * hides the boundaries it needs.
 *
 * @param {File} file
 * @returns {Promise<{pages: string[], hasText: boolean, pageCount: number}>}
 */
export async function extractPdfText(file) {
  const lib = await loadPdfJs();
  const data = await file.arrayBuffer();
  const doc = await lib.getDocument({ data }).promise;

  const pages = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();

    let out = "";
    for (const item of content.items) {
      if (typeof item.str !== "string") continue;
      out += item.str;
      if (item.hasEOL) out += "\n";
      else if (!item.str.endsWith(" ")) out += " ";
    }
    pages.push(out.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim());
  }

  const joined = pages.join("").trim();
  return { pages, hasText: joined.length > 40, pageCount: doc.numPages };
}
