import { useState, useEffect } from "react";

import { structureOcr } from "./src/engine/ocr.js";
import { groqChat, GroqError, OCR_MODEL, EVAL_MODEL } from "./src/engine/groq.js";
import { createLlm } from "./src/engine/llm.js";
import { loadPdfJs, extractPdfText } from "./src/engine/pdf.js";
import { extractExamWithLlm, validateExam, structuralMarks, deriveMarks } from "./src/engine/exam.js";
import { chunkDocument, createRetriever } from "./src/engine/reference.js";
import { matchAnswers } from "./src/engine/match.js";
import { assessPaper, summarisePaper, toEvaluation, GROUNDING } from "./src/engine/assess.js";
import { extractJson } from "./src/engine/json.js";
import {
  anchorAnnotations,
  measurePages,
  attachGeometry,
  resolveAnnotationBoxes,
  overallGeometry,
} from "./src/engine/marking.js";

/* ============================================================================
   QIQ — Descriptive Paper Checker   (Groq edition)
   Single-file React demo. No dependencies beyond React.

   Pipeline:  Upload → Rasterise → OCR (vision) → Evaluate (reasoning) → Results

   Models (verified live against this account's /v1/models):
     OCR    qwen/qwen3.6-27b    — multimodal, reasoning turned OFF for speed
     Grade  openai/gpt-oss-120b — reasoning_effort:"medium" + JSON mode
   ========================================================================== */

/* Token budgets are sized against a measured free-tier limit of 8000 tokens per
   minute, shared across both models. Groq counts prompt + max_completion_tokens
   toward that ceiling *up front*, so an oversized budget is rejected outright
   (413) even when the model would never have used it. Measured costs:
     a full page image  ≈ 1870 prompt tokens (capped — 1400px and 2000px cost
                          the same, so we keep the higher quality render)
     evaluation prompt  ≈ 1000-2000 tokens, and answers at "medium" effort came
                          back in ~2300 completion tokens.
   Raise these if you move to a paid tier. */
/* Measured on a real 9-page handwritten matrices paper: a sparse page of MCQ
   answers returned 122 completion tokens, the densest page of algebra 353. The
   old budget of 3000 reserved roughly 8x what any page actually used — and Groq
   charges the reservation against the per-minute ceiling whether or not it is
   spent, which is what turned a 9-page upload into an instant 413. This leaves
   ~3x headroom over the worst page seen, and runOcr retries with the old budget
   if a page ever does run out. */
const OCR_MAX_TOKENS = 1100;
const OCR_MAX_TOKENS_RETRY = 3000;
const EVAL_MAX_TOKENS = 5000;

/* "high" spends the entire completion budget on reasoning and returns nothing
   at these limits. "medium" produced the richest marking in testing — full
   margin notes plus explicit "missing" points — so it is the teacher dial. */
const EVAL_EFFORT = "medium";

/* Rasterising limits. Groq caps base64 images at ~4 MB, and a 12 MP phone photo
   of a page is wasted tokens anyway, so every page is normalised to JPEG. */
const MAX_EDGE = 2000;
const JPEG_QUALITY = 0.85;
const MAX_B64_CHARS = 3500000;
const MAX_PAGES = 12;

/* ------------------------------------------------------------ presentation */
const REVEAL_MS = 300; // gap between each highlight appearing
const TYPE_MS = 18; // per character of the teacher's remark
const COUNT_MS = 1500; // marks counter run time
const LOW_CONFIDENCE = 60; // below this the mark is flagged for manual review
const PASS_THRESHOLD = 0.4; // 40% and above earns the PASS stamp

const HISTORY_KEY = "qiq.paperchecker.history.v1";
const HISTORY_LIMIT = 5;

const OCR_SYSTEM =
  "You are an OCR engine. Extract ALL text from this student answer paper exactly as written, " +
  "including handwriting. Preserve paragraph structure. Do not correct spelling, do not summarise, " +
  "do not add commentary. Return only the extracted text.";

const EVAL_SYSTEM =
  "You are an experienced school/college teacher and examiner with 20 years of experience. " +
  "You evaluate student answer papers exactly like a human teacher — carefully, fairly, with " +
  "partial marking, inline annotations, and a final verdict. You reply with JSON only.";

const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"];

/* Filled into the marking scheme box by the "Use sample" button. Kept separate
   from the placeholder on purpose: a placeholder that looks like a completed
   form reads as real input, and the paper then fails validation for no visible
   reason. The placeholder describes what to write; this is what to write. */
const SAMPLE_SCHEME = `Q. Explain the process of photosynthesis. (20 marks)

1. Definition of photosynthesis — 3 marks
2. Balanced chemical equation — 4 marks
3. Role of chlorophyll and sunlight — 5 marks
4. Light vs dark reaction, with the correct site of each — 5 marks
5. Two real-world significances — 3 marks`;

/* ---------------------------------------------------------------- palette -- */
const C = {
  navy: "#0A0F1E",
  card: "#0F172A",
  border: "#1E293B",
  borderSoft: "#243247",
  blue: "#2563EB",
  purple: "#7C3AED",
  text: "#E2E8F0",
  dim: "#94A3B8",
  faint: "#64748B",
  green: "#22C55E",
  amber: "#F59E0B",
  red: "#EF4444",
};

const TYPE_STYLE = {
  correct: { label: "Correct", color: C.green, bg: "rgba(34,197,94,0.16)", icon: "✓", sign: "+" },
  partial: { label: "Partial", color: C.amber, bg: "rgba(245,158,11,0.16)", icon: "~", sign: "~" },
  wrong: { label: "Incorrect", color: C.red, bg: "rgba(239,68,68,0.16)", icon: "✗", sign: "✗" },
  missing: { label: "Missing", color: "#A855F7", bg: "rgba(168,85,247,0.16)", icon: "!", sign: "!" },
};
const typeStyle = (t) => TYPE_STYLE[t] || TYPE_STYLE.partial;

/** The floating badge text on a highlight: "+2", "~1", "✗0". */
function marksBadge(ann) {
  const n = Number(ann.marks);
  return typeStyle(ann.type).sign + (Number.isFinite(n) ? n : 0);
}

const isLowConfidence = (ann) => {
  const c = Number(ann && ann.confidence);
  return Number.isFinite(c) && c < LOW_CONFIDENCE;
};

const gradeColor = (g) => {
  const u = String(g || "").toUpperCase();
  if (u.startsWith("A")) return C.green;
  if (u.startsWith("B")) return "#3B82F6";
  if (u.startsWith("C")) return C.amber;
  return C.red;
};

const todayLabel = () =>
  new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });

/* The Groq client lives in src/engine/groq.js. */

/* ============================================================== HISTORY === */

/* localStorage throws in private-mode Safari and when storage is full, and a
   corrupt entry should never take the whole app down with it. */
function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

function saveHistory(list) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, HISTORY_LIMIT)));
  } catch {
    /* history is a convenience, never a hard failure */
  }
}

/* ====================================================== PAGE RASTERISING == */

function encodeCanvas(canvas) {
  let q = JPEG_QUALITY;
  let url = canvas.toDataURL("image/jpeg", q);
  while (url.length > MAX_B64_CHARS && q > 0.4) {
    q -= 0.15;
    url = canvas.toDataURL("image/jpeg", q);
  }
  return url;
}

function paintToCanvas(source, w, h) {
  const scale = Math.min(1, MAX_EDGE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(w * scale));
  canvas.height = Math.max(1, Math.round(h * scale));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#FFFFFF"; // flatten transparency; scans read better on white
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function rasterizeImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = paintToCanvas(img, img.naturalWidth, img.naturalHeight);
        resolve([
          {
            label: file.name,
            dataUrl: encodeCanvas(canvas),
            width: canvas.width,
            height: canvas.height,
          },
        ]);
      } catch (e) {
        reject(e);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not decode "${file.name}".`));
    };
    img.src = url;
  });
}

async function rasterizePdf(file) {
  const lib = await loadPdfJs();
  const data = await file.arrayBuffer();
  const doc = await lib.getDocument({ data }).promise;
  const count = Math.min(doc.numPages, MAX_PAGES);
  const pages = [];

  for (let i = 1; i <= count; i++) {
    const page = await doc.getPage(i);
    const base = page.getViewport({ scale: 1 });
    const scale = Math.min(2.5, MAX_EDGE / Math.max(base.width, base.height));
    const viewport = page.getViewport({ scale });

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    await page.render({ canvasContext: ctx, viewport }).promise;
    pages.push({
      label: `${file.name} · page ${i}`,
      dataUrl: encodeCanvas(canvas),
      width: canvas.width,
      height: canvas.height,
    });
  }
  return pages;
}

/* Annotation matching and page geometry live in src/engine/marking.js. */

/* ======================================================= ANIMATION HOOKS == */

/** Counts 0 → target on an eased curve. Returns the target immediately if the
 *  viewer prefers reduced motion. */
function useCountUp(target, duration, runKey) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    const goal = Number(target) || 0;
    if (prefersReducedMotion()) {
      setValue(goal);
      return undefined;
    }
    let frame;
    const started = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(goal * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
      else setValue(goal);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration, runKey]);

  return value;
}

/** Types `text` out one character at a time. `skip()` jumps to the end. */
function useTypewriter(text, msPerChar, runKey) {
  const [count, setCount] = useState(0);
  const full = String(text || "");

  useEffect(() => {
    if (prefersReducedMotion()) {
      setCount(full.length);
      return undefined;
    }
    setCount(0);
    if (!full) return undefined;
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setCount(i);
      if (i >= full.length) clearInterval(id);
    }, msPerChar);
    return () => clearInterval(id);
  }, [full, msPerChar, runKey]);

  return {
    shown: full.slice(0, count),
    done: count >= full.length,
    skip: () => setCount(full.length),
  };
}

function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/* ================================================================ SPEECH == */

const speechSupported = () => typeof window !== "undefined" && "speechSynthesis" in window;

/** Wraps the SpeechSynthesis singleton. It is global browser state, so the hook
 *  always cancels on unmount — otherwise the voice keeps talking after the tab
 *  contents have changed. */
function useSpeech() {
  const [state, setState] = useState("idle"); // idle | speaking | paused

  useEffect(() => {
    return () => {
      if (speechSupported()) window.speechSynthesis.cancel();
    };
  }, []);

  const speak = (text) => {
    if (!speechSupported() || !text) return;
    const synth = window.speechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(String(text));
    u.rate = 0.95;
    u.pitch = 1;
    u.onend = () => setState("idle");
    u.onerror = () => setState("idle");
    synth.speak(u);
    setState("speaking");
  };

  const pause = () => {
    if (!speechSupported()) return;
    window.speechSynthesis.pause();
    setState("paused");
  };

  const resume = () => {
    if (!speechSupported()) return;
    window.speechSynthesis.resume();
    setState("speaking");
  };

  const stop = () => {
    if (!speechSupported()) return;
    window.speechSynthesis.cancel();
    setState("idle");
  };

  return { state, speak, pause, resume, stop, supported: speechSupported() };
}

/* =============================================================== SUB-UI ==== */

/**
 * `step` is how far the paper has got; `running` says whether that step is
 * actually executing right now. Without the distinction, merely uploading a
 * file lights up "OCR" as if it were in progress, and any later error reads as
 * an OCR failure when nothing has been sent yet.
 */
function PipelineBar({ step, running }) {
  const steps = ["Upload", "OCR", "Evaluate", "Results"];
  return (
    <div className="qiq-pipeline">
      {steps.map((label, i) => {
        const done = i < step;
        const active = i === step && running;
        return (
          <div key={label} className="qiq-pipe-item">
            <div className={`qiq-pipe-dot${active ? " is-active" : ""}${done ? " is-done" : ""}`}>
              {done ? "✓" : i + 1}
            </div>
            <span
              style={{
                fontSize: 13,
                fontWeight: active || done ? 600 : 500,
                color: active ? C.text : done ? C.dim : C.faint,
                whiteSpace: "nowrap",
              }}
              title={active ? "in progress" : done ? "done" : "not started"}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="qiq-pipe-line" style={{ background: done ? C.blue : C.border }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function Legend() {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
      {["correct", "partial", "wrong", "missing"].map((t) => {
        const s = typeStyle(t);
        return (
          <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: C.dim }}>
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: 3,
                background: s.bg,
                border: `1px solid ${s.color}`,
                display: "inline-block",
              }}
            />
            {s.label}
          </span>
        );
      })}
    </div>
  );
}

function SectionTitle({ n, title, action }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "20px 0 10px" }}>
      <span className="qiq-step-num">{n}</span>
      <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.7, textTransform: "uppercase", color: C.dim }}>
        {title}
      </span>
      {action && <span style={{ marginLeft: "auto" }}>{action}</span>}
    </div>
  );
}

/** ⚠️ marker shown wherever the model told us it was unsure. */
function ConfidenceFlag({ confidence, compact }) {
  return (
    <span
      className={`qiq-warn${compact ? " is-compact" : ""}`}
      title={`AI is unsure — please verify this mark manually (confidence ${Math.round(
        Number(confidence) || 0
      )}%)`}
    >
      ⚠️
    </span>
  );
}

/* ============================================================ MAIN APP ===== */

export default function PaperChecker() {
  const [pages, setPages] = useState([]); // { id, label, dataUrl }
  const [dragging, setDragging] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [totalMarks, setTotalMarks] = useState("20");

  /* Report-card identity fields — kept at this level so the history log and the
     printed report agree, and so they survive tab switches. */
  const [studentName, setStudentName] = useState("");
  const [subject, setSubject] = useState("");
  const [reportDate, setReportDate] = useState(todayLabel);

  const [studentAnswerText, setStudentAnswerText] = useState("");
  const [editedText, setEditedText] = useState("");
  const [evaluation, setEvaluation] = useState(null);
  const [evalRun, setEvalRun] = useState(0); // bumps per result; drives replays
  const [rawResponse, setRawResponse] = useState("");
  const [reasoning, setReasoning] = useState("");

  /* Where the writing physically sits. `pageBands` is measured from the images
     once and cached; `ocrDoc` carries the lines those measurements attach to;
     `geometryByPage` records how much the placement can be trusted. Together
     they are what lets a mark land on the handwriting rather than on a
     transcript of it. */
  const [pageBands, setPageBands] = useState({});
  const [ocrDoc, setOcrDoc] = useState(null);
  const [geometryByPage, setGeometryByPage] = useState({});

  /* The question paper drives everything: it says what was asked, how many
     questions there are and what each is worth. When one is supplied the
     question-paper-first pipeline runs; without one the original
     marking-scheme flow is used unchanged, so nothing that worked stops
     working. */
  const [examPages, setExamPages] = useState([]);
  const [exam, setExam] = useState(null);
  const [examBusy, setExamBusy] = useState(false);

  /* Reference material, chunked and indexed locally. */
  const [refFiles, setRefFiles] = useState([]); // { name, chunks, pageCount }
  const [refChunks, setRefChunks] = useState([]);
  const [refBusy, setRefBusy] = useState(false);

  const [markProgress, setMarkProgress] = useState({ done: 0, total: 0, label: "" });

  const [stage, setStage] = useState("idle"); // idle | ocr | evaluating | done
  const [ocrProgress, setOcrProgress] = useState({ done: 0, total: 0 });
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [invalid, setInvalid] = useState(""); // "" | "pages" | "scheme" | "marks"
  const [tab, setTab] = useState("paper"); // the marked-up page leads
  const [activeAnn, setActiveAnn] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const [revealed, setRevealed] = useState(0); // how many annotations are visible
  const [history, setHistory] = useState(loadHistory);

  const busy = stage === "ocr" || stage === "evaluating";

  useEffect(() => {
    if (!busy) return undefined;
    setElapsed(0);
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [busy]);

  const pipelineStep =
    stage === "ocr" ? 1 : stage === "evaluating" ? 2 : stage === "done" ? 3 : pages.length ? 1 : 0;

  /* --------------------------------------------------------------- derive -- */
  const annotations = evaluation && Array.isArray(evaluation.annotations) ? evaluation.annotations : [];
  const keyPoints = evaluation && Array.isArray(evaluation.keyPoints) ? evaluation.keyPoints : [];

  const scoreTotal = Number(evaluation && evaluation.totalMarks) || Number(totalMarks) || 0;
  const scoreAwarded = !evaluation
    ? 0
    : Number.isFinite(Number(evaluation.totalMarksAwarded))
    ? Number(evaluation.totalMarksAwarded)
    : keyPoints.reduce((sum, k) => sum + (Number(k.marksAwarded) || 0), 0);

  const anchored = evaluation ? anchorAnnotations(studentAnswerText, annotations) : null;
  const lowConfidenceCount = annotations.filter(isLowConfidence).length;

  /* Annotation index → boxes on the page images. Empty when the pages could not
     be measured, which is what makes the text view the honest fallback. */
  const annBoxes = anchored ? resolveAnnotationBoxes(ocrDoc, anchored.placed) : {};
  const placedOnPage = Object.keys(annBoxes).length;
  const geometryLevel = overallGeometry(geometryByPage);

  /* --------------------------------------------------- staggered reveal --- */
  /* The marking is dealt out one annotation at a time so the teacher's eye can
     follow it, the way a person marks a page rather than a diff appearing. */
  useEffect(() => {
    if (!evaluation || annotations.length === 0) {
      setRevealed(0);
      return undefined;
    }
    if (prefersReducedMotion()) {
      setRevealed(annotations.length);
      return undefined;
    }
    setRevealed(0);
    let n = 0;
    const id = setInterval(() => {
      n += 1;
      setRevealed(n);
      if (n >= annotations.length) clearInterval(id);
    }, REVEAL_MS);
    return () => clearInterval(id);
  }, [evalRun, annotations.length, evaluation]);

  const revealAll = () => setRevealed(annotations.length);
  const markingInProgress = evaluation && revealed < annotations.length;

  /* ------------------------------------------------------------- uploads -- */
  async function addFiles(list) {
    const incoming = Array.from(list || []);
    if (!incoming.length) return;

    setError("");
    setPreparing(true);
    const collected = [];

    for (const file of incoming) {
      const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
      const isImage = ACCEPTED_IMAGE_TYPES.includes(file.type);
      if (!isPdf && !isImage) {
        setError(`"${file.name}" is not supported. Upload a PDF or a JPG / PNG / GIF / WEBP image.`);
        continue;
      }
      try {
        const produced = isPdf ? await rasterizePdf(file) : await rasterizeImage(file);
        collected.push(...produced);
      } catch (e) {
        setError(e.message || `Could not read "${file.name}".`);
      }
    }

    if (collected.length) {
      const stamp = String(Date.now());
      setPages((prev) => {
        const merged = prev.concat(collected.map((p, i) => ({ ...p, id: `${stamp}-${i}` })));
        if (merged.length > MAX_PAGES) {
          setError(`Only the first ${MAX_PAGES} pages are kept.`);
          return merged.slice(0, MAX_PAGES);
        }
        return merged;
      });
    }
    setPreparing(false);
  }

  const removePage = (id) => setPages((prev) => prev.filter((p) => p.id !== id));

  function resetAll() {
    setPages([]);
    setStudentAnswerText("");
    setEditedText("");
    setEvaluation(null);
    setRawResponse("");
    setReasoning("");
    setStage("idle");
    setError("");
    setNotice("");
    setActiveAnn(null);
    setRevealed(0);
    setOcrProgress({ done: 0, total: 0 });
    setPageBands({});
    setOcrDoc(null);
    setGeometryByPage({});
    setMarkProgress({ done: 0, total: 0, label: "" });
    setTab("paper");
  }

  /* ------------------------------------------------------------ pipeline -- */

  /* ------------------------------------------------- question paper ------ */

  /** Read the question paper and turn it into the exam structure. */
  async function ingestExam(files) {
    setError("");
    setExamBusy(true);
    try {
      const collected = [];
      for (const file of Array.from(files || [])) {
        const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
        const isImage = ACCEPTED_IMAGE_TYPES.includes(file.type);
        if (!isPdf && !isImage) {
          setError(`"${file.name}" is not a supported question paper. Upload a PDF or an image.`);
          continue;
        }
        collected.push(...(isPdf ? await rasterizePdf(file) : await rasterizeImage(file)));
      }
      if (collected.length === 0) return;

      const withIds = collected.map((p, i) => ({ ...p, id: `qp-${Date.now()}-${i}` }));
      setExamPages(withIds);

      const llm = createLlm({ onRetry });

      /* A digital question paper already carries its text; only scans need the
         vision pass. Either way the model reads the structure, and segment.js
         re-reads the printed marks independently as a cross-check. */
      const pageTexts = [];
      for (let i = 0; i < withIds.length; i++) {
        setNotice(`Reading question paper page ${i + 1} of ${withIds.length}…`);
        pageTexts.push(
          await llm.callText({
            stage: "question paper OCR",
            system: OCR_SYSTEM,
            user: "Extract all text from this examination question paper, preserving question numbers and printed marks.",
            maxTokens: OCR_MAX_TOKENS,
            model: OCR_MODEL,
            images: [withIds[i].dataUrl],
          })
        );
      }

      const doc = structureOcr(pageTexts);
      const raw = await extractExamWithLlm(doc.text, llm);
      const validated = validateExam(raw, { structural: structuralMarks(doc.lines) });

      setExam(validated);
      setNotice("");
      if (validated.blocking) {
        setError(
          "The question paper needs checking before grading can start — see the warnings below."
        );
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setExamBusy(false);
      setNotice("");
    }
  }

  /** Manual correction of a mark the paper did not print, or read wrongly. */
  function setQuestionMarks(id, value) {
    setExam((prev) => {
      if (!prev) return prev;
      const n = value === "" ? null : Number(value);
      const questions = prev.questions.map((q) =>
        q.id === id ? { ...q, maxMarks: Number.isFinite(n) && n > 0 ? n : null } : q
      );
      /* Recompute the mark-dependent warnings too, so the "enter the missing
         marks" notice disappears the moment they are entered instead of
         standing there contradicting the filled-in boxes. */
      return { ...prev, ...deriveMarks(questions, prev.declaredTotal, prev.baseWarnings) };
    });
  }

  function clearExam() {
    setExam(null);
    setExamPages([]);
  }

  /* -------------------------------------------------- reference PDFs ----- */

  async function ingestReferences(files) {
    setError("");
    setRefBusy(true);
    try {
      const added = [];
      const chunks = [];

      for (const file of Array.from(files || [])) {
        if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
          setError(`"${file.name}" is not a PDF. Reference material must be a PDF.`);
          continue;
        }
        const { pages, hasText, pageCount } = await extractPdfText(file);
        if (!hasText) {
          setError(
            `"${file.name}" has no text layer — it looks like a scan. Reference material needs a ` +
              `text-based PDF so it can be searched.`
          );
          continue;
        }
        const produced = pages.flatMap((text, i) =>
          chunkDocument(text, { source: file.name, page: i + 1 })
        );
        chunks.push(...produced);
        added.push({ name: file.name, chunks: produced.length, pageCount });
      }

      if (added.length) {
        setRefFiles((prev) => prev.concat(added));
        setRefChunks((prev) => prev.concat(chunks));
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setRefBusy(false);
    }
  }

  function clearReferences() {
    setRefFiles([]);
    setRefChunks([]);
  }

  const onRetry = (seconds) =>
    setNotice(`Groq's per-minute token limit was reached — resuming in ${seconds}s…`);

  /**
   * How many API keys the proxy is scheduling over. Each one supports another
   * question in flight; the server keeps the key values to itself and reports
   * only the count.
   */
  async function poolSize() {
    try {
      const res = await fetch("/api/groq/stats");
      if (!res.ok) return 1;
      const { keys } = await res.json();
      return Math.max(1, Number(keys) || 1);
    } catch {
      return 1;
    }
  }

  /** One vision call per page — keeps every request small and gives live progress. */
  async function runOcr() {
    setOcrProgress({ done: 0, total: pages.length });
    const chunks = [];

    /* One page, at the lean budget; if it truly needs more, pay for more once. */
    const ocrPage = async (i, budget) => {
      const { text } = await groqChat(
        {
          model: OCR_MODEL,
          max_completion_tokens: budget,
          temperature: 0,
          reasoning_format: "hidden",
          reasoning_effort: "none",
          messages: [
            { role: "system", content: OCR_SYSTEM },
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text:
                    pages.length > 1
                      ? `Extract the student's answer from page ${i + 1} of ${pages.length}.`
                      : "Extract the student's answer from this answer paper.",
                },
                { type: "image_url", image_url: { url: pages[i].dataUrl } },
              ],
            },
          ],
        },
        onRetry
      );
      return text;
    };

    for (let i = 0; i < pages.length; i++) {
      let text;
      try {
        text = await ocrPage(i, OCR_MAX_TOKENS);
      } catch (e) {
        /* groqChat throws when the model stops on "length". A page dense enough
           to exhaust the lean budget is rare, so pay the large one just for it
           rather than reserving it for every page. */
        if (!/output limit/i.test(e.message)) throw e;
        setNotice(`Page ${i + 1} is unusually dense — re-reading with a larger budget…`);
        text = await ocrPage(i, OCR_MAX_TOKENS_RETRY);
      }
      setNotice("");
      // One entry per page, unjoined and unlabelled. structureOcr needs the page
      // boundaries to attribute lines, and a "[Page 2]" header would be text the
      // student never wrote — the examiner could quote it and then no highlight
      // could ever be placed on it.
      chunks.push(text);
      setOcrProgress({ done: i + 1, total: pages.length });
    }

    return chunks;
  }

  async function runEvaluation(answerText) {
    const marks = Number(totalMarks) || 0;
    const prompt = `MARKING SCHEME / EXPECTED ANSWER:
${expectedAnswer.trim()}

TOTAL MARKS: ${marks}

STUDENT'S ANSWER (extracted via OCR):
${answerText}

Evaluate this student answer exactly like a human teacher. Read it carefully, check which key points of the marking scheme are covered and which are missed, judge conceptual understanding as well as language and clarity, deduct marks for wrong or missing content, and award partial marks wherever they are deserved.

Important: every value of "text" in "annotations" MUST be an exact, verbatim substring copied character-for-character from the student's answer above, so it can be highlighted in place. Keep each one to a phrase or a single sentence. Use type "missing" only for expected content that is absent — for those, describe the missing idea in "text" and it will be shown as a margin note instead of a highlight.

For every annotation also give "confidence": an integer from 0 to 100 saying how sure you are that this mark is correct. Be honest and use the low end when the handwriting was ambiguous, the phrasing was unclear, or the marking scheme did not settle the case — anything you score below 60 is flagged for the teacher to check by hand.

Return ONLY valid JSON in this exact structure, with no commentary and no markdown fences:

{
  "annotations": [
    {
      "text": "exact phrase from student answer",
      "type": "correct" | "partial" | "wrong" | "missing",
      "comment": "teacher's inline comment",
      "marks": number,
      "confidence": number
    }
  ],
  "keyPoints": [
    {
      "point": "key point from marking scheme",
      "covered": true | false,
      "quality": "well" | "partially" | "not",
      "marksAwarded": number,
      "marksTotal": number,
      "teacherNote": "specific feedback"
    }
  ],
  "totalMarksAwarded": number,
  "totalMarks": ${marks},
  "grade": "A+" | "A" | "B" | "C" | "D" | "F",
  "overallRemark": "teacher's final comment written like a real teacher — encouraging but honest, 3-4 sentences",
  "thingsWellDone": ["list of positives"],
  "improvementAreas": ["short topic name per item, e.g. 'Sites of the light and dark reaction'"]
}`;

    let out;
    try {
      out = await groqChat(
        {
          model: EVAL_MODEL,
          max_completion_tokens: EVAL_MAX_TOKENS,
          temperature: 0.3,
          reasoning_effort: EVAL_EFFORT, // the "think like a teacher" dial
          response_format: { type: "json_object" },
          messages: [
            { role: "system", content: EVAL_SYSTEM },
            { role: "user", content: prompt },
          ],
        },
        onRetry
      );
      setNotice("");
    } catch (e) {
      // JSON mode can reject the generation but still hand back the raw attempt.
      if (e instanceof GroqError && e.failedGeneration) {
        out = { text: e.failedGeneration, reasoning: "" };
      } else {
        throw e;
      }
    }

    setRawResponse(out.text);
    setReasoning(out.reasoning);
    return extractJson(out.text);
  }

  /** Append this result to the rolling five-paper log. */
  function recordHistory(result) {
    const total = Number(result.totalMarks) || Number(totalMarks) || 0;
    const kp = Array.isArray(result.keyPoints) ? result.keyPoints : [];
    const score = Number.isFinite(Number(result.totalMarksAwarded))
      ? Number(result.totalMarksAwarded)
      : kp.reduce((s, k) => s + (Number(k.marksAwarded) || 0), 0);

    const entry = {
      id: `${Date.now()}`,
      studentName: studentName.trim() || "Unnamed student",
      subject: subject.trim() || "—",
      score,
      totalMarks: total,
      grade: result.grade || "—",
      date: todayLabel(),
    };

    setHistory((prev) => {
      const next = [entry].concat(prev).slice(0, HISTORY_LIMIT);
      saveHistory(next);
      return next;
    });
  }

  /** Returns the first problem plus which field caused it, so the UI can point at it. */
  function validate({ needPages, needScheme = true }) {
    if (needPages && pages.length === 0)
      return { field: "pages", message: "Upload the student's answer paper first." };

    /* With a question paper loaded, the questions and their marks come from the
       paper itself — the scheme box and the total field are not needed. */
    if (!needScheme) return null;

    if (!expectedAnswer.trim())
      return {
        field: "scheme",
        message:
          "The marking scheme box is still empty — the faint text in it is only an example. " +
          "Type or paste the scheme, or press “Use sample”.",
      };
    const m = Number(totalMarks);
    if (!m || m <= 0) return { field: "marks", message: "Enter a valid total marks value." };
    return null;
  }

  /** Flag the field, then put the cursor in it so the fix is one keystroke away. */
  function reportProblem(problem) {
    setInvalid(problem.field);
    setError(problem.message);
    const el = document.getElementById(`qiq-field-${problem.field}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      if (el.focus) setTimeout(() => el.focus(), 300);
    }
  }

  /* True when the question paper is driving the marking. Without one the
     original marking-scheme flow runs, exactly as before. */
  const examMode = !!(exam && exam.questions.length > 0);

  async function checkPaper() {
    if (examMode && exam.blocking) {
      setError("Enter the missing question marks before grading — see the question paper section.");
      return;
    }

    const problem = validate({ needPages: true, needScheme: !examMode });
    if (problem) {
      reportProblem(problem);
      return;
    }
    setInvalid("");

    setError("");
    setEvaluation(null);
    setActiveAnn(null);

    try {
      setStage("ocr");
      const pageTexts = await runOcr();

      /* Read the ink while the vision result is still fresh. This is local
         canvas work, not a network call, so it costs no tokens and cannot fail
         the run — an unmeasurable page simply loses its overlay. */
      const measured = await measurePages(pages);
      const doc = structureOcr(pageTexts);
      const confidence = attachGeometry(doc, measured);

      setPageBands(measured);
      setOcrDoc(doc);
      setGeometryByPage(confidence);
      setStudentAnswerText(doc.text);
      setEditedText(doc.text);

      setStage("evaluating");
      const result = examMode
        ? await runExamEvaluation(doc)
        : await runEvaluation(doc.text);

      setEvaluation(result);
      setEvalRun((n) => n + 1);
      recordHistory(result);
      setStage("done");
      setTab("paper");
    } catch (e) {
      setError(e.message || String(e));
      setStage(studentAnswerText ? "done" : "idle");
    } finally {
      setNotice("");
      setMarkProgress({ done: 0, total: 0, label: "" });
    }
  }

  /**
   * The question-paper-first marking run: match every answer to its question,
   * mark each one against the reference material, then let the engine compute
   * the totals. Projected onto the existing evaluation shape at the end so the
   * report card, annotations, animation and history all keep working.
   */
  async function runExamEvaluation(doc) {
    const llm = createLlm({ onRetry });
    const warnings = [];

    const { answers } = await matchAnswers(doc, exam, { llm, warnings });

    const retriever = refChunks.length ? createRetriever(refChunks) : null;
    const concurrency = await poolSize();

    const paper = await assessPaper({
      exam,
      answers,
      retriever,
      llm,
      warnings,
      concurrency,
      // The scheme box doubles as the answer key in exam mode. Question papers
      // exported from question banks often have empty "Answer Key:" fields, and
      // without a key the examiner must re-derive every answer itself.
      answerKey: expectedAnswer,
      onProgress: (done, total, label) => {
        setMarkProgress({ done, total, label });
        setNotice(label ? `Marking ${label} — ${done + 1} of ${total}` : "");
      },
    });

    setNotice("Writing the final remark…");
    const remark = await summarisePaper({ paper, llm });

    setRawResponse(JSON.stringify(paper, null, 2));
    setReasoning("");
    return toEvaluation(paper, remark);
  }

  /** Re-grade against an edited scheme or corrected OCR, skipping the vision pass. */
  async function reEvaluate(sourceText) {
    const problem = validate({ needPages: false, needScheme: !examMode });
    if (problem) {
      reportProblem(problem);
      return;
    }
    setInvalid("");

    const text = String(sourceText !== undefined ? sourceText : editedText || studentAnswerText).trim();
    if (!text) {
      setError("There is no extracted answer text to evaluate.");
      return;
    }

    setError("");
    setActiveAnn(null);
    try {
      setStage("evaluating");
      setStudentAnswerText(text);

      /* Corrected OCR text shifts every character offset, so the line index has
         to be rebuilt before annotations can be traced back to the page. The
         ink measurements still hold — only the text changed. On a single page
         this stays exact. Across several pages an edit destroys the page
         boundaries, and rather than guess at them we drop the overlay for this
         run and say so. */
      let doc = ocrDoc;
      let confidence = geometryByPage;

      if (!doc || text !== doc.text) {
        if (pages.length <= 1) {
          doc = structureOcr([text]);
          confidence = attachGeometry(doc, pageBands);
        } else {
          doc = null;
          confidence = {};
        }
        setOcrDoc(doc);
        setGeometryByPage(confidence);
      }

      const result = examMode && doc ? await runExamEvaluation(doc) : await runEvaluation(text);
      setEvaluation(result);
      setEvalRun((n) => n + 1);
      recordHistory(result);
      setStage("done");
      setTab(doc ? "paper" : "annotated");
    } catch (e) {
      setError(e.message || String(e));
      setStage("done");
    } finally {
      setNotice("");
    }
  }

  function clearHistory() {
    setHistory([]);
    saveHistory([]);
  }

  /* ============================================================== RENDER === */
  return (
    <div className="qiq-root">
      <style>{CSS}</style>

      <header className="qiq-header qiq-noprint">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="qiq-logo">Q</div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: -0.2 }}>
              QIQ <span style={{ color: C.faint, fontWeight: 500 }}>/</span> Descriptive Paper Checker
            </div>
            <div style={{ fontSize: 12, color: C.faint, marginTop: 2 }}>
              Groq · {OCR_MODEL} vision OCR · {EVAL_MODEL} examiner
            </div>
          </div>
        </div>
        <PipelineBar step={pipelineStep} running={busy} />
      </header>

      <div className="qiq-grid">
        {/* ================================================== LEFT PANEL === */}
        <aside className="qiq-panel qiq-noprint">
          <SectionTitle
            n="1"
            title="Question paper"
            action={
              exam && (
                <button className="qiq-mini-btn" onClick={clearExam}>
                  Remove
                </button>
              )
            }
          />

          {!exam && (
            <>
              <label className="qiq-drop qiq-drop-sm">
                <input
                  type="file"
                  accept="application/pdf,image/*"
                  multiple
                  hidden
                  onChange={(e) => {
                    ingestExam(e.target.files);
                    e.target.value = "";
                  }}
                />
                <div style={{ fontSize: 13, fontWeight: 600 }}>
                  {examBusy ? "Reading the question paper…" : "Upload the question paper"}
                </div>
                <div style={{ fontSize: 11.5, color: C.faint, marginTop: 4 }}>
                  PDF, scan or photo — QIQ reads the questions and their marks
                </div>
              </label>
              <div className="qiq-hint">
                Optional. Without one, QIQ falls back to the marking-scheme flow below.
              </div>
            </>
          )}

          {exam && <ExamPanel exam={exam} onMarks={setQuestionMarks} />}

          <SectionTitle
            n="2"
            title="Reference material"
            action={
              refFiles.length > 0 && (
                <button className="qiq-mini-btn" onClick={clearReferences}>
                  Clear
                </button>
              )
            }
          />
          <label className="qiq-drop qiq-drop-sm">
            <input
              type="file"
              accept="application/pdf"
              multiple
              hidden
              onChange={(e) => {
                ingestReferences(e.target.files);
                e.target.value = "";
              }}
            />
            <div style={{ fontSize: 13, fontWeight: 600 }}>
              {refBusy ? "Indexing…" : "Add reference PDFs"}
            </div>
            <div style={{ fontSize: 11.5, color: C.faint, marginTop: 4 }}>
              Textbook, notes, syllabus or model answers — marked against these first
            </div>
          </label>

          {refFiles.length > 0 && (
            <div className="qiq-reflist">
              {refFiles.map((f, i) => (
                <div key={i} className="qiq-refitem">
                  <span className="qiq-reficon">📄</span>
                  <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {f.name}
                  </span>
                  <span style={{ color: C.faint }}>
                    {f.pageCount}p · {f.chunks} chunks
                  </span>
                </div>
              ))}
              <div className="qiq-hint" style={{ marginTop: 2 }}>
                {refChunks.length} searchable passages indexed locally.
              </div>
            </div>
          )}
          {refFiles.length === 0 && (
            <div className="qiq-hint">
              Optional. Without it, marking falls back to the model's own knowledge and says so.
            </div>
          )}

          <SectionTitle n="3" title="Answer paper" />

          <label
            id="qiq-field-pages"
            className={`qiq-drop${dragging ? " is-dragging" : ""}${invalid === "pages" ? " qiq-invalid" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              addFiles(e.dataTransfer.files);
            }}
          >
            <input
              type="file"
              accept="image/png,image/jpeg,image/gif,image/webp,application/pdf"
              multiple
              style={{ display: "none" }}
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <div className="qiq-drop-icon">{preparing ? <span className="qiq-spinner" /> : "⬆"}</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>
              {preparing ? "Preparing pages…" : "Drop the answer sheet here"}
            </div>
            <div style={{ fontSize: 12, color: C.faint, marginTop: 4 }}>
              or click to browse · PDF, JPG, PNG · handwriting supported
            </div>
          </label>

          {pages.length > 0 && (
            <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
              {pages.map((p, i) => (
                <div key={p.id} className="qiq-file">
                  <img src={p.dataUrl} alt="" className="qiq-thumb" />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="qiq-ellipsis" style={{ fontSize: 13, fontWeight: 500 }}>
                      {p.label}
                    </div>
                    <div style={{ fontSize: 11, color: C.faint }}>
                      page {i + 1} · {(p.dataUrl.length / 1365).toFixed(0)} KB
                    </div>
                  </div>
                  <button className="qiq-x" onClick={() => removePage(p.id)} disabled={busy} title="Remove">
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <SectionTitle n="4" title="Student" />
          <div style={{ display: "grid", gap: 8 }}>
            <input
              className="qiq-input"
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              placeholder="Student name (appears on the report)"
            />
            <input
              className="qiq-input"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="Subject, e.g. Biology"
            />
          </div>

          {/* The question paper supplies the questions and their marks, so these
              two inputs are only needed for the original scheme-based flow. */}
          {!examMode && (
            <>
            <SectionTitle
              n="5"
              title="Expected answer / marking scheme"
              action={
                !expectedAnswer.trim() && (
                  <button
                    className="qiq-mini-btn"
                    onClick={() => {
                      setExpectedAnswer(SAMPLE_SCHEME);
                      setInvalid("");
                      setError("");
                    }}
                  >
                    Use sample
                  </button>
                )
              }
            />
            <textarea
              id="qiq-field-scheme"
              className={`qiq-input qiq-textarea${invalid === "scheme" ? " qiq-invalid" : ""}`}
              value={expectedAnswer}
              onChange={(e) => {
                setExpectedAnswer(e.target.value);
                if (invalid === "scheme") {
                  setInvalid("");
                  setError("");
                }
              }}
              placeholder="Type or paste what a full-mark answer must cover — one line per point, with the marks for each."
            />
            <div className="qiq-hint">
              {expectedAnswer.trim()
                ? `${expectedAnswer.trim().split(/\s+/).length} words entered`
                : "Empty — this box must contain real text before the paper can be graded."}
            </div>

            <SectionTitle n="6" title="Total marks" />
            <input
              id="qiq-field-marks"
              className={`qiq-input${invalid === "marks" ? " qiq-invalid" : ""}`}
              type="number"
              min="1"
              value={totalMarks}
              onChange={(e) => {
                setTotalMarks(e.target.value);
                if (invalid === "marks") {
                  setInvalid("");
                  setError("");
                }
              }}
              placeholder="20"
            />
            </>
          )}

          {error && (
            <div className="qiq-error">
              <strong style={{ display: "block", marginBottom: 4 }}>Something went wrong</strong>
              {error}
            </div>
          )}

          <button className="qiq-btn" onClick={checkPaper} disabled={busy || preparing}>
            {busy ? (
              <>
                <span className="qiq-spinner" />
                {stage === "ocr" ? "Reading paper…" : "Evaluating…"}
              </>
            ) : evaluation ? (
              "Check Paper again"
            ) : (
              "Check Paper"
            )}
          </button>

          {evaluation && !busy && (
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button className="qiq-btn qiq-btn-ghost" onClick={() => reEvaluate()} style={{ flex: 1 }}>
                Re-grade only
              </button>
              <button className="qiq-btn qiq-btn-ghost" onClick={resetAll} style={{ flex: 1 }}>
                New paper
              </button>
            </div>
          )}

          <HistoryPanel history={history} onClear={clearHistory} />

          <p style={{ fontSize: 11, color: C.faint, marginTop: 14, lineHeight: 1.6 }}>
            “Re-grade only” reuses the text already extracted, so you can tweak the marking scheme or fix
            OCR mistakes without paying for another vision pass.
          </p>
        </aside>

        {/* ================================================= RIGHT PANEL === */}
        <main className="qiq-panel qiq-right">
          {busy && (
            <Processing
              stage={stage}
              elapsed={elapsed}
              progress={ocrProgress}
              notice={notice}
              marking={markProgress}
            />
          )}

          {!busy && !evaluation && <EmptyState hasPages={pages.length > 0} />}

          {!busy && evaluation && (
            <>
              <div className="qiq-tabs qiq-noprint">
                {[
                  ["paper", "Marked Paper"],
                  ["annotated", "Annotated Text"],
                  ["score", "Report Card"],
                  ["raw", "Raw OCR Text"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={`qiq-tab${tab === id ? " is-active" : ""}`}
                    onClick={() => setTab(id)}
                  >
                    {label}
                    {id === "score" && (
                      <span className="qiq-tab-badge" style={{ background: gradeColor(evaluation.grade) }}>
                        {scoreAwarded}/{scoreTotal}
                      </span>
                    )}
                    {id === "paper" && lowConfidenceCount > 0 && (
                      <span className="qiq-tab-warn" title={`${lowConfidenceCount} mark(s) need manual review`}>
                        ⚠️ {lowConfidenceCount}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              <div className="qiq-tabbody">
                {tab === "paper" && (
                  <MarkedPaper
                    pages={pages}
                    annotations={annotations}
                    annBoxes={annBoxes}
                    geometryByPage={geometryByPage}
                    geometryLevel={geometryLevel}
                    placedOnPage={placedOnPage}
                    activeAnn={activeAnn}
                    setActiveAnn={setActiveAnn}
                    revealed={revealed}
                    markingInProgress={markingInProgress}
                    onRevealAll={revealAll}
                    onShowText={() => setTab("annotated")}
                  />
                )}
                {tab === "annotated" && (
                  <AnnotatedView
                    anchored={anchored}
                    annotations={annotations}
                    activeAnn={activeAnn}
                    setActiveAnn={setActiveAnn}
                    revealed={revealed}
                    markingInProgress={markingInProgress}
                    onRevealAll={revealAll}
                  />
                )}
                {tab === "score" && (
                  <ReportCard
                    evaluation={evaluation}
                    keyPoints={keyPoints}
                    awarded={scoreAwarded}
                    total={scoreTotal}
                    runKey={evalRun}
                    studentName={studentName}
                    setStudentName={setStudentName}
                    subject={subject}
                    setSubject={setSubject}
                    reportDate={reportDate}
                    setReportDate={setReportDate}
                  />
                )}
                {tab === "raw" && (
                  <RawView
                    text={editedText}
                    setText={setEditedText}
                    dirty={editedText !== studentAnswerText}
                    onReEvaluate={() => reEvaluate(editedText)}
                    rawResponse={rawResponse}
                    reasoning={reasoning}
                  />
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

/* ============================================================ PIECES ====== */

function EmptyState({ hasPages }) {
  return (
    <div className="qiq-empty">
      <div className="qiq-empty-icon">📝</div>
      <div style={{ fontSize: 16, fontWeight: 600, color: C.text }}>
        {hasPages ? "Ready to grade" : "No paper checked yet"}
      </div>
      <p style={{ fontSize: 13, color: C.faint, maxWidth: 420, lineHeight: 1.7, marginTop: 8 }}>
        {hasPages
          ? "Add the marking scheme on the left, set the total marks, then press Check Paper."
          : "Upload a scanned or typed answer sheet, describe what a full-mark answer should cover, and QIQ will read, annotate and grade it the way an experienced teacher would."}
      </p>
    </div>
  );
}

function Processing({ stage, elapsed, progress, notice, marking }) {
  const steps = [
    {
      id: "ocr",
      title: "Reading the paper",
      sub:
        progress.total > 1
          ? `Transcribing page ${Math.min(progress.done + 1, progress.total)} of ${progress.total}, handwriting included`
          : "Vision model is transcribing the page, handwriting included",
    },
    {
      id: "evaluating",
      title: "Evaluating like a teacher",
      /* Per-question marking is sequential and rate-limited, so a paper can sit
         here for minutes. Saying which question is being marked is the
         difference between "working" and "frozen". */
      sub:
        marking && marking.total > 0
          ? `Marking ${marking.label || "question"} — ${Math.min(marking.done + 1, marking.total)} of ${marking.total}`
          : "Reasoning through the marking scheme and partial credit at high effort",
    },
  ];
  const current = steps.findIndex((s) => s.id === stage);

  return (
    <div className="qiq-empty">
      <div className="qiq-orb" />
      <div style={{ fontSize: 16, fontWeight: 600, marginTop: 22 }}>Checking the paper…</div>
      <div style={{ fontSize: 12, color: C.faint, marginTop: 4 }}>{elapsed}s elapsed</div>

      <div style={{ display: "grid", gap: 10, marginTop: 26, width: "min(440px, 100%)" }}>
        {steps.map((s, i) => {
          const done = i < current;
          const active = i === current;
          return (
            <div key={s.id} className={`qiq-proc${active ? " is-active" : ""}`}>
              <div className="qiq-proc-dot">
                {done ? "✓" : active ? <span className="qiq-spinner" /> : i + 1}
              </div>
              <div style={{ textAlign: "left" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: done || active ? C.text : C.faint }}>
                  {s.title}
                </div>
                <div style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>{s.sub}</div>
              </div>
            </div>
          );
        })}
      </div>

      {notice && <div className="qiq-notice">{notice}</div>}
    </div>
  );
}

/* --------------------------------------------------------- student history -- */

function HistoryPanel({ history, onClear }) {
  if (!history.length) return null;

  const pct = (h) => (h.totalMarks > 0 ? (h.score / h.totalMarks) * 100 : 0);

  return (
    <div style={{ marginTop: 22 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.7, textTransform: "uppercase", color: C.dim }}>
          Recent checks
        </span>
        <button className="qiq-mini-btn" style={{ marginLeft: "auto" }} onClick={onClear}>
          Clear
        </button>
      </div>

      <div style={{ display: "grid", gap: 7 }}>
        {history.map((h, i) => {
          /* Compare against the next entry down, which is the previous paper. */
          const prev = history[i + 1];
          const delta = prev ? pct(h) - pct(prev) : 0;
          const trend = !prev ? "→" : delta > 1 ? "↑" : delta < -1 ? "↓" : "→";
          const trendColor = !prev ? C.faint : delta > 1 ? C.green : delta < -1 ? C.red : C.faint;

          return (
            <div key={h.id} className="qiq-hist">
              <span className="qiq-hist-trend" style={{ color: trendColor }} title={
                prev ? `${delta > 0 ? "+" : ""}${delta.toFixed(0)}% vs previous paper` : "First recorded paper"
              }>
                {trend}
              </span>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="qiq-ellipsis" style={{ fontSize: 12.5, fontWeight: 600, color: C.text }}>
                  {h.studentName}
                </div>
                <div className="qiq-ellipsis" style={{ fontSize: 10.5, color: C.faint }}>
                  {h.subject} · {h.date}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 12.5, fontWeight: 700, color: C.text, fontVariantNumeric: "tabular-nums" }}>
                  {h.score}/{h.totalMarks}
                </div>
                <div style={{ fontSize: 10.5, fontWeight: 700, color: gradeColor(h.grade) }}>{h.grade}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Where a question's marking came from. This is the visible half of the
 * reference-first rule: "the textbook says so" and "I knew this already" are
 * different claims, and a student is entitled to know which one they were
 * marked on.
 */
function GroundingBadge({ grounding }) {
  const map = {
    [GROUNDING.REFERENCE]: ["is-ref", "From reference"],
    [GROUNDING.GENERAL]: ["is-general", "General knowledge"],
    [GROUNDING.INSUFFICIENT]: ["is-insufficient", "Reference insufficient"],
  };
  const [cls, label] = map[grounding] || map[GROUNDING.GENERAL];
  return <span className={`qiq-ground ${cls}`}>{label}</span>;
}

/* ----------------------------------------------------------- exam panel -- */

/**
 * The question paper as QIQ understood it.
 *
 * Every mark is editable, because the validator refuses to invent one that the
 * paper did not print — showing the gap and letting the teacher fill it is the
 * honest alternative to a confident guess that then caps a real student's score.
 */
function ExamPanel({ exam, onMarks }) {
  const missing = exam.questions.filter((q) => q.maxMarks === null).length;

  return (
    <div className="qiq-exam">
      <div className="qiq-exam-head">
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
            {exam.title || "Question paper"}
          </div>
          <div style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>
            {exam.subject ? exam.subject + " · " : ""}
            {exam.questions.length} question{exam.questions.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="qiq-exam-total">
          <span style={{ fontSize: 17, fontWeight: 800, color: missing ? C.amber : C.green }}>
            {exam.totalMarks}
          </span>
          <span style={{ fontSize: 10.5, color: C.faint, display: "block" }}>total marks</span>
        </div>
      </div>

      <div className="qiq-exam-rows">
        {exam.questions.map((q) => (
          <div key={q.id} className={`qiq-exam-row${q.maxMarks === null ? " is-missing" : ""}`}>
            <span className="qiq-exam-num">{q.number}</span>
            <span className="qiq-exam-text" title={q.text}>
              {q.text || <em style={{ color: C.faint }}>no wording read</em>}
            </span>
            <span className="qiq-exam-type">{q.type.replace(/_/g, " ")}</span>
            <input
              className="qiq-exam-marks"
              type="number"
              min="1"
              value={q.maxMarks === null ? "" : q.maxMarks}
              placeholder="?"
              onChange={(e) => onMarks(q.id, e.target.value)}
              aria-label={`Marks for question ${q.number}`}
            />
          </div>
        ))}
      </div>

      {exam.warnings.length > 0 && (
        <div className="qiq-exam-warn">
          {exam.warnings.map((w, i) => (
            <div key={i} className="qiq-exam-warnrow">
              <span aria-hidden="true">⚠️</span>
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------- Tab 1: marked paper -- */

/* What to admit when the placement is not solid. Silence would be worse than a
   rough box: a student trusts a mark that sits on their own handwriting. */
const GEOMETRY_NOTE = {
  high: "",
  medium: "Lines were matched approximately — a mark may sit one line off.",
  low:
    "This page did not separate into clean lines of writing, so the marks below are placed roughly. " +
    "Use “Annotated Text” to see exactly which words each comment refers to.",
  none: "The pages could not be measured, so no mark could be placed on the handwriting.",
};

/**
 * The student's own page, marked.
 *
 * This is the view the product is really about: the handwriting stays, and the
 * marking is laid over it where the teacher would have written it. It shares the
 * annotation objects, the type colours, the badges, the reveal timing and the
 * margin notes with the text view — only the positioning differs, from inline
 * spans to boxes measured off the image.
 */
function MarkedPaper({
  pages,
  annotations,
  annBoxes,
  geometryByPage,
  geometryLevel,
  placedOnPage,
  activeAnn,
  setActiveAnn,
  revealed,
  markingInProgress,
  onRevealAll,
  onShowText,
}) {
  const focus = (idx) => {
    setActiveAnn(idx);
    const el = document.getElementById(`qiq-note-${idx}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  /* One annotation can occupy boxes on more than one page, so the grouping is
     by page rather than by annotation. Top-to-bottom order keeps the tab key
     and the eye moving down the page together. */
  const boxesForPage = (pageNumber) => {
    const out = [];
    for (const [key, list] of Object.entries(annBoxes)) {
      for (const entry of list) {
        if (entry.page === pageNumber) out.push({ idx: Number(key), bbox: entry.bbox });
      }
    }
    return out.sort((a, b) => a.bbox.y - b.bbox.y);
  };

  /* Pages can be removed after a grading run, and an edited transcript drops the
     geometry on purpose. Either way this view has nothing to stand on. */
  const note = pages.length === 0
    ? "The pages have been removed, so there is nothing left to mark on."
    : GEOMETRY_NOTE[geometryLevel] || "";
  const unplaceable = annotations.length - placedOnPage;

  return (
    <div className="qiq-annot-wrap">
      <div>
        <div className="qiq-subhead">
          <div>
            <div style={{ fontSize: 13, fontWeight: 700 }}>
              The student's paper, marked
              {markingInProgress && <span className="qiq-marking-dot" />}
            </div>
            <div style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>
              {markingInProgress
                ? `Marking… ${revealed} of ${annotations.length}`
                : `${placedOnPage} of ${annotations.length} marks placed on the handwriting` +
                  (unplaceable > 0 ? ` · ${unplaceable} in the margin` : "")}
            </div>
          </div>
          {markingInProgress ? (
            <button className="qiq-mini-btn" onClick={onRevealAll}>
              Skip animation
            </button>
          ) : (
            <Legend />
          )}
        </div>

        {note && (
          <div className="qiq-geom-note">
            <span aria-hidden="true">⚠️</span>
            <span>
              {note}{" "}
              <button className="qiq-linkbtn" onClick={onShowText}>
                Open the text view
              </button>
            </span>
          </div>
        )}

        <div className="qiq-pages">
          {pages.map((page, i) => {
            const pageNumber = i + 1;
            const boxes = boxesForPage(pageNumber);
            const shown = boxes.filter((b) => b.idx < revealed).length;

            return (
              <figure key={page.id} className="qiq-page-block">
                <div className="qiq-page-stage">
                  <img className="qiq-page-img" src={page.dataUrl} alt={`Page ${pageNumber}`} />

                  {boxes.map(({ idx, bbox }) => {
                    /* Before its turn the box is not rendered at all, so the
                       marking lands on the page one comment at a time — the
                       same gate the text view uses. */
                    if (idx >= revealed) return null;

                    const ann = annotations[idx];
                    if (!ann) return null;
                    const s = typeStyle(ann.type);
                    const unsure = isLowConfidence(ann);
                    // No room outside the box for the badge — tuck it inside.
                    const tight = bbox.x + bbox.width > 0.86;

                    return (
                      <div
                        key={`${idx}-${pageNumber}`}
                        className={`qiq-ann-box${activeAnn === idx ? " is-active" : ""}${
                          tight ? " is-tight" : ""
                        }`}
                        style={{
                          left: `${bbox.x * 100}%`,
                          top: `${bbox.y * 100}%`,
                          width: `${bbox.width * 100}%`,
                          height: `${bbox.height * 100}%`,
                          background: s.bg,
                          boxShadow: `inset 0 -2px 0 ${s.color}`,
                        }}
                        onMouseEnter={() => setActiveAnn(idx)}
                        onMouseLeave={() => setActiveAnn(null)}
                        onClick={() => focus(idx)}
                        title={ann.comment}
                      >
                        <span className="qiq-ann-badge" style={{ background: s.color }}>
                          {marksBadge(ann)}
                        </span>
                        {unsure && <span className="qiq-ann-warn">⚠️</span>}

                        <span className="qiq-pop">
                          <span style={{ color: s.color, fontWeight: 700 }}>
                            {s.icon} {s.label}
                          </span>
                          {Number.isFinite(Number(ann.marks)) && (
                            <span style={{ color: C.dim }}>
                              {" "}
                              · {ann.marks} mark{Math.abs(Number(ann.marks)) === 1 ? "" : "s"}
                            </span>
                          )}
                          <span style={{ display: "block", marginTop: 5, color: C.text }}>
                            {ann.comment}
                          </span>
                          {unsure && (
                            <span
                              style={{ display: "block", marginTop: 6, color: C.amber, fontSize: 11.5 }}
                            >
                              ⚠️ AI is unsure — please verify this mark manually
                            </span>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <figcaption className="qiq-page-cap">
                  <span>
                    Page {pageNumber} of {pages.length}
                  </span>
                  <span style={{ color: C.faint }}>
                    {shown} mark{shown === 1 ? "" : "s"}
                    {geometryByPage[pageNumber] === "high" ? "" : " · approximate placement"}
                  </span>
                </figcaption>
              </figure>
            );
          })}
        </div>
      </div>

      <MarginNotes
        annotations={annotations}
        revealed={revealed}
        activeAnn={activeAnn}
        setActiveAnn={setActiveAnn}
        isUnpinned={(idx) => !annBoxes[idx]}
      />
    </div>
  );
}

/* --------------------------------------------------- Tab 2: annotated text -- */

function AnnotatedView({
  anchored,
  annotations,
  activeAnn,
  setActiveAnn,
  revealed,
  markingInProgress,
  onRevealAll,
}) {
  const { segments, anchoredCount, unanchored } = anchored;

  const focus = (idx) => {
    setActiveAnn(idx);
    const el = document.getElementById(`qiq-note-${idx}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="qiq-annot-wrap">
      <div>
        <div className="qiq-subhead">
          <div>
            <div style={{ fontSize: 13, fontWeight: 700 }}>
              Student's answer, marked
              {markingInProgress && <span className="qiq-marking-dot" />}
            </div>
            <div style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>
              {markingInProgress
                ? `Marking… ${revealed} of ${annotations.length}`
                : `${anchoredCount} of ${annotations.length} comments pinned to the text — hover or click a highlight to read the remark`}
            </div>
          </div>
          {markingInProgress ? (
            <button className="qiq-mini-btn" onClick={onRevealAll}>
              Skip animation
            </button>
          ) : (
            <Legend />
          )}
        </div>

        <div className="qiq-paper">
          {segments.map((seg, i) => {
            if (!seg.ann) return <span key={i}>{seg.text}</span>;

            const shown = seg.idx < revealed;
            const s = typeStyle(seg.ann.type);
            const unsure = isLowConfidence(seg.ann);

            /* Before its turn a highlight renders as plain text, so the marking
               visibly lands on the page one comment at a time. */
            if (!shown) return <span key={i}>{seg.text}</span>;

            return (
              <span
                key={i}
                className={`qiq-hl${activeAnn === seg.idx ? " is-active" : ""}`}
                style={{
                  background: s.bg,
                  boxShadow: `inset 0 -2px 0 ${s.color}`,
                  animationDelay: "0ms",
                }}
                onMouseEnter={() => setActiveAnn(seg.idx)}
                onMouseLeave={() => setActiveAnn(null)}
                onClick={() => focus(seg.idx)}
              >
                {seg.text}
                <span className="qiq-marks-badge" style={{ background: s.color }}>
                  {marksBadge(seg.ann)}
                </span>
                {unsure && <ConfidenceFlag confidence={seg.ann.confidence} compact />}
                <span className="qiq-pop">
                  <span style={{ color: s.color, fontWeight: 700 }}>
                    {s.icon} {s.label}
                  </span>
                  {Number.isFinite(Number(seg.ann.marks)) && (
                    <span style={{ color: C.dim }}>
                      {" "}
                      · {seg.ann.marks} mark{Math.abs(Number(seg.ann.marks)) === 1 ? "" : "s"}
                    </span>
                  )}
                  <span style={{ display: "block", marginTop: 5, color: C.text }}>{seg.ann.comment}</span>
                  {unsure && (
                    <span style={{ display: "block", marginTop: 6, color: C.amber, fontSize: 11.5 }}>
                      ⚠️ AI is unsure — please verify this mark manually
                    </span>
                  )}
                </span>
              </span>
            );
          })}
        </div>
      </div>

      <MarginNotes
        annotations={annotations}
        revealed={revealed}
        activeAnn={activeAnn}
        setActiveAnn={setActiveAnn}
        isUnpinned={(idx) => unanchored.some((u) => u.idx === idx)}
      />
    </div>
  );
}

/**
 * The teacher's notes down the side. Shared by both marking views, so a comment
 * reads identically whether it was pinned to the handwriting or to the
 * transcript — and so an unpinnable one still has somewhere to live.
 */
function MarginNotes({ annotations, revealed, activeAnn, setActiveAnn, isUnpinned }) {
  return (
    <div className="qiq-margin">
      <div
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: 0.7,
          textTransform: "uppercase",
          color: C.dim,
          marginBottom: 10,
        }}
      >
        Teacher's margin notes
      </div>

      {annotations.length === 0 && (
        <div style={{ fontSize: 12.5, color: C.faint }}>No inline comments were returned.</div>
      )}

      {annotations.map((ann, idx) => {
        if (idx >= revealed) return null; // slides in when its highlight lands

        const s = typeStyle(ann.type);
        const unpinned = isUnpinned(idx);
        const unsure = isLowConfidence(ann);

        return (
          <div
            key={idx}
            id={`qiq-note-${idx}`}
            className={`qiq-note qiq-note-enter${activeAnn === idx ? " is-active" : ""}${
              unsure ? " is-unsure" : ""
            }`}
            style={{ borderLeft: `3px solid ${s.color}` }}
            onMouseEnter={() => setActiveAnn(idx)}
            onMouseLeave={() => setActiveAnn(null)}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
              <span className="qiq-note-num" style={{ background: s.color }}>
                {idx + 1}
              </span>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: s.color,
                  textTransform: "uppercase",
                  letterSpacing: 0.4,
                }}
              >
                {s.label}
              </span>
              {unsure && <ConfidenceFlag confidence={ann.confidence} />}
              <span
                className="qiq-note-marks"
                style={{ marginLeft: "auto", background: s.color }}
              >
                {marksBadge(ann)}
              </span>
            </div>
            {unpinned && ann.text && (
              <div style={{ fontSize: 11.5, color: C.faint, fontStyle: "italic", marginBottom: 4 }}>
                “{String(ann.text).slice(0, 90)}
                {String(ann.text).length > 90 ? "…" : ""}”
              </div>
            )}
            <div className="qiq-handwrite" style={{ fontSize: 14, lineHeight: 1.55, color: C.text }}>
              {ann.comment}
            </div>
            {unsure && (
              <div style={{ marginTop: 6, fontSize: 11, color: C.amber, lineHeight: 1.5 }}>
                AI is unsure — please verify this mark manually
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------ Tab 2: report card -- */

function ScoreArc({ awarded, total, runKey }) {
  const shown = useCountUp(awarded, COUNT_MS, runKey);
  const pct = total > 0 ? Math.max(0, Math.min(1, shown / total)) : 0;
  const finalPct = total > 0 ? (awarded / total) * 100 : 0;
  const color = finalPct >= 75 ? C.green : finalPct >= 50 ? "#3B82F6" : finalPct >= 35 ? C.amber : C.red;

  const R = 62;
  const CIRC = 2 * Math.PI * R;

  return (
    <div className="qiq-arc-wrap">
      <svg width="152" height="152" viewBox="0 0 152 152" className="qiq-arc">
        <circle cx="76" cy="76" r={R} className="qiq-arc-track" />
        <circle
          cx="76"
          cy="76"
          r={R}
          className="qiq-arc-fill"
          stroke={color}
          strokeDasharray={CIRC}
          strokeDashoffset={CIRC * (1 - pct)}
        />
      </svg>
      <div className="qiq-arc-label">
        <div className="qiq-arc-score" style={{ color }}>
          {Math.round(shown)}
          <span className="qiq-arc-total"> / {total}</span>
        </div>
        <div className="qiq-arc-pct">{finalPct.toFixed(0)}%</div>
      </div>
    </div>
  );
}

function ReportCard({
  evaluation,
  keyPoints,
  awarded,
  total,
  runKey,
  studentName,
  setStudentName,
  subject,
  setSubject,
  reportDate,
  setReportDate,
}) {
  const g = gradeColor(evaluation.grade);
  const positives = Array.isArray(evaluation.thingsWellDone) ? evaluation.thingsWellDone : [];
  const improvements = Array.isArray(evaluation.improvementAreas) ? evaluation.improvementAreas : [];
  const passed = total > 0 && awarded / total >= PASS_THRESHOLD;

  const remark = String(evaluation.overallRemark || "");
  const typed = useTypewriter(remark, TYPE_MS, runKey);
  const speech = useSpeech();

  const qualityChip = (q, covered) => {
    if (q === "well" || (covered && !q)) return { label: "Covered well", color: C.green };
    if (q === "partially") return { label: "Partial", color: C.amber };
    return { label: "Not covered", color: C.red };
  };

  return (
    <div>
      <div className="qiq-report-actions qiq-noprint">
        <button className="qiq-btn qiq-btn-sm" onClick={() => window.print()}>
          🖨️ Print Report
        </button>
        <span style={{ fontSize: 11.5, color: C.faint }}>
          Prints black-on-white — the dark theme and all controls are stripped out.
        </span>
      </div>

      <div className="qiq-report">
        {/* ------------------------------------------------ report header */}
        <div className="qiq-report-head">
          <div className="qiq-report-brand">
            <div className="qiq-report-logo">Q</div>
            <div>
              <div className="qiq-report-title">Assessment Report</div>
              <div className="qiq-report-sub">QIQ · AI-assisted descriptive evaluation</div>
            </div>
          </div>
          <div className="qiq-report-serial">
            No. {String(runKey).padStart(4, "0")}
          </div>
        </div>

        {/* -------------------------------------------------- identity row */}
        <div className="qiq-report-fields">
          <label className="qiq-rfield">
            <span>Student name</span>
            <input
              className="qiq-rinput"
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              placeholder="—"
            />
          </label>
          <label className="qiq-rfield">
            <span>Subject</span>
            <input
              className="qiq-rinput"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="—"
            />
          </label>
          <label className="qiq-rfield">
            <span>Date</span>
            <input
              className="qiq-rinput"
              value={reportDate}
              onChange={(e) => setReportDate(e.target.value)}
              placeholder="—"
            />
          </label>
        </div>

        {/* ------------------------------------------------- score + stamp */}
        <div className="qiq-report-score">
          <ScoreArc awarded={awarded} total={total} runKey={runKey} />

          <div className="qiq-grade-block">
            <div className="qiq-grade-stamp" key={`g${runKey}`} style={{ borderColor: g, color: g }}>
              {evaluation.grade || "—"}
            </div>
            <div className="qiq-grade-caption">Grade awarded</div>
          </div>

          <div
            className={`qiq-passmark ${passed ? "is-pass" : "is-fail"}`}
            key={`p${runKey}`}
            title={`Pass mark is ${Math.round(PASS_THRESHOLD * 100)}%`}
          >
            {passed ? "PASS" : "FAIL"}
          </div>
        </div>

        {/* ---------------------------------------------------- breakdown */}
        <div className="qiq-report-section">
          <div className="qiq-report-h">Marks breakdown</div>
          <div className="qiq-table-wrap">
            <table className="qiq-table">
              <thead>
                <tr>
                  <th style={{ width: "36%" }}>Key point</th>
                  <th style={{ width: 130 }}>Status</th>
                  <th style={{ width: 90, textAlign: "right" }}>Marks</th>
                  <th>Teacher's note</th>
                </tr>
              </thead>
              <tbody>
                {keyPoints.map((k, i) => {
                  const chip = qualityChip(k.quality, k.covered);
                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 600 }}>{k.point}</td>
                      <td>
                        <span
                          className="qiq-chip"
                          style={{ color: chip.color, borderColor: `${chip.color}55`, background: `${chip.color}14` }}
                        >
                          {chip.label}
                        </span>
                      </td>
                      <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        <strong style={{ color: chip.color }}>{k.marksAwarded ?? 0}</strong>
                        <span style={{ color: C.faint }}> / {k.marksTotal ?? "—"}</span>
                      </td>
                      <td style={{ color: C.dim, lineHeight: 1.55 }}>
                        {k.teacherNote}
                        {/* Where the judgement came from. Present only for the
                            question-paper pipeline; the older flow has no
                            reference material to attribute to. */}
                        {k.grounding && (
                          <div style={{ marginTop: 6, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            <GroundingBadge grounding={k.grounding} />
                            {k.evidence && k.evidence.length > 0 && (
                              <span style={{ fontSize: 10.5, color: C.faint }}>
                                {[...new Set(k.evidence.map((e) => e.source).filter(Boolean))].join(", ")}
                              </span>
                            )}
                            {Number.isFinite(k.confidence) && k.confidence < LOW_CONFIDENCE && (
                              <ConfidenceFlag confidence={k.confidence} compact />
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {keyPoints.length === 0 && (
                  <tr>
                    <td colSpan={4} style={{ color: C.faint }}>
                      No key-point breakdown returned.
                    </td>
                  </tr>
                )}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2} style={{ fontWeight: 700 }}>
                    Total
                  </td>
                  <td style={{ textAlign: "right", fontWeight: 800, color: g, fontVariantNumeric: "tabular-nums" }}>
                    {awarded} / {total}
                  </td>
                  <td />
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

        {/* ------------------------------------------------------- remark */}
        <div className="qiq-report-section">
          <div className="qiq-report-h">
            Teacher's remark
            {speech.supported && (
              <span className="qiq-voice qiq-noprint">
                {speech.state === "idle" && (
                  <button className="qiq-mini-btn" onClick={() => speech.speak(remark)}>
                    🔊 Read Remark
                  </button>
                )}
                {speech.state === "speaking" && (
                  <>
                    <button className="qiq-mini-btn" onClick={speech.pause}>
                      ⏸ Pause
                    </button>
                    <button className="qiq-mini-btn" onClick={speech.stop}>
                      ⏹ Stop
                    </button>
                    <span className="qiq-speaking">speaking…</span>
                  </>
                )}
                {speech.state === "paused" && (
                  <>
                    <button className="qiq-mini-btn" onClick={speech.resume}>
                      ▶ Resume
                    </button>
                    <button className="qiq-mini-btn" onClick={speech.stop}>
                      ⏹ Stop
                    </button>
                  </>
                )}
              </span>
            )}
          </div>

          <blockquote className="qiq-remark" onClick={typed.skip} title={typed.done ? "" : "Click to skip"}>
            {typed.shown}
            {!typed.done && <span className="qiq-caret" />}
            <footer className="qiq-remark-sign">— Examiner, QIQ</footer>
          </blockquote>
        </div>

        {/* --------------------------------------- positives / improvements */}
        <div className="qiq-two-col">
          <div className="qiq-listcard" style={{ borderTop: `2px solid ${C.green}` }}>
            <div style={{ fontSize: 12.5, fontWeight: 700, color: C.green, marginBottom: 10 }}>
              What was done well
            </div>
            {positives.length === 0 && <div style={{ fontSize: 12.5, color: C.faint }}>No positives recorded.</div>}
            <ul className="qiq-list">
              {positives.map((t, i) => (
                <li key={i}>
                  <span style={{ color: C.green, fontWeight: 700, flexShrink: 0 }}>✓</span>
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="qiq-listcard" style={{ borderTop: `2px solid ${C.amber}` }}>
            <div style={{ fontSize: 12.5, fontWeight: 700, color: C.amber, marginBottom: 10 }}>
              Areas to improve
            </div>
            {improvements.length === 0 && (
              <div style={{ fontSize: 12.5, color: C.faint }}>No improvement areas recorded.</div>
            )}
            <ul className="qiq-list">
              {improvements.map((t, i) => (
                <li key={i}>
                  <span style={{ color: C.amber, fontWeight: 700, flexShrink: 0 }}>↗</span>
                  <span>
                    {t}{" "}
                    <a
                      className="qiq-study qiq-noprint"
                      href={`https://www.google.com/search?q=${encodeURIComponent(t)}+explanation+for+students`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Study →
                    </a>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="qiq-report-foot">
          Generated by QIQ using {EVAL_MODEL} · Marks flagged ⚠️ were low-confidence and should be
          verified by the teacher before this report is issued.
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- Tab 3: raw -- */

function RawView({ text, setText, dirty, onReEvaluate, rawResponse, reasoning }) {
  const [panel, setPanel] = useState("");
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const toggle = (id) => setPanel((p) => (p === id ? "" : id));

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="qiq-subhead">
        <div>
          <div style={{ fontSize: 13, fontWeight: 700 }}>Extracted text</div>
          <div style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>
            {words} words · {text.length} characters · fix any misread handwriting and re-grade
          </div>
        </div>
        {dirty && (
          <button className="qiq-btn qiq-btn-sm" onClick={onReEvaluate}>
            Re-grade corrected text
          </button>
        )}
      </div>

      <textarea className="qiq-mono" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <button className="qiq-link" onClick={() => toggle("raw")}>
          {panel === "raw" ? "Hide" : "Show"} raw evaluator response
        </button>
        {reasoning && (
          <button className="qiq-link" onClick={() => toggle("reasoning")}>
            {panel === "reasoning" ? "Hide" : "Show"} examiner's reasoning trace
          </button>
        )}
      </div>

      {panel === "raw" && <pre className="qiq-pre">{rawResponse}</pre>}
      {panel === "reasoning" && <pre className="qiq-pre">{reasoning}</pre>}
    </div>
  );
}

/* ================================================================= CSS ==== */

const CSS = `
.qiq-root {
  min-height: 100vh; background:
    radial-gradient(900px 500px at 12% -8%, rgba(37,99,235,0.16), transparent 60%),
    radial-gradient(700px 460px at 92% 0%, rgba(124,58,237,0.14), transparent 60%),
    ${C.navy};
  color: ${C.text};
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Roboto, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  padding: 20px;
}
.qiq-root *, .qiq-root *::before, .qiq-root *::after { box-sizing: border-box; }

.qiq-header {
  display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
  background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(15,23,42,.7));
  border:1px solid ${C.border}; border-radius:16px; padding:14px 18px; margin-bottom:16px;
  backdrop-filter: blur(8px);
}
.qiq-logo {
  width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
  background: linear-gradient(135deg, ${C.blue}, ${C.purple});
  font-weight:800; font-size:19px; color:#fff; box-shadow:0 6px 20px rgba(37,99,235,.38);
}

.qiq-pipeline { display:flex; align-items:center; flex-wrap:wrap; }
.qiq-pipe-item { display:flex; align-items:center; gap:8px; }
.qiq-pipe-dot {
  width:24px; height:24px; border-radius:50%; display:grid; place-items:center;
  font-size:11px; font-weight:700; background:#0B1220; color:${C.faint};
  border:1px solid ${C.border}; transition:.3s;
}
.qiq-pipe-dot.is-done { background:${C.blue}; border-color:${C.blue}; color:#fff; }
.qiq-pipe-dot.is-active {
  background: linear-gradient(135deg, ${C.blue}, ${C.purple}); border-color:transparent; color:#fff;
  animation: qiq-pulse 2s infinite;
}
.qiq-pipe-line { width:34px; height:2px; margin:0 10px; border-radius:2px; transition:.4s; }
@keyframes qiq-pulse { 0%,100%{box-shadow:0 0 0 4px rgba(37,99,235,.18)} 50%{box-shadow:0 0 0 8px rgba(37,99,235,.06)} }

.qiq-grid { display:grid; grid-template-columns: 380px minmax(0,1fr); gap:16px; align-items:start; }
@media (max-width: 1080px) { .qiq-grid { grid-template-columns: 1fr; } }
.qiq-panel {
  background: ${C.card}; border:1px solid ${C.border}; border-radius:16px; padding:18px;
  box-shadow: 0 12px 40px rgba(0,0,0,.28);
}
.qiq-right { min-height: 560px; display:flex; flex-direction:column; }

.qiq-step-num {
  width:20px; height:20px; border-radius:6px; display:grid; place-items:center;
  background: rgba(37,99,235,.16); color:#93B4FF; font-size:11px; font-weight:800;
  border:1px solid rgba(37,99,235,.3);
}
.qiq-input {
  width:100%; background:#0B1220; border:1px solid ${C.border}; border-radius:10px;
  color:${C.text}; padding:11px 12px; font-size:13.5px; font-family:inherit; outline:none;
  transition: border-color .18s, box-shadow .18s;
}
.qiq-input:focus { border-color:${C.blue}; box-shadow:0 0 0 3px rgba(37,99,235,.16); }
/* Placeholders are deliberately italic and dim so they can never be mistaken
   for text the teacher actually typed. */
.qiq-input::placeholder { color:#455066; font-style:italic; }
.qiq-textarea { min-height:150px; resize:vertical; line-height:1.65; }

.qiq-invalid { border-color:${C.red} !important; box-shadow:0 0 0 3px rgba(239,68,68,.14) !important; }
.qiq-hint { margin-top:6px; font-size:11px; color:${C.faint}; line-height:1.5; }
.qiq-mini-btn {
  background: rgba(37,99,235,.14); border:1px solid rgba(37,99,235,.34); color:#93B4FF;
  font-size:11px; font-weight:700; padding:3px 9px; border-radius:99px; cursor:pointer;
  font-family:inherit; transition:.15s; letter-spacing:.2px; white-space:nowrap;
}
.qiq-mini-btn:hover { background: rgba(37,99,235,.26); color:#C7D8FF; }

.qiq-drop {
  display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
  border:1.5px dashed ${C.borderSoft}; border-radius:14px; padding:26px 16px; cursor:pointer;
  background: linear-gradient(180deg, rgba(37,99,235,.04), transparent); transition:.2s;
}
.qiq-drop:hover, .qiq-drop.is-dragging {
  border-color:${C.blue}; background: rgba(37,99,235,.09); transform: translateY(-1px);
}
.qiq-drop-icon {
  width:40px; height:40px; border-radius:12px; display:grid; place-items:center; margin-bottom:10px;
  background: linear-gradient(135deg, rgba(37,99,235,.22), rgba(124,58,237,.22));
  border:1px solid rgba(37,99,235,.32); font-size:17px; color:#A9C3FF;
}
.qiq-file {
  display:flex; align-items:center; gap:10px; background:#0B1220;
  border:1px solid ${C.border}; border-radius:10px; padding:8px 10px;
}
.qiq-thumb {
  width:34px; height:44px; object-fit:cover; border-radius:6px;
  border:1px solid ${C.border}; flex-shrink:0; background:#fff;
}
.qiq-ellipsis { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.qiq-x {
  background:transparent; border:none; color:${C.faint}; font-size:20px; line-height:1;
  cursor:pointer; padding:2px 6px; border-radius:6px; transition:.15s;
}
.qiq-x:hover:not(:disabled) { color:${C.red}; background: rgba(239,68,68,.12); }
.qiq-x:disabled { opacity:.4; cursor:not-allowed; }

.qiq-btn {
  width:100%; margin-top:18px; display:inline-flex; align-items:center; justify-content:center; gap:9px;
  background: linear-gradient(135deg, ${C.blue}, ${C.purple}); color:#fff; border:none;
  border-radius:11px; padding:13px 16px; font-size:14px; font-weight:700; cursor:pointer;
  font-family:inherit; box-shadow:0 8px 24px rgba(37,99,235,.3); transition:.18s;
}
.qiq-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow:0 12px 30px rgba(37,99,235,.42); }
.qiq-btn:disabled { opacity:.65; cursor:progress; box-shadow:none; }
.qiq-btn-ghost {
  margin-top:0; background:#0B1220; border:1px solid ${C.border}; color:${C.dim};
  box-shadow:none; font-weight:600; font-size:12.5px; padding:10px 12px;
}
.qiq-btn-ghost:hover:not(:disabled) { color:${C.text}; border-color:${C.blue}; transform:none; box-shadow:none; }
.qiq-btn-sm { width:auto; margin-top:0; padding:9px 15px; font-size:12.5px; }
.qiq-link {
  background:none; border:none; color:#93B4FF; font-size:12.5px; cursor:pointer;
  padding:0; font-family:inherit; text-decoration:underline; text-underline-offset:3px;
}

.qiq-error {
  margin-top:16px; background: rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.32);
  color:#FCA5A5; border-radius:10px; padding:11px 13px; font-size:12.5px; line-height:1.6;
  animation: qiq-fade .25s ease;
}
@keyframes qiq-fade { from{opacity:0; transform:translateY(-4px)} to{opacity:1; transform:none} }

.qiq-spinner {
  width:14px; height:14px; border-radius:50%; display:inline-block;
  border:2px solid rgba(255,255,255,.28); border-top-color:#fff; animation: qiq-spin .7s linear infinite;
}
@keyframes qiq-spin { to { transform: rotate(360deg); } }

.qiq-empty {
  flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:60px 24px;
}
.qiq-empty-icon {
  width:64px; height:64px; border-radius:18px; display:grid; place-items:center; font-size:27px;
  background: linear-gradient(135deg, rgba(37,99,235,.16), rgba(124,58,237,.16));
  border:1px solid ${C.border}; margin-bottom:18px;
}
.qiq-orb {
  width:56px; height:56px; border-radius:50%;
  background: conic-gradient(from 0deg, ${C.blue}, ${C.purple}, ${C.blue});
  mask: radial-gradient(circle 20px at center, transparent 98%, #000 100%);
  -webkit-mask: radial-gradient(circle 20px at center, transparent 98%, #000 100%);
  animation: qiq-spin 1.1s linear infinite;
}
.qiq-proc {
  display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:11px;
  background:#0B1220; border:1px solid ${C.border}; transition:.25s;
}
.qiq-proc.is-active { border-color: rgba(37,99,235,.5); background: rgba(37,99,235,.08); }
.qiq-proc-dot {
  width:26px; height:26px; border-radius:50%; display:grid; place-items:center; flex-shrink:0;
  background:#111C33; border:1px solid ${C.border}; font-size:11px; font-weight:700; color:${C.dim};
}
.qiq-notice {
  margin-top:16px; max-width:440px; background: rgba(245,158,11,.1);
  border:1px solid rgba(245,158,11,.32); color:#FCD34D; border-radius:10px;
  padding:10px 13px; font-size:12.5px; line-height:1.6; animation: qiq-fade .25s ease;
}

/* ---------------------------------------------------------- history cards */
.qiq-hist {
  display:flex; align-items:center; gap:10px; background:#0B1220;
  border:1px solid ${C.border}; border-radius:10px; padding:8px 11px; transition:.15s;
}
.qiq-hist:hover { border-color:${C.borderSoft}; transform: translateX(2px); }
.qiq-hist-trend { font-size:15px; font-weight:800; width:14px; text-align:center; flex-shrink:0; }

/* ------------------------------------------------------------------- tabs */
.qiq-tabs { display:flex; gap:6px; border-bottom:1px solid ${C.border}; padding-bottom:2px; flex-wrap:wrap; }
.qiq-tab {
  display:inline-flex; align-items:center; gap:8px; background:transparent; border:none;
  border-bottom:2px solid transparent; color:${C.faint}; font-size:13px; font-weight:600;
  padding:10px 14px; cursor:pointer; font-family:inherit; transition:.18s; margin-bottom:-3px;
}
.qiq-tab:hover { color:${C.dim}; }
.qiq-tab.is-active { color:${C.text}; border-bottom-color:${C.blue}; }
.qiq-tab-badge {
  font-size:10.5px; font-weight:800; color:#04121F; border-radius:99px; padding:2px 7px;
}
.qiq-tab-warn {
  font-size:10.5px; font-weight:800; color:#FCD34D; background: rgba(245,158,11,.16);
  border:1px solid rgba(245,158,11,.34); border-radius:99px; padding:1px 7px;
}
.qiq-tabbody { padding-top:18px; overflow:auto; flex:1; }
.qiq-subhead {
  display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
  flex-wrap:wrap; margin-bottom:12px;
}

/* ------------------------------------------------------- annotated answer */
.qiq-annot-wrap { display:grid; grid-template-columns: minmax(0,1fr) 300px; gap:18px; align-items:start; }
@media (max-width: 900px) { .qiq-annot-wrap { grid-template-columns: 1fr; } }
.qiq-paper {
  background: linear-gradient(180deg, #0C1526, #0B1220);
  border:1px solid ${C.border}; border-radius:14px; padding:26px 28px;
  font-size:15px; line-height:2.35; white-space:pre-wrap; word-break:break-word;
  color:#DCE5F2; max-height:62vh; overflow:auto;
  background-image: repeating-linear-gradient(180deg, transparent 0 34px, rgba(148,163,184,.055) 34px 35px);
}
.qiq-hl {
  position:relative; border-radius:4px; padding:2px 1px; cursor:pointer;
  transition: filter .15s; animation: qiq-hl-in .42s cubic-bezier(.22,1,.36,1) both;
}
.qiq-hl:hover, .qiq-hl.is-active { filter: brightness(1.45); }
@keyframes qiq-hl-in {
  0%   { background-color: transparent !important; box-shadow:none !important; }
  55%  { transform: scale(1.015); }
  100% { transform: none; }
}

/* floating "+2" / "~1" / "✗0" badge pinned to the right edge of a highlight */
.qiq-marks-badge {
  display:inline-block; margin-left:5px; padding:1px 6px; border-radius:99px;
  font-size:10px; font-weight:800; color:#06101F; vertical-align:super; line-height:1.5;
  letter-spacing:.2px; box-shadow:0 2px 8px rgba(0,0,0,.4);
  animation: qiq-badge-pop .34s cubic-bezier(.34,1.56,.64,1) both .12s;
}
@keyframes qiq-badge-pop {
  0% { transform: scale(0) translateY(4px); opacity:0; }
  100% { transform: scale(1) translateY(0); opacity:1; }
}
.qiq-warn { margin-left:4px; font-size:11px; cursor:help; vertical-align:super; }
.qiq-warn.is-compact { font-size:10px; }

.qiq-pop {
  position:absolute; bottom:calc(100% + 9px); left:0; z-index:30; width:280px;
  background:#111C33; border:1px solid ${C.borderSoft}; border-radius:10px; padding:10px 12px;
  font-size:12.5px; line-height:1.6; white-space:normal; color:${C.dim};
  box-shadow:0 14px 40px rgba(0,0,0,.55); opacity:0; visibility:hidden;
  transform: translateY(4px); transition:.16s; pointer-events:none;
}
.qiq-hl:hover .qiq-pop { opacity:1; visibility:visible; transform:none; }

/* ------------------------------------------------- question paper + refs */
.qiq-drop-sm { padding:14px 16px; min-height:0; text-align:center; }

.qiq-exam { border:1px solid ${C.border}; border-radius:12px; background:#0B1220; padding:12px 13px; }
.qiq-exam-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.qiq-exam-total { text-align:right; flex-shrink:0; line-height:1.2; }
.qiq-exam-rows { display:grid; gap:4px; margin-top:11px; max-height:230px; overflow:auto; }
.qiq-exam-row {
  display:grid; grid-template-columns: 34px minmax(0,1fr) auto 48px; gap:7px;
  align-items:center; font-size:11.5px; padding:5px 6px; border-radius:7px; background:#0F172A;
}
.qiq-exam-row.is-missing { background: rgba(245,158,11,.10); }
.qiq-exam-num { font-weight:800; color:${C.text}; }
.qiq-exam-text { color:${C.dim}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.qiq-exam-type { color:${C.faint}; font-size:10px; text-transform:uppercase; letter-spacing:.4px; }
.qiq-exam-marks {
  width:46px; background:#0A0F1E; border:1px solid ${C.borderSoft}; border-radius:6px;
  color:${C.text}; font-size:11.5px; padding:3px 5px; text-align:center; font-family:inherit;
}
.qiq-exam-row.is-missing .qiq-exam-marks { border-color:${C.amber}; }
.qiq-exam-warn { margin-top:10px; display:grid; gap:6px; }
.qiq-exam-warnrow {
  display:flex; gap:7px; align-items:flex-start; font-size:11.5px; line-height:1.55;
  color:${C.text}; background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.28);
  border-radius:8px; padding:7px 9px;
}

.qiq-reflist { display:grid; gap:5px; margin-top:8px; }
.qiq-refitem {
  display:flex; gap:8px; align-items:center; font-size:11.5px; color:${C.dim};
  background:#0B1220; border:1px solid ${C.border}; border-radius:8px; padding:6px 9px;
}
.qiq-reficon { flex-shrink:0; }

.qiq-ground {
  display:inline-block; font-size:9.5px; font-weight:800; letter-spacing:.4px;
  text-transform:uppercase; padding:2px 6px; border-radius:99px; vertical-align:middle;
}
.qiq-ground.is-ref { background:rgba(34,197,94,.16); color:${C.green}; }
.qiq-ground.is-general { background:rgba(148,163,184,.14); color:${C.dim}; }
.qiq-ground.is-insufficient { background:rgba(245,158,11,.16); color:${C.amber}; }

/* ----------------------------------------------------------- marked paper */
/* The page image is the base layer and every mark is positioned in percentages
   of it, so the overlay stays aligned at any width without measuring anything
   in JavaScript. */
.qiq-pages { display:grid; gap:20px; max-height:62vh; overflow:auto; padding-right:4px; }
.qiq-page-block { margin:0; }
.qiq-page-stage {
  position:relative; line-height:0; border-radius:12px;
  border:1px solid ${C.border}; background:#0B1220; overflow:hidden;
}
.qiq-page-img { display:block; width:100%; height:auto; }
.qiq-page-cap {
  display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;
  font-size:11px; color:${C.dim}; padding:7px 3px 0; line-height:1.5;
}

.qiq-ann-box {
  position:absolute; border-radius:4px; cursor:pointer; min-height:12px;
  transition: filter .15s, transform .15s;
  animation: qiq-box-in .42s cubic-bezier(.22,1,.36,1) both;
}
.qiq-ann-box:hover, .qiq-ann-box.is-active { filter: brightness(1.5); transform: scale(1.012); }
.qiq-ann-box.is-active { outline:2px solid rgba(226,232,240,.55); outline-offset:1px; }
@keyframes qiq-box-in {
  0%   { opacity:0; background-color:transparent !important; box-shadow:none !important; }
  55%  { opacity:1; transform: scale(1.03); }
  100% { opacity:1; transform:none; }
}

/* Badge rides just outside the right edge of the marked region, where a teacher
   would put it, and falls inside when the writing runs to the page edge. */
.qiq-ann-badge {
  position:absolute; left:100%; top:50%; transform:translate(6px,-50%);
  padding:1px 6px; border-radius:99px; font-size:10px; font-weight:800;
  color:#06101F; line-height:1.5; letter-spacing:.2px; white-space:nowrap;
  box-shadow:0 2px 8px rgba(0,0,0,.45);
  animation: qiq-badge-pop .34s cubic-bezier(.34,1.56,.64,1) both .12s;
}
/* Writing that runs to the page edge leaves no room outside it, so the badge
   tucks back inside rather than being clipped by the stage. */
.qiq-ann-box.is-tight .qiq-ann-badge { left:auto; right:3px; transform:translate(0,-50%); }
.qiq-ann-warn {
  position:absolute; right:100%; top:50%; transform:translate(-5px,-50%);
  font-size:11px; line-height:1; pointer-events:none;
}
.qiq-ann-box.is-tight .qiq-ann-warn { right:auto; left:3px; transform:translate(0,-50%); }

/* Opens downward: on a page, the space under a line is free, and a comment
   above would cover the writing it refers to. Clicking still scrolls the full
   remark into the margin, so a clipped popover never hides anything. */
.qiq-ann-box .qiq-pop { top:calc(100% + 8px); bottom:auto; width:min(280px, 62vw); }
.qiq-ann-box:hover .qiq-pop { opacity:1; visibility:visible; transform:none; }

.qiq-geom-note {
  display:flex; gap:9px; align-items:flex-start; margin-bottom:14px;
  background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.3);
  border-radius:10px; padding:10px 12px; font-size:12px; line-height:1.6; color:${C.text};
}
.qiq-linkbtn {
  background:none; border:none; padding:0; font:inherit; color:${C.blue};
  text-decoration:underline; cursor:pointer;
}
.qiq-linkbtn:hover { color:#60A5FA; }

.qiq-marking-dot {
  display:inline-block; width:7px; height:7px; border-radius:50%; margin-left:8px;
  background:${C.blue}; animation: qiq-blink 1s infinite;
}
@keyframes qiq-blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.qiq-margin { display:grid; gap:10px; max-height:62vh; overflow:auto; padding-right:4px; align-content:start; }
.qiq-note { background:#0B1220; border:1px solid ${C.border}; border-radius:10px; padding:10px 12px; transition:.18s; }
.qiq-note.is-active { background:#111C33; border-color:${C.borderSoft}; transform: translateX(-3px); }
.qiq-note.is-unsure { background: rgba(245,158,11,.05); }
.qiq-note-enter { animation: qiq-note-in .44s cubic-bezier(.22,1,.36,1) both; }
@keyframes qiq-note-in {
  0%   { opacity:0; transform: translateX(38px); }
  100% { opacity:1; transform: translateX(0); }
}
.qiq-note-num {
  width:17px; height:17px; border-radius:50%; display:grid; place-items:center;
  font-size:9.5px; font-weight:800; color:#06101F;
}
.qiq-note-marks {
  padding:1px 7px; border-radius:99px; font-size:10px; font-weight:800; color:#06101F;
}
/* The teacher's comment. This used to be set in a cursive "handwriting" face to
   suggest a marked-up page, but the comment is the single most important thing
   a student reads here, and a script face at 14px on a dark background is hard
   work — especially for the long, specific feedback the new per-question marking
   produces. The marked-up feel is carried by the colour, the left rule and the
   badges instead; the words themselves are plain. */
.qiq-handwrite {
  font-family: inherit;
  font-size: 13.5px;
  line-height: 1.65;
  letter-spacing: 0.1px;
  color: ${C.text};
}

/* -------------------------------------------------------------- report card */
.qiq-report-actions {
  display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:16px;
}
.qiq-report {
  background: linear-gradient(180deg, #0E1A2F, #0B1220);
  border:1px solid ${C.border}; border-radius:16px; padding:26px 28px;
}
.qiq-report-head {
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding-bottom:16px; border-bottom:2px solid ${C.border};
}
.qiq-report-brand { display:flex; align-items:center; gap:12px; }
.qiq-report-logo {
  width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
  background: linear-gradient(135deg, ${C.blue}, ${C.purple}); color:#fff;
  font-weight:800; font-size:21px;
}
.qiq-report-title { font-size:19px; font-weight:800; letter-spacing:-.3px; }
.qiq-report-sub { font-size:11.5px; color:${C.faint}; margin-top:2px; }
.qiq-report-serial {
  font-size:11px; color:${C.faint}; letter-spacing:1.4px; text-transform:uppercase;
  font-variant-numeric: tabular-nums;
}

.qiq-report-fields {
  display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:16px; margin:20px 0 4px;
}
@media (max-width: 780px) { .qiq-report-fields { grid-template-columns: 1fr; } }
.qiq-rfield { display:block; }
.qiq-rfield > span {
  display:block; font-size:10px; letter-spacing:1.1px; text-transform:uppercase;
  color:${C.faint}; font-weight:700; margin-bottom:5px;
}
.qiq-rinput {
  width:100%; background:transparent; border:none; border-bottom:1.5px solid ${C.border};
  color:${C.text}; font-family:inherit; font-size:15px; font-weight:600;
  padding:5px 2px; outline:none; transition:border-color .18s;
}
.qiq-rinput:focus { border-bottom-color:${C.blue}; }
.qiq-rinput::placeholder { color:#3D4A61; }

.qiq-report-score {
  display:flex; align-items:center; justify-content:space-around; gap:28px; flex-wrap:wrap;
  margin:26px 0; padding:24px 20px; border-radius:14px;
  background: linear-gradient(135deg, rgba(37,99,235,.10), rgba(124,58,237,.08));
  border:1px solid ${C.border};
}
.qiq-arc-wrap { position:relative; width:152px; height:152px; flex-shrink:0; }
.qiq-arc { transform: rotate(-90deg); }
.qiq-arc-track { fill:none; stroke:#152238; stroke-width:11; }
.qiq-arc-fill { fill:none; stroke-width:11; stroke-linecap:round; }
.qiq-arc-label {
  position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center;
}
.qiq-arc-score { font-size:33px; font-weight:800; letter-spacing:-1.2px; font-variant-numeric: tabular-nums; }
.qiq-arc-total { font-size:16px; font-weight:600; color:${C.faint}; letter-spacing:0; }
.qiq-arc-pct { font-size:11px; color:${C.faint}; margin-top:2px; letter-spacing:.6px; }

.qiq-grade-block { text-align:center; }
.qiq-grade-stamp {
  width:104px; height:104px; border-radius:26px; display:grid; place-items:center;
  font-size:40px; font-weight:800; border:3px solid; background:#0B1220; letter-spacing:-1.5px;
  animation: qiq-stamp .6s cubic-bezier(.34,1.56,.64,1) both .3s;
}
@keyframes qiq-stamp {
  0%   { transform: scale(0) rotate(-14deg); opacity:0; }
  60%  { transform: scale(1.1) rotate(3deg); opacity:1; }
  100% { transform: scale(1) rotate(0deg); opacity:1; }
}
.qiq-grade-caption {
  font-size:10px; letter-spacing:1.2px; text-transform:uppercase;
  color:${C.faint}; margin-top:10px; font-weight:700;
}

/* the rubber stamp — deliberately off-axis like a real one */
.qiq-passmark {
  width:126px; height:126px; border-radius:50%; display:grid; place-items:center;
  font-size:25px; font-weight:900; letter-spacing:2.5px; border:5px double currentColor;
  transform: rotate(-3deg); opacity:.92; flex-shrink:0;
  animation: qiq-stamp-in .5s cubic-bezier(.34,1.56,.64,1) both .55s;
}
.qiq-passmark.is-pass { color:${C.green}; background: rgba(34,197,94,.07); }
.qiq-passmark.is-fail { color:${C.red}; background: rgba(239,68,68,.07); }
@keyframes qiq-stamp-in {
  0%   { transform: scale(0) rotate(-24deg); opacity:0; }
  60%  { transform: scale(1.1) rotate(1deg); opacity:1; }
  100% { transform: scale(1) rotate(-3deg); opacity:.92; }
}

.qiq-report-section { margin-top:26px; }
.qiq-report-h {
  display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  font-size:11px; letter-spacing:1.2px; text-transform:uppercase;
  color:${C.dim}; font-weight:800; margin-bottom:12px;
}
.qiq-voice { display:inline-flex; align-items:center; gap:6px; margin-left:auto; }
.qiq-speaking {
  font-size:10.5px; color:${C.blue}; text-transform:none; letter-spacing:0;
  animation: qiq-blink 1.4s infinite;
}

.qiq-remark {
  margin:0; padding:20px 24px; border-radius:14px; cursor:default;
  background: linear-gradient(135deg, rgba(124,58,237,.10), rgba(37,99,235,.06));
  border:1px solid ${C.border}; border-left:3px solid ${C.purple};
  /* Upright, not italic: this is several sentences of real feedback, and a long
     italic run is markedly slower to read on screen. The serif and the size keep
     it feeling like a written remark rather than UI chrome. */
  font-family: Georgia, "Times New Roman", serif;
  font-size:17px; line-height:1.75; color:#F1F5F9; min-height:96px;
}
.qiq-remark-sign {
  margin-top:14px; text-align:right; font-size:13px; color:${C.faint}; font-style:italic;
}
.qiq-caret {
  display:inline-block; width:2px; height:1em; background:${C.purple};
  margin-left:2px; vertical-align:-2px; animation: qiq-blink .8s steps(1) infinite;
}

.qiq-table-wrap { overflow-x:auto; border:1px solid ${C.border}; border-radius:12px; }
.qiq-table { width:100%; border-collapse:collapse; font-size:13px; min-width:640px; }
.qiq-table th {
  text-align:left; font-size:10.5px; letter-spacing:.8px; text-transform:uppercase;
  color:${C.dim}; font-weight:700; padding:11px 14px; background:#0B1220;
  border-bottom:1px solid ${C.border};
}
.qiq-table td { padding:13px 14px; border-bottom:1px solid rgba(30,41,59,.7); vertical-align:top; }
.qiq-table tbody tr:hover { background: rgba(37,99,235,.05); }
.qiq-table tfoot td { background:#0B1220; border-bottom:none; font-size:13.5px; }
.qiq-chip {
  display:inline-block; font-size:11px; font-weight:700; padding:3px 9px;
  border-radius:99px; border:1px solid; white-space:nowrap;
}

.qiq-two-col { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:26px; }
@media (max-width: 760px) { .qiq-two-col { grid-template-columns: 1fr; } }
.qiq-listcard { background:#0B1220; border:1px solid ${C.border}; border-radius:12px; padding:16px 18px; }
.qiq-list { list-style:none; padding:0; margin:0; display:grid; gap:9px; }
.qiq-list li { display:flex; gap:9px; font-size:13px; line-height:1.6; color:${C.dim}; }
.qiq-study {
  color:#93B4FF; text-decoration:none; font-size:11.5px; font-weight:700;
  border-bottom:1px dotted #93B4FF; white-space:nowrap; margin-left:2px;
}
.qiq-study:hover { color:#C7D8FF; border-bottom-style:solid; }

.qiq-report-foot {
  margin-top:24px; padding-top:14px; border-top:1px solid ${C.border};
  font-size:10.5px; color:${C.faint}; line-height:1.6;
}

.qiq-mono {
  width:100%; min-height:420px; background:#0B1220; border:1px solid ${C.border};
  border-radius:12px; color:#CBD5E1; padding:18px 20px; resize:vertical; outline:none;
  font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size:13px; line-height:1.8; white-space:pre-wrap;
}
.qiq-mono:focus { border-color:${C.blue}; box-shadow:0 0 0 3px rgba(37,99,235,.16); }
.qiq-pre {
  margin-top:10px; background:#0B1220; border:1px solid ${C.border}; border-radius:10px;
  padding:14px; font-size:11.5px; color:${C.faint}; max-height:280px; overflow:auto;
  white-space:pre-wrap; word-break:break-word;
  font-family: ui-monospace, "SF Mono", Consolas, monospace;
}

.qiq-root ::-webkit-scrollbar { width:9px; height:9px; }
.qiq-root ::-webkit-scrollbar-track { background:transparent; }
.qiq-root ::-webkit-scrollbar-thumb { background:#233149; border-radius:99px; }
.qiq-root ::-webkit-scrollbar-thumb:hover { background:#2F4364; }

/* Respect the OS setting: no reveal, no stamp, no typing. */
@media (prefers-reduced-motion: reduce) {
  .qiq-root *, .qiq-root *::before, .qiq-root *::after {
    animation-duration: .001ms !important; animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }
}

/* ============================================================== PRINT ==== */
/* Everything is hidden and only the report is re-shown, so a stray wrapper can
   never drag the dark theme onto the page. */
@media print {
  @page { margin: 14mm; }

  html, body { background:#fff !important; }
  body * { visibility: hidden !important; }
  .qiq-report, .qiq-report * { visibility: visible !important; }
  .qiq-noprint, .qiq-noprint * { display:none !important; }

  .qiq-root { background:#fff !important; padding:0 !important; }

  .qiq-report {
    position:absolute !important; left:0; top:0; width:100%;
    background:#fff !important; color:#000 !important;
    border:none !important; border-radius:0 !important; padding:0 !important;
    box-shadow:none !important;
  }
  .qiq-report * { color:#000 !important; text-shadow:none !important; }

  .qiq-report-head { border-bottom:2px solid #000 !important; }
  .qiq-report-logo { background:#000 !important; color:#fff !important; }
  .qiq-report-logo * { color:#fff !important; }

  .qiq-report-score {
    background:none !important; border:1px solid #000 !important;
    page-break-inside: avoid; break-inside: avoid;
  }
  .qiq-arc-track { stroke:#DDD !important; }
  .qiq-arc-fill  { stroke:#000 !important; }
  .qiq-grade-stamp { border-color:#000 !important; background:#fff !important; }
  .qiq-passmark { border-color:#000 !important; background:#fff !important; opacity:1 !important; }

  /* Inputs must read as filled-in fields, not as form controls. */
  .qiq-rinput {
    border-bottom:1px solid #000 !important; background:transparent !important;
    color:#000 !important; -webkit-text-fill-color:#000 !important;
  }

  .qiq-remark {
    background:none !important; border:1px solid #000 !important;
    border-left:3px solid #000 !important; page-break-inside: avoid; break-inside: avoid;
  }
  .qiq-caret { display:none !important; }

  .qiq-table-wrap, .qiq-listcard { border:1px solid #000 !important; }
  .qiq-table th { background:#EEE !important; border-bottom:1px solid #000 !important; }
  .qiq-table td { border-bottom:1px solid #CCC !important; }
  .qiq-table tfoot td { background:#EEE !important; }
  .qiq-chip { border:1px solid #000 !important; background:none !important; }
  .qiq-two-col { page-break-inside: avoid; break-inside: avoid; }
  .qiq-report-foot { border-top:1px solid #000 !important; }

  /* Animations must not leave anything mid-flight on paper. */
  .qiq-report *, .qiq-report *::before, .qiq-report *::after {
    animation: none !important; transition: none !important; transform: none !important;
    opacity: 1 !important;
  }
}
`;
