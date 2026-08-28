import { useState, useEffect, useRef } from "react";

import { structureOcr, linesToText } from "./src/engine/ocr.js";
import { groqChat, GroqError, GROQ_BASE, OCR_MODEL, EVAL_MODEL } from "./src/engine/groq.js";
import { createLlm } from "./src/engine/llm.js";
import { loadPdfJs, extractPdfText } from "./src/engine/pdf.js";
import { isDocx, isLegacyDoc, extractDocxText, LEGACY_DOC_MESSAGE } from "./src/engine/docx.js";
import { extractExamWithLlm, validateExam, structuralMarks, deriveMarks } from "./src/engine/exam.js";
import { chunkDocument, createRetriever } from "./src/engine/reference.js";
import { matchAnswers } from "./src/engine/match.js";
import { applyChoice, describeChoice } from "./src/engine/choice.js";
import {
  assessPaper,
  summarisePaper,
  toEvaluation,
  gradeFor,
  answerStatus,
  ANSWER_STATUS,
  GROUNDING,
  pairReferenceAnswers,
} from "./src/engine/assess.js";
import { extractJson } from "./src/engine/json.js";
import { normalizeText } from "./src/engine/text.js";
import { unionBoxes } from "./src/engine/geometry.js";
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
/* An answer booklet runs longer than a handful of pages, and a paper cut short
   is a paper marked wrong. The cap exists only to bound browser memory; when it
   bites, the upload says so instead of quietly losing the tail. */
const MAX_PAGES = 30;

/* ------------------------------------------------------------ presentation */
const REVEAL_MS = 300; // gap between each highlight appearing
const TYPE_MS = 18; // per character of the teacher's remark
const COUNT_MS = 1500; // marks counter run time
const LOW_CONFIDENCE = 60; // below this the mark is flagged for manual review
const PASS_THRESHOLD = 0.4; // 40% and above earns the PASS stamp

const HISTORY_KEY = "qiq.paperchecker.history.v1";
const HISTORY_LIMIT = 5;

/* Three things downstream depend on this transcript, and each one used to be
   left to chance:
     · the question labels the student wrote are the strongest route to matching
       an answer to its question (95% confidence against ~70% for a model guess);
     · one output line per written line is what the ink measurement is aligned
       against — reflowed prose makes the line count disagree with the bands and
       the marks lose their place on the page;
     · "[illegible]" is what the legibility score in ocr.js reads, and nothing
       was asking the transcriber to produce it, so a guessed word and a clearly
       read one scored the same. */
const OCR_SYSTEM =
  "You are an OCR engine transcribing a student's handwritten answer paper. Follow these rules:\n" +
  "- Transcribe exactly what is written. Do not correct spelling or grammar, do not summarise, " +
  "do not explain, do not add commentary.\n" +
  "- Output ONE line of text for each line written on the page, in reading order. Do not merge " +
  "lines and do not re-wrap them: this transcript is aligned against the page image line by line.\n" +
  "- Keep the student's own question numbers and labels exactly as written and at the start of " +
  "the line they appear on: \"1.\", \"Q3\", \"(a)\", \"iii)\".\n" +
  "- Keep mathematics, equations, units and symbols as written, in plain text.\n" +
  "- Where handwriting is genuinely unreadable, write [illegible] in place of that word. Never " +
  "invent a word to fill a gap.\n" +
  "- Add nothing the student did not write — no page headers, no numbering, no notes.\n" +
  "Return only the transcript.";

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
  dim: "#B5C0D1",
  faint: "#8492A8",
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
/* The badge states a mark only when there is one to state. An annotation the
   model gave no marks for used to render as "+0" — a correct point stamped with
   a zero it never earned. With no number the verdict shows on its own. */
function marksBadge(ann) {
  const style = typeStyle(ann.type);
  const n = Number(ann && ann.marks);
  if (!Number.isFinite(n) || n === 0) return style.icon;
  return style.sign + Math.abs(n);
}

const hasMarks = (ann) => {
  const n = Number(ann && ann.marks);
  return Number.isFinite(n) && n !== 0;
};

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
  // The caller is told the true length so a truncated upload can be reported.

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
  return { pages, total: doc.numPages };
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
  const steps = ["Upload", "Read Paper", "Mark Answers", "Results"];
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
                fontSize: 15,
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
          <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 14, color: C.dim }}>
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

function SectionTitle({ n, title, action, compactTop = false }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, margin: `${compactTop ? -8 : 20}px 0 10px` }}>
      <span className="qiq-step-num">{n}</span>
      <span style={{ fontSize: 15, fontWeight: 750, letterSpacing: 0.2, color: C.text }}>
        {title}
      </span>
      {action && <span style={{ marginLeft: "auto" }}>{action}</span>}
    </div>
  );
}

/** Quiet label shown wherever a mark needs the teacher's attention. */
function ConfidenceFlag({ confidence, compact }) {
  return (
    <span
      className={`qiq-warn${compact ? " is-compact" : ""}`}
      title={`Please verify this mark manually (certainty ${Math.round(
        Number(confidence) || 0
      )}%)`}
    >
      {compact ? "Check" : "Needs review"}
    </span>
  );
}

/* ============================================================ MAIN APP ===== */

export default function PaperChecker() {
  const [pages, setPages] = useState([]); // { id, label, dataUrl }
  /* A Word upload carries digital text, so it skips rasterising and the vision
     pass entirely — `textDoc` holds it until Check Paper runs. */
  const [textDoc, setTextDoc] = useState(null); // { name, text } | null
  const [dragging, setDragging] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [totalMarks, setTotalMarks] = useState("20");

  /* Report-card identity fields — kept at this level so the history log and the
     printed report agree, and so they survive tab switches. */
  const [studentName, setStudentName] = useState("");
  const [subject, setSubject] = useState("");
  const [reportDate, setReportDate] = useState(todayLabel);
  const [studentDetails, setStudentDetails] = useState([]);
  const [markNotice, setMarkNotice] = useState("");

  const [studentAnswerText, setStudentAnswerText] = useState("");
  /* The transcript is held per page, because the page boundaries are what the
     ink measurements are attached to. Holding it as one string meant an edit on
     a multi-page paper destroyed those boundaries and the whole overlay was
     dropped for the re-grade. The joined form below is derived, never stored. */
  const [editedPages, setEditedPages] = useState([]);
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

  /* The run's own account of itself. Every entry is something that actually
     happened — a page read, a question matched, a mark decided — with the
     number that came back from it. It is written as the work happens, so what
     the examiner watches is the pipeline, not an animation of one. */
  /* Uploads that did not fit, kept on screen rather than flashed once: a paper
     that lost pages is marked on less than the student wrote, and the examiner
     has to know before the marks are read. */
  const [truncated, setTruncated] = useState([]);

  const [trace, setTrace] = useState([]);
  const traceStart = useRef(0);
  const step = (text, detail = "", kind = "info") =>
    setTrace((t) =>
      t.concat({
        at: (Date.now() - traceStart.current) / 1000,
        text,
        detail,
        kind,
      })
    );
  const beginTrace = () => {
    traceStart.current = Date.now();
    setTrace([]);
  };
  const [geometryByPage, setGeometryByPage] = useState({});

  /* The question paper drives everything: it says what was asked, how many
     questions there are and what each is worth. When one is supplied the
     question-paper-first pipeline runs; without one the original
     marking-scheme flow is used unchanged, so nothing that worked stops
     working. */
  const [examPages, setExamPages] = useState([]);
  const [exam, setExam] = useState(null);
  const [examBusy, setExamBusy] = useState(false);
  const [examSources, setExamSources] = useState([]);

  /* Reference material, chunked and indexed locally. */
  const [refFiles, setRefFiles] = useState([]); // { name, chunks, pageCount }
  const [refChunks, setRefChunks] = useState([]);
  /* The reference material's own text, kept beside the search index. A textbook
     is searched by topic; a set of model answers is filed by question number,
     and that number is the strongest link there is between what was asked and
     what a full-mark answer says. */
  const [refText, setRefText] = useState("");
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

  /* The examiner's decisions. The AI proposes a mark per key point; the human
     final mark is whatever the examiner sets, falling back to the proposal.
     Keyed by key-point index, reset with every new result. */
  const [markOverrides, setMarkOverrides] = useState({});

  /* Which question the examiner is inspecting on the page. This is the single
     source of truth for the page viewer: the page, the visible annotation
     boxes and the margin notes are all derived from it — nothing keeps its own
     copy that could go stale. null means "show everything". */
  const [selectedQuestionId, setSelectedQuestionId] = useState(null);

  /* Real pipeline progress. `phase` moves only when the corresponding work
     actually runs; `coverageInfo` is filled the moment answer matching
     finishes, so the processing screen can say "3 of 5 answers detected"
     while marking is still underway. */
  const [phase, setPhase] = useState("idle"); // ocr | measure | match | mark | review
  const [coverageInfo, setCoverageInfo] = useState(null);

  /* The history row belonging to the result on screen, so examiner mark
     adjustments can be written back to it without disturbing older entries. */
  const [activeHistoryId, setActiveHistoryId] = useState(null);

  const busy = stage === "ocr" || stage === "evaluating";

  useEffect(() => {
    if (!busy) return undefined;
    setElapsed(0);
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [busy]);

  const pipelineStep =
    stage === "ocr" ? 1 : stage === "evaluating" ? 2 : stage === "done" ? 3 : pages.length || textDoc ? 1 : 0;

  /* --------------------------------------------------------------- derive -- */
  const editedText = editedPages.join("\n\n");

  const annotations = evaluation && Array.isArray(evaluation.annotations) ? evaluation.annotations : [];
  const keyPoints = evaluation && Array.isArray(evaluation.keyPoints) ? evaluation.keyPoints : [];

  /* Human-adjusted marks win wherever the examiner set one. The AI total only
     survives untouched when there is no per-point breakdown to adjust. */
  const overrideCount = Object.keys(markOverrides).length;
  const keyPointAwarded = (k, i) =>
    Number.isFinite(markOverrides[i]) ? markOverrides[i] : Number(k.marksAwarded) || 0;

  /* "Answer any 3 of the following 5" is re-applied here, not just at marking
     time, because the examiner can change a mark — and if they raise the fourth
     attempt above the third, it is the fourth that should now count. The choice
     follows the marks on screen. */
  const choiceGroups = (evaluation && evaluation.choice) || [];
  const chosen = choiceGroups.length
    ? applyChoice(
        keyPoints.map((k, i) => ({
          questionId: k.questionId,
          number: k.questionNumber,
          maxMarks: k.marksTotal,
          marksAwarded: keyPointAwarded(k, i),
          skipped: k.skipped,
          failed: k.failed,
        })),
        choiceGroups
      )
    : null;
  const isCounted = (k) => !chosen || chosen.counted.has(k.questionId);

  const scoreTotal = chosen
    ? chosen.maximumMarks
    : Number(evaluation && evaluation.totalMarks) || Number(totalMarks) || 0;
  const scoreAwarded = !evaluation
    ? 0
    : keyPoints.length
    ? keyPoints.reduce((sum, k, i) => sum + (isCounted(k) ? keyPointAwarded(k, i) : 0), 0)
    : Number(evaluation.totalMarksAwarded) || 0;

  /* Grade follows the marks on screen. Untouched results keep the engine's
     grade (which came from the same scale); any override recomputes it. */
  const effectiveGrade = !evaluation
    ? "—"
    : overrideCount > 0 && scoreTotal > 0
    ? gradeFor((scoreAwarded / scoreTotal) * 100)
    : evaluation.grade || "—";

  /* What the report card and the review list render: the AI proposal, plus the
     examiner's mark where one was given. `aiMarks` keeps the proposal visible
     next to the examiner's decision. */
  const displayKeyPoints = keyPoints.map((k, i) => {
    const withOverride = Number.isFinite(markOverrides[i])
      ? { ...k, aiMarks: k.marksAwarded, marksAwarded: markOverrides[i], overridden: true }
      : k;
    return chosen ? { ...withOverride, counted: isCounted(k) } : withOverride;
  });

  /* The selected question, derived from one id — never stored twice. The page
     viewer, its boxes and its margin notes all read from this single lookup,
     so a stale copy of "which question am I showing" cannot exist. */
  const paperQuestions =
    evaluation && evaluation.paper && Array.isArray(evaluation.paper.questions)
      ? evaluation.paper.questions
      : [];
  const selectedQuestion = selectedQuestionId
    ? paperQuestions.find((q) => q.questionId === selectedQuestionId) || null
    : null;
  const selectedPages =
    selectedQuestion && Number.isFinite(selectedQuestion.pageStart)
      ? Array.from(
          {
            length:
              (Number.isFinite(selectedQuestion.pageEnd)
                ? selectedQuestion.pageEnd
                : selectedQuestion.pageStart) -
              selectedQuestion.pageStart +
              1,
          },
          (_, i) => selectedQuestion.pageStart + i
        )
      : null;

  /* Where the selected question's answer physically sits, page by page: the
     union of its own OCR lines' measured boxes. It is the same measurement the
     annotation boxes are drawn from, so the outline and the marks inside it
     cannot disagree — and a page whose lines were never measured simply
     contributes no region rather than a guessed one. */
  const selectedIndex = selectedQuestion ? paperQuestions.indexOf(selectedQuestion) : -1;
  const selectedRegions =
    selectedQuestion && ocrDoc && Array.isArray(selectedQuestion.lineIds)
      ? (() => {
          const ids = new Set(selectedQuestion.lineIds);
          const byPage = new Map();
          for (const line of ocrDoc.lines) {
            if (!ids.has(line.id) || !line.bbox) continue;
            if (!byPage.has(line.page)) byPage.set(line.page, []);
            byPage.get(line.page).push(line.bbox);
          }
          return Array.from(byPage.entries())
            .map(([page, boxes]) => ({ page, bbox: unionBoxes(boxes) }))
            .filter((r) => r.bbox)
            .sort((a, b) => a.page - b.page);
        })()
      : [];

  /* The mark shown beside the question is the one that counts: the examiner's
     where they set one, the AI's proposal otherwise. */
  const selectedMark =
    selectedIndex >= 0 && Number.isFinite(markOverrides[selectedIndex])
      ? markOverrides[selectedIndex]
      : selectedQuestion
      ? selectedQuestion.marksAwarded
      : null;

  const viewQuestionOnPage = (k) => {
    if (!k.questionId) return;
    setSelectedQuestionId(k.questionId);
    setTab("paper");
  };

  /* A new result invalidates the previous paper's decisions and selection. */
  useEffect(() => {
    setMarkOverrides({});
    setSelectedQuestionId(null);
  }, [evalRun]);

  /* The examiner's marks are the final ones — write them back to this paper's
     history row, keeping the AI's original proposal alongside so the log never
     blurs who decided what. */
  useEffect(() => {
    if (!activeHistoryId || !evaluation) return;
    setHistory((prev) => {
      const idx = prev.findIndex((h) => h.id === activeHistoryId);
      if (idx === -1) return prev;
      const cur = prev[idx];
      const aiScore = Number.isFinite(cur.aiScore) ? cur.aiScore : cur.score;
      const finalScore = overrideCount > 0 ? scoreAwarded : aiScore;
      if (cur.score === finalScore && (cur.overrides || 0) === overrideCount && cur.grade === effectiveGrade)
        return prev;
      const next = prev.slice();
      next[idx] = { ...cur, aiScore, score: finalScore, overrides: overrideCount, grade: effectiveGrade };
      saveHistory(next);
      return next;
    });
  }, [scoreAwarded, overrideCount, effectiveGrade, activeHistoryId, evaluation]);

  const anchored = evaluation ? anchorAnnotations(studentAnswerText, annotations) : null;
  const lowConfidenceCount = annotations.filter(isLowConfidence).length;

  /* What the Evaluate tab counts on its badge: questions the machine could not
     settle (skipped, failed, low-confidence) plus pipeline warnings such as
     unregistered writing. These are the rows the examiner must look at first. */
  const reviewIssueCount = !evaluation
    ? 0
    : keyPoints.filter(
        (k) => k.skipped || k.failed || (Number.isFinite(k.confidence) && k.confidence < LOW_CONFIDENCE)
      ).length +
      (evaluation.paper && Array.isArray(evaluation.paper.warnings)
        ? evaluation.paper.warnings.length
        : 0);

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
    const cut = [];

    for (const file of incoming) {
      if (isLegacyDoc(file)) {
        setError(LEGACY_DOC_MESSAGE(file.name));
        continue;
      }
      if (isDocx(file)) {
        /* Digital text, not ink: read it directly instead of paying a vision
           pass to transcribe pixels. One Word document is the whole answer
           paper, so the newest upload replaces the previous one. */
        try {
          const text = await extractDocxText(file);
          setTextDoc({ name: file.name, text });
        } catch (e) {
          setError(e.message || `Could not read "${file.name}".`);
        }
        continue;
      }
      const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
      const isImage = ACCEPTED_IMAGE_TYPES.includes(file.type);
      if (!isPdf && !isImage) {
        setError(`"${file.name}" is not supported. Upload a PDF, a Word (.docx) file, or a JPG / PNG / GIF / WEBP image.`);
        continue;
      }
      try {
        const produced = isPdf ? await rasterizePdf(file) : { pages: await rasterizeImage(file), total: 1 };
        if (produced.total > produced.pages.length)
          cut.push({ scope: "answer", name: file.name, kept: produced.pages.length, total: produced.total });
        collected.push(...produced.pages);
      } catch (e) {
        setError(e.message || `Could not read "${file.name}".`);
      }
    }

    if (collected.length) {
      const stamp = String(Date.now());
      setPages((prev) => {
        const merged = prev.concat(collected.map((p, i) => ({ ...p, id: `${stamp}-${i}` })));
        if (merged.length > MAX_PAGES) {
          cut.push({ scope: "answer", name: "this upload", kept: MAX_PAGES, total: merged.length });
          return merged.slice(0, MAX_PAGES);
        }
        return merged;
      });
    }
    setTruncated((prev) => prev.filter((x) => x.scope !== "answer").concat(cut));
    setPreparing(false);
  }

  const removePage = (id) => setPages((prev) => prev.filter((p) => p.id !== id));

  function resetAll() {
    setPages([]);
    setTextDoc(null);
    setStudentAnswerText("");
    setEditedPages([]);
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
    setTruncated([]);
    setTrace([]);
    setTab("paper");
  }

  /* ------------------------------------------------------------ pipeline -- */

  /* ------------------------------------------------- question paper ------ */

  /** Read the question paper and turn it into the exam structure. */
  async function ingestExam(files) {
    setError("");
    setExamBusy(true);
    try {
      const incoming = Array.from(files || []);
      const collected = [];
      const digitalTexts = [];
      const cut = [];
      for (const file of incoming) {
        if (isLegacyDoc(file)) {
          setError(LEGACY_DOC_MESSAGE(file.name));
          continue;
        }
        if (isDocx(file)) {
          /* A Word question paper already carries its text — read it directly
             instead of spending a vision pass per page. */
          digitalTexts.push(await extractDocxText(file));
          continue;
        }
        const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
        const isImage = ACCEPTED_IMAGE_TYPES.includes(file.type);
        if (!isPdf && !isImage) {
          setError(`"${file.name}" is not a supported question paper. Upload a PDF, a Word (.docx) file, or an image.`);
          continue;
        }
        const produced = isPdf ? await rasterizePdf(file) : { pages: await rasterizeImage(file), total: 1 };
        if (produced.total > produced.pages.length)
          cut.push({ scope: "question", name: file.name, kept: produced.pages.length, total: produced.total });
        collected.push(...produced.pages);
      }
      setTruncated((prev) => prev.filter((x) => x.scope !== "question").concat(cut));
      if (collected.length === 0 && digitalTexts.length === 0) return;
      setExamSources(incoming.map((file) => file.name));

      const withIds = collected.map((p, i) => ({ ...p, id: `qp-${Date.now()}-${i}` }));
      setExamPages(withIds);

      const llm = createLlm({ onRetry });

      /* A digital question paper already carries its text; only scans need the
         vision pass. Either way the model reads the structure, and segment.js
         re-reads the printed marks independently as a cross-check. */
      const pageTexts = digitalTexts.slice();
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
      /* The paper's own lines go in too: the rubric that says "answer any 3 of
         the following 5" is printed on it, and the total depends on reading it. */
      const validated = validateExam(raw, {
        structural: structuralMarks(doc.lines),
        lines: doc.lines,
      });

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
      return {
        ...prev,
        ...deriveMarks(questions, prev.declaredTotal, prev.baseWarnings, prev.choice),
      };
    });
  }

  /** Correct a misread "answer any N" — the denominator depends on it. */
  function setChoiceRequired(index, value) {
    setExam((prev) => {
      if (!prev || !Array.isArray(prev.choice) || !prev.choice[index]) return prev;
      const group = prev.choice[index];
      const n = Number(value);
      if (!Number.isFinite(n) || n < 1 || n > group.numbers.length) return prev;

      const choice = prev.choice.map((g, i) => (i === index ? { ...g, required: n } : g));
      return {
        ...prev,
        choice,
        ...deriveMarks(prev.questions, prev.declaredTotal, prev.baseWarnings, choice),
      };
    });
  }

  function clearExam() {
    setExam(null);
    setExamPages([]);
    setExamSources([]);
  }

  /* -------------------------------------------------- reference PDFs ----- */

  async function ingestReferences(files) {
    setError("");
    setRefBusy(true);
    try {
      const added = [];
      const chunks = [];
      const texts = [];
      const cut = [];

      for (const file of Array.from(files || [])) {
        if (isLegacyDoc(file)) {
          setError(LEGACY_DOC_MESSAGE(file.name));
          continue;
        }
        if (isDocx(file)) {
          const text = await extractDocxText(file);
          const produced = chunkDocument(text, { source: file.name, page: 1 });
          texts.push(text);
          chunks.push(...produced);
          added.push({ name: file.name, chunks: produced.length, pageCount: 1 });
          continue;
        }
        if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
          setError(`"${file.name}" is not supported. Reference material must be a PDF or a Word (.docx) file.`);
          continue;
        }
        let { pages, hasText, pageCount } = await extractPdfText(file);

        /* A scanned textbook used to be refused outright, which meant the paper
           was then marked with no reference material at all — the failure the
           whole retrieval stage exists to prevent. It costs one vision call per
           page, so the user is told that is what is happening. */
        if (!hasText) {
          const raster = await rasterizePdf(file);
          if (raster.total > raster.pages.length)
            cut.push({
              scope: "reference",
              name: file.name,
              kept: raster.pages.length,
              total: raster.total,
            });

          const llm = createLlm({ onRetry });
          const read = [];
          for (let i = 0; i < raster.pages.length; i++) {
            setNotice(
              `"${file.name}" is a scan — reading page ${i + 1} of ${raster.pages.length} with the vision model…`
            );
            read.push(
              await llm.callText({
                stage: "reference OCR",
                system: OCR_SYSTEM,
                user:
                  "Extract all text from this page of reference material, preserving any question " +
                  "or section numbering exactly as printed.",
                maxTokens: OCR_MAX_TOKENS,
                model: OCR_MODEL,
                images: [raster.pages[i].dataUrl],
              })
            );
          }
          setNotice("");
          pages = read;
          pageCount = raster.pages.length;
        }

        const produced = pages.flatMap((text, i) =>
          chunkDocument(text, { source: file.name, page: i + 1 })
        );
        texts.push(pages.join("\n\n"));
        chunks.push(...produced);
        added.push({ name: file.name, chunks: produced.length, pageCount, scanned: !hasText });
      }

      setTruncated((prev) => prev.filter((x) => x.scope !== "reference").concat(cut));
      if (added.length) {
        setRefFiles((prev) => prev.concat(added));
        setRefChunks((prev) => prev.concat(chunks));
        setRefText((prev) => (prev ? prev + "\n\n" : "") + texts.join("\n\n"));
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
    setRefText("");
  }

  const onRetry = (seconds) =>
    setNotice(`The marking service is briefly busy — continuing in ${seconds}s…`);

  /**
   * How many API keys the proxy is scheduling over. Each one supports another
   * question in flight; the server keeps the key values to itself and reports
   * only the count.
   */
  async function poolSize() {
    try {
      const res = await fetch(`${GROQ_BASE}/stats`);
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
      const read = String(text || "");
      step(
        `Read page ${i + 1} of ${pages.length}`,
        `${read.length} characters · ${read.split(/\n/).filter((l) => l.trim()).length} lines`,
        read.trim() ? "ok" : "warn"
      );
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

    step(
      "Checking answers against the marking scheme",
      `${marks} marks available · answer ready to mark`
    );

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
    const parsed = extractJson(out.text);
    step(
      "Marking complete",
      `${(parsed.keyPoints || []).length} key point(s) · ${(parsed.annotations || []).length} annotation(s)`,
      "ok"
    );
    return parsed;
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
      aiScore: score, // the machine's proposal, preserved when the examiner adjusts
      overrides: 0,
      totalMarks: total,
      grade: result.grade || "—",
      /* Questions the machine could not mark. The score is still logged, but a
         row carrying pending marks is not a finished result and says so. */
      pending: kp.filter((k) => k.failed).length,
      date: todayLabel(),
    };

    setHistory((prev) => {
      const next = [entry].concat(prev).slice(0, HISTORY_LIMIT);
      saveHistory(next);
      return next;
    });
    setActiveHistoryId(entry.id);
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

    const problem = validate({ needPages: !textDoc, needScheme: !examMode });
    if (problem) {
      reportProblem(problem);
      return;
    }
    setInvalid("");

    setError("");
    setEvaluation(null);
    setActiveAnn(null);
    beginTrace();
    step(
      pages.length ? `Starting on ${pages.length} page${pages.length === 1 ? "" : "s"}` : "Starting on a text document",
      examMode ? `${exam.questions.length} question${exam.questions.length === 1 ? "" : "s"} on the paper` : "",
      "start"
    );

    try {
      if (textDoc && pages.length === 0) {
        /* The Word document's own text is the ground truth, so there is no
           vision pass and no ink geometry. The annotated text view is where
           marks land when there is no page image to draw them on. */
        setStage("evaluating");
        setCoverageInfo(null);
        if (!examMode) setPhase("mark");
        const doc = structureOcr([textDoc.text]);
        setPageBands({});
        setOcrDoc(doc);
        setGeometryByPage({});
        setStudentAnswerText(doc.text);
        setEditedPages([doc.text]);

        const result = examMode
          ? await runExamEvaluation(doc)
          : await runEvaluation(doc.text);

        setEvaluation(result);
        setEvalRun((n) => n + 1);
        recordHistory(result);
        setStage("done");
        setTab("review");
        return;
      }

      setStage("ocr");
      setPhase("ocr");
      setCoverageInfo(null);
      const pageTexts = await runOcr();

      /* Read the ink while the vision result is still fresh. This is local
         canvas work, not a network call, so it costs no tokens and cannot fail
         the run — an unmeasurable page simply loses its overlay. */
      setPhase("measure");
      const measured = await measurePages(pages);
      const doc = structureOcr(pageTexts);
      const confidence = attachGeometry(doc, measured);

      for (let p = 1; p <= doc.pageCount; p++) {
        const bands = measured[p] && measured[p].bands ? measured[p].bands.length : 0;
        const lines = doc.lines.filter((l) => l.page === p).length;
        const skew = measured[p] && Number.isFinite(measured[p].skew) ? measured[p].skew : 0;
        step(
          `Measured page ${p}`,
          `${bands} ink band${bands === 1 ? "" : "s"} vs ${lines} line${lines === 1 ? "" : "s"} — placement ${confidence[p]}` +
            (Math.abs(skew) > 0.2 ? ` · page was ${Math.abs(skew)}° crooked, corrected` : ""),
          confidence[p] === "high" || confidence[p] === "medium" ? "ok" : "warn"
        );
      }

      setPageBands(measured);
      setOcrDoc(doc);
      setGeometryByPage(confidence);
      setStudentAnswerText(doc.text);
      setEditedPages(pageTexts.slice());

      setStage("evaluating");
      if (!examMode) setPhase("mark");
      const result = examMode
        ? await runExamEvaluation(doc)
        : await runEvaluation(doc.text);

      step(
        "Finished",
        `${result.totalMarksAwarded ?? "—"}/${result.totalMarks ?? "—"} · grade ${result.grade || "—"}`,
        "ok"
      );
      setEvaluation(result);
      setEvalRun((n) => n + 1);
      recordHistory(result);
      setStage("done");
      setTab("review");
    } catch (e) {
      step("Run stopped", e.message || String(e), "warn");
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

    setPhase("match");
    step("Matching answers to questions", `${exam.questions.length} questions · ${doc.lines.length} lines of writing`);
    const { answers, unassignedLineIds } = await matchAnswers(doc, exam, { llm, warnings });

    /* One line per question, saying how it was found and how much that route is
       worth. This is the step that decides what gets marked at all, so it is
       the one worth watching. */
    for (const a of answers) {
      const where =
        Number.isFinite(a.pageStart) && a.pageStart > 0
          ? a.pageEnd && a.pageEnd !== a.pageStart
            ? ` · pages ${a.pageStart}–${a.pageEnd}`
            : ` · page ${a.pageStart}`
          : "";
      step(
        `Q${a.number}`,
        a.skipped
          ? "no answer found on the paper"
          : `${a.method === "label" ? "found by its written number" : "matched to this question"} · ` +
            `${a.confidence}% certainty · ${a.answerText.length} characters${where}`,
        a.skipped ? "warn" : "ok"
      );
    }

    /* Writing that matched no question is how a missed answer actually looks.
       Leftover boilerplate ("All the best", a name, a stray mark) is normal, so
       only a substantial block is worth interrupting the examiner for. */
    const leftoverText = linesToText(doc.lines, unassignedLineIds || []);
    const unassignedChars = normalizeText(leftoverText).length;
    const hasUnassignedWriting = unassignedChars >= 40;
    if (hasUnassignedWriting) {
      warnings.push(
        `Possible missed answer: ${unassignedChars} characters of student writing ` +
          `(${unassignedLineIds.length} line${unassignedLineIds.length === 1 ? "" : "s"}) could not be ` +
          `assigned to any question. Check the Answer Text tab before finalising marks.`
      );
    }

    /* Answer coverage, known the moment matching finishes — before any marks
       exist. This is what the processing screen and the Evaluate tab report. */
    const coverage = {
      total: answers.length,
      detected: answers.filter(
        (a) => answerStatus(a, { hasUnassignedWriting }) === ANSWER_STATUS.DETECTED
      ).length,
      verify: answers.filter((a) => {
        const s = answerStatus(a, { hasUnassignedWriting });
        return s === ANSWER_STATUS.UNCERTAIN || s === ANSWER_STATUS.NOT_DETECTED;
      }).length,
      unanswered: answers.filter(
        (a) => answerStatus(a, { hasUnassignedWriting }) === ANSWER_STATUS.UNANSWERED
      ).length,
      hasUnassignedWriting,
      unassignedChars,
    };
    if (hasUnassignedWriting)
      step(
        "Writing left over",
        `${unassignedChars} characters on ${unassignedLineIds.length} line(s) belong to no question`,
        "warn"
      );
    setCoverageInfo(coverage);

    const retriever = refChunks.length ? createRetriever(refChunks) : null;
    const concurrency = await poolSize();

    /* Pair the reference's model answers to the paper's questions by number.
       Where both sides carry the same number, the marking compares like with
       like; where they do not, retrieval still supplies the topic. */
    if (Array.isArray(exam.choice) && exam.choice.length)
      step(
        "Applying the paper's choice",
        `${describeChoice(exam.choice)} — the total is out of ${exam.totalMarks}, not ${exam.printedMarks}`,
        "start"
      );

    const referenceAnswers = pairReferenceAnswers(refText, exam.questions);
    const paired = referenceAnswers.size;
    if (refText.trim())
      step(
        "Paired reference answers to questions",
        `${paired} of ${exam.questions.length} question(s) have a model answer under the same number` +
          (paired < exam.questions.length ? " — the rest are marked from the retrieved passages" : ""),
        paired > 0 ? "ok" : "warn"
      );

    setPhase("mark");
    step(
      "Marking each answer",
      `${answers.filter((a) => !a.skipped).length} answer(s) to judge` +
        (retriever ? ` · reference material available` : ` · no reference material`) +
        ` · ${concurrency} in parallel`
    );
    const paper = await assessPaper({
      exam,
      answers,
      retriever,
      llm,
      warnings,
      concurrency,
      referenceAnswers,
      // The scheme box doubles as the answer key in exam mode. Question papers
      // exported from question banks often have empty "Answer Key:" fields, and
      // without a key the examiner must re-derive every answer itself.
      answerKey: expectedAnswer,
      onProgress: (done, total, label, question) => {
        setMarkProgress({ done, total, label });
        setNotice(label ? `Marking ${label} — ${done + 1} of ${total}` : "");
        if (!question) return;
        step(
          `Marked ${label}`,
          question.failed
            ? "marking could not finish — left for the teacher to review"
            : `${question.marksAwarded}/${question.maxMarks} marks · ${question.grounding === GROUNDING.REFERENCE ? "based on the reference" : question.grounding === GROUNDING.INSUFFICIENT ? "reference did not cover this" : "based on subject knowledge"} · ${question.confidence}% certainty`,
          question.failed ? "warn" : "ok"
        );
      },
    });

    setPhase("review");
    setNotice("Writing the final remark…");
    step("Preparing the teacher's summary", `${paper.totalMarks}/${paper.maximumMarks} · grade ${paper.grade}`);
    const remark = await summarisePaper({ paper, llm });

    setRawResponse(JSON.stringify(paper, null, 2));
    setReasoning("");
    const evaluation = toEvaluation(paper, remark);
    /* Detection context the UI needs to separate "the student wrote nothing"
       from "we could not find what the student wrote". */
    evaluation.hasUnassignedWriting = coverage.hasUnassignedWriting;
    evaluation.unassignedChars = coverage.unassignedChars;
    return evaluation;
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
      setError("There is no readable answer text to mark.");
      return;
    }

    setError("");
    setActiveAnn(null);
    beginTrace();
    step("Marking again", "using the corrected answer text", "start");
    try {
      setStage("evaluating");
      setCoverageInfo(null);
      if (!examMode) setPhase("mark");
      setStudentAnswerText(text);

      /* Corrected text shifts every character offset, so the line index has to be
         rebuilt before annotations can be traced back to the page. The ink
         measurements still hold — only the text changed — and because the
         transcript is held per page, the page boundaries the measurements are
         attached to survive the edit. That used to be true on a single page
         only; a multi-page edit dropped the overlay entirely. */
      const source = sourceText !== undefined ? [String(sourceText)] : editedPages;
      let doc = ocrDoc;
      let confidence = geometryByPage;

      if (!doc || text !== doc.text) {
        doc = structureOcr(source.length ? source : [text]);
        confidence = attachGeometry(doc, pageBands);
        setOcrDoc(doc);
        setGeometryByPage(confidence);
        step(
          "Rebuilt the line index",
          `${doc.lines.length} lines across ${doc.pageCount} page(s) — ink measurements reattached`
        );
      }

      const result = examMode && doc ? await runExamEvaluation(doc) : await runEvaluation(text);
      setEvaluation(result);
      setEvalRun((n) => n + 1);
      recordHistory(result);
      setStage("done");
      setTab("review");
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

  const answerPaperReady = pages.length > 0 || !!textDoc;
  const markingBasisReady = examMode
    ? !exam.blocking
    : expectedAnswer.trim().length > 0 && Number(totalMarks) > 0;
  const readyToCheck = answerPaperReady && markingBasisReady;

  const addStudentDetail = () =>
    setStudentDetails((rows) => rows.concat({ id: Date.now(), label: "", value: "" }));
  const updateStudentDetail = (id, field, value) =>
    setStudentDetails((rows) => rows.map((row) => (row.id === id ? { ...row, [field]: value } : row)));
  const removeStudentDetail = (id) =>
    setStudentDetails((rows) => rows.filter((row) => row.id !== id));

  const updateFinalMark = (i, value) => {
    setMarkOverrides((prev) => {
      const next = { ...prev };
      if (value === null) delete next[i];
      else next[i] = value;
      return next;
    });
    setMarkNotice("Mark updated");
    window.setTimeout(() => setMarkNotice(""), 1800);
  };

  /* ============================================================== RENDER === */
  return (
    <div className="qiq-root">
      <style>{CSS}</style>

      <header className="qiq-header qiq-noprint">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="qiq-logo">Q</div>
          <div>
            <div className="qiq-brand-title">
              QIQ <span style={{ color: C.faint, fontWeight: 500 }}>/</span> Paper Checker for Educators
            </div>
            <div className="qiq-brand-subtitle">
              Upload the papers, review the marking, and create a student-ready report
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
            compactTop
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
                  accept="application/pdf,image/*,.doc,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword"
                  multiple
                  hidden
                  onChange={(e) => {
                    ingestExam(e.target.files);
                    e.target.value = "";
                  }}
                />
                <div className="qiq-upload-icon" aria-hidden="true">{examBusy ? "…" : "1"}</div>
                <div className="qiq-upload-body">
                  <div className="qiq-upload-title">
                    {examBusy ? "Reading the question paper…" : "Choose question paper"}
                  </div>
                  <div className="qiq-upload-copy">
                    PDF, Word, scan, or photo. Questions and marks are read automatically.
                  </div>
                </div>
                {!examBusy && <span className="qiq-upload-action">Browse files</span>}
              </label>
              <TruncatedNotice items={truncated.filter((x) => x.scope === "question")} />
              <div className="qiq-hint qiq-optional-note">
                No question paper? Use a marking scheme below instead.
              </div>
            </>
          )}

          {exam && (
            <>
              <div className="qiq-upload-confirm">
                <span>✓</span>
                <span>
                  <strong>{examSources.join(", ") || "Question paper"}</strong> uploaded · {exam.questions.length} questions found
                </span>
              </div>
              <ExamPanel exam={exam} onMarks={setQuestionMarks} onChoice={setChoiceRequired} />
            </>
          )}

          <SectionTitle
            n="2"
            title="Reference material · Optional"
            action={
              refFiles.length > 0 && (
                <button className="qiq-mini-btn" onClick={clearReferences}>
                  Clear
                </button>
              )
            }
          />
          <label className={`qiq-drop qiq-drop-sm${refFiles.length ? " is-complete" : ""}`}>
            <input
              type="file"
              accept="application/pdf,.doc,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword"
              multiple
              hidden
              onChange={(e) => {
                ingestReferences(e.target.files);
                e.target.value = "";
              }}
            />
            <div className="qiq-upload-icon is-optional" aria-hidden="true">{refBusy ? "…" : "2"}</div>
            <div className="qiq-upload-body">
              <div className="qiq-upload-title">
                {refBusy ? "Preparing reference material…" : "Choose reference material"}
              </div>
              <div className="qiq-upload-copy">
                Textbook, notes, syllabus, or model answers to guide marking.
              </div>
            </div>
            {!refBusy && (
              <span className="qiq-upload-action">{refFiles.length ? "Add more files" : "Browse files"}</span>
            )}
          </label>

          {refFiles.length > 0 && (
            <div className="qiq-upload-confirm">
              <span>✓</span>
              <span>
                {refFiles.length} reference {refFiles.length === 1 ? "file" : "files"} uploaded successfully
              </span>
            </div>
          )}

          {refFiles.length > 0 && (
            <div className="qiq-reflist">
              {refFiles.map((f, i) => (
                <div key={i} className="qiq-refitem">
                  <span className="qiq-reficon">📄</span>
                  <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {f.name}
                  </span>
                  <span style={{ color: C.faint }}>
                    {f.pageCount} page{f.pageCount === 1 ? "" : "s"} · {f.chunks} searchable section{f.chunks === 1 ? "" : "s"}
                    {f.scanned && (
                      <span
                        style={{ color: C.amber }}
                        title="This file is a scan, so each page was read from its image. Please check unclear text before finalising marks."
                      >
                        {" "}
                        · scanned copy
                      </span>
                    )}
                  </span>
                </div>
              ))}
              <div className="qiq-hint" style={{ marginTop: 2 }}>
                {refChunks.length} reference section{refChunks.length === 1 ? "" : "s"} ready for marking.
              </div>
              <TruncatedNotice items={truncated.filter((x) => x.scope === "reference")} />
            </div>
          )}
          {refFiles.length === 0 && (
            <div className="qiq-hint">
              No reference: marking uses general subject knowledge.
            </div>
          )}

          <SectionTitle n="3" title="Student answer sheet · Required" />

          <label
            id="qiq-field-pages"
            className={`qiq-drop${pages.length || textDoc ? " is-complete" : ""}${dragging ? " is-dragging" : ""}${invalid === "pages" ? " qiq-invalid" : ""}`}
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
              accept="image/png,image/jpeg,image/gif,image/webp,application/pdf,.doc,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword"
              multiple
              style={{ display: "none" }}
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <div className="qiq-drop-icon">{preparing ? <span className="qiq-spinner" /> : "↑"}</div>
            <div className="qiq-upload-body">
              <div className="qiq-upload-title is-large">
                {preparing ? "Preparing answer pages…" : "Choose the student's answer sheet"}
              </div>
              <div className="qiq-upload-copy">
                Drag and drop a PDF, Word document, JPG, or PNG. Scanned and handwritten answer sheets are supported.
              </div>
            </div>
            {!preparing && (
              <span className="qiq-upload-action">
                {pages.length || textDoc ? "Add more answer files" : "Browse answer files"}
              </span>
            )}
          </label>

          {(pages.length > 0 || textDoc) && (
            <div className="qiq-upload-confirm">
              <span>✓</span>
              <span>
                {textDoc
                  ? `${textDoc.name} uploaded successfully`
                  : `${pages.length} answer ${pages.length === 1 ? "page" : "pages"} uploaded successfully`}
              </span>
            </div>
          )}

          <TruncatedNotice items={truncated.filter((x) => x.scope === "answer")} />

          {pages.length > 0 && (
            <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
              {pages.map((p, i) => (
                <div key={p.id} className="qiq-file">
                  <img src={p.dataUrl} alt="" className="qiq-thumb" />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="qiq-ellipsis" style={{ fontSize: 15, fontWeight: 500 }}>
                      {p.label}
                    </div>
                    <div style={{ fontSize: 13, color: C.faint }}>
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

          {textDoc && (
            <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
              <div className="qiq-file">
                <span className="qiq-reficon">📄</span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="qiq-ellipsis" style={{ fontSize: 15, fontWeight: 500 }}>
                    {textDoc.name}
                  </div>
                  <div style={{ fontSize: 13, color: C.faint }}>
                    Word document · ready to mark
                  </div>
                </div>
                <button className="qiq-x" onClick={() => setTextDoc(null)} disabled={busy} title="Remove">
                  ×
                </button>
              </div>
            </div>
          )}

          <details className="qiq-collapsible">
            <summary className="qiq-collapsible-summary">
              <span className="qiq-step-num">4</span>
              <span className="qiq-collapsible-title">Student details · Optional</span>
              {(studentName.trim() || subject.trim() || studentDetails.some((row) => row.value.trim())) && (
                <span className="qiq-collapsible-value">
                  {[studentName.trim(), subject.trim(), ...studentDetails.map((row) => row.value.trim())]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              )}
              <span className="qiq-collapsible-chevron" aria-hidden="true">⌄</span>
            </summary>
            <div className="qiq-collapsible-body">
              <input
                className="qiq-input"
                value={studentName}
                onChange={(e) => setStudentName(e.target.value)}
                placeholder="Student name for the report"
              />
              <input
                className="qiq-input"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Subject, for example Biology"
              />
              {studentDetails.map((row) => (
                <div className="qiq-detail-row" key={row.id}>
                  <input
                    className="qiq-input"
                    value={row.label}
                    onChange={(e) => updateStudentDetail(row.id, "label", e.target.value)}
                    placeholder="Field, for example Roll number"
                    aria-label="Student detail field name"
                  />
                  <input
                    className="qiq-input"
                    value={row.value}
                    onChange={(e) => updateStudentDetail(row.id, "value", e.target.value)}
                    placeholder="Value"
                    aria-label={row.label || "Student detail value"}
                  />
                  <button
                    className="qiq-detail-remove"
                    type="button"
                    onClick={() => removeStudentDetail(row.id)}
                    aria-label="Remove this student detail"
                  >
                    Remove
                  </button>
                </div>
              ))}
              <button className="qiq-add-detail" type="button" onClick={addStudentDetail}>
                Add another detail
              </button>
            </div>
          </details>

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

          <button className="qiq-btn qiq-primary-action" onClick={checkPaper} disabled={busy || preparing || !readyToCheck}>
            {busy ? (
              <>
                <span className="qiq-spinner" />
                {stage === "ocr" ? "Reading paper…" : "Marking answers…"}
              </>
            ) : evaluation ? (
              "Check this paper again"
            ) : (
              "Start checking paper"
            )}
          </button>

          {evaluation && !busy && (
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button className="qiq-btn qiq-btn-ghost" onClick={() => reEvaluate()} style={{ flex: 1 }}>
                Mark again
              </button>
              <button className="qiq-btn qiq-btn-ghost" onClick={resetAll} style={{ flex: 1 }}>
                New paper
              </button>
            </div>
          )}

          <HistoryPanel history={history} onClear={clearHistory} />

          <p style={{ fontSize: 13, color: C.faint, marginTop: 14, lineHeight: 1.6 }}>
            “Mark again” uses the answer text already read, so you can adjust the marking scheme or correct
            misread words without reading every page again.
          </p>
        </aside>

        {/* ================================================= RIGHT PANEL === */}
        <main className="qiq-panel qiq-right">
          {busy && (
            <Processing
              stage={stage}
              phase={phase}
              elapsed={elapsed}
              progress={ocrProgress}
              notice={notice}
              marking={markProgress}
              coverage={coverageInfo}
              examMode={!!(exam && exam.questions.length)}
              trace={trace}
            />
          )}

          {!busy && !evaluation && (
            <EmptyState
              hasPages={pages.length > 0 || !!textDoc}
              hasExam={!!exam}
              hasReference={refFiles.length > 0}
            />
          )}

          {!busy && evaluation && (
            <>
              <div className="qiq-tabs qiq-noprint">
                {[
                  ["review", "Review Marks"],
                  ["paper", "Student Answer Sheet"],
                  ["annotated", "Text & Comments"],
                  ["score", "Final Report"],
                  ["raw", "Answer Text"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={`qiq-tab${tab === id ? " is-active" : ""}`}
                    onClick={() => setTab(id)}
                  >
                    {label}
                    {id === "review" && reviewIssueCount > 0 && (
                      <span className="qiq-tab-warn" title={`${reviewIssueCount} item(s) need the teacher's attention`}>
                        {reviewIssueCount}
                      </span>
                    )}
                    {id === "score" && (
                      <span className="qiq-tab-badge" style={{ background: gradeColor(effectiveGrade) }}>
                        {scoreAwarded}/{scoreTotal}
                      </span>
                    )}
                    {id === "paper" && lowConfidenceCount > 0 && (
                      <span className="qiq-tab-warn" title={`${lowConfidenceCount} mark(s) need manual review`}>
                        {lowConfidenceCount}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              <div className="qiq-tabbody">
                {tab === "review" && (
                  <ReviewPanel
                    evaluation={evaluation}
                    keyPoints={displayKeyPoints}
                    markOverrides={markOverrides}
                    onOverride={updateFinalMark}
                    markNotice={markNotice}
                    awarded={scoreAwarded}
                    total={scoreTotal}
                    grade={effectiveGrade}
                    hasPages={pages.length > 0}
                    hasReference={refFiles.length > 0}
                    geometryByPage={geometryByPage}
                    onViewOnPage={viewQuestionOnPage}
                    onRetryEval={() => reEvaluate()}
                    onShowRaw={() => setTab("raw")}
                  />
                )}
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
                    selectedQuestion={selectedQuestion}
                    selectedPages={selectedPages}
                    selectedRegions={selectedRegions}
                    selectedMark={selectedMark}
                    selectedOverridden={selectedIndex >= 0 && Number.isFinite(markOverrides[selectedIndex])}
                    hasUnassignedWriting={!!evaluation.hasUnassignedWriting}
                    onClearSelection={() => setSelectedQuestionId(null)}
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
                    keyPoints={displayKeyPoints}
                    awarded={scoreAwarded}
                    total={scoreTotal}
                    grade={effectiveGrade}
                    overrideCount={overrideCount}
                    runKey={evalRun}
                    studentName={studentName}
                    setStudentName={setStudentName}
                    subject={subject}
                    setSubject={setSubject}
                    reportDate={reportDate}
                    setReportDate={setReportDate}
                    hasReference={refFiles.length > 0}
                    studentDetails={studentDetails}
                    setStudentDetails={setStudentDetails}
                  />
                )}
                {tab === "raw" && (
                  <RawView
                    pageTexts={editedPages}
                    setPageTexts={setEditedPages}
                    geometryByPage={geometryByPage}
                    dirty={editedText !== studentAnswerText}
                    onReEvaluate={() => reEvaluate()}
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

function EmptyState({ hasPages, hasExam, hasReference }) {
  const steps = [
    { done: hasExam, label: "Question paper", note: hasExam ? "Questions and marks added" : "Recommended to save setup time" },
    { done: hasReference, label: "Reference material", note: hasReference ? "Ready to guide marking" : "Optional — textbook, notes, or model answers" },
    { done: hasPages, label: "Student answer paper", note: hasPages ? "Ready to be checked" : "Required before checking" },
  ];
  return (
    <div className="qiq-empty">
      <div className="qiq-empty-kicker">New paper</div>
      <div className="qiq-empty-icon">✓</div>
      <div className="qiq-empty-title">
        {hasPages ? "Your answer paper is ready" : "Set up a paper in three simple steps"}
      </div>
      <p className="qiq-empty-copy">
        {hasPages
          ? "Review the setup below, then use the button at the bottom of the left panel to begin checking."
          : "Use the upload panel on the left. QIQ accepts typed documents, scans, phone photos, and handwritten work."}
      </p>
      <div className="qiq-setup-list">
        {steps.map((item, i) => (
          <div key={item.label} className={`qiq-setup-item${item.done ? " is-done" : ""}`}>
            <span className="qiq-setup-status">{item.done ? "✓" : i + 1}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.note}</small>
            </span>
          </div>
        ))}
      </div>
      <div className="qiq-privacy-note">
        🔒 Your access key stays protected on the server. Uploaded pages are sent securely for reading and marking.
      </div>
    </div>
  );
}

/**
 * What the pipeline is doing, in the order it actually does it.
 *
 * The step list is driven by `phase`, which moves only when the corresponding
 * work genuinely starts (ocr → measure → match → mark → review), so a lit step
 * is a fact, not an animation. Coverage counts appear the moment answer
 * matching finishes — the examiner learns "3 of 5 answers detected" while the
 * marking is still running.
 */
function Processing({ stage, phase, elapsed, progress, notice, marking, coverage, examMode, trace = [] }) {
  const order = examMode
    ? ["ocr", "measure", "match", "mark", "review"]
    : ["ocr", "measure", "mark"];

  const steps = [
    {
      id: "ocr",
      title: "Reading answer pages",
      sub:
        progress.total > 1
          ? `Transcribing page ${Math.min(progress.done + 1, progress.total)} of ${progress.total}, handwriting included`
          : "Reading the page, including handwriting",
    },
    { id: "measure", title: "Detecting handwriting position", sub: "Measuring where the ink sits on each page" },
    { id: "match", title: "Mapping answers to questions", sub: "Linking each block of writing to its question number" },
    {
      id: "mark",
      title: "Marking answers",
      sub:
        marking && marking.total > 0
          ? `Marking ${marking.label || "question"} — ${Math.min(marking.done + 1, marking.total)} of ${marking.total}`
          : "Reasoning through the marking scheme and partial credit",
    },
    { id: "review", title: "Preparing teacher review", sub: "Writing the final comment and answer summary" },
  ].filter((s) => order.includes(s.id));

  const current = Math.max(0, order.indexOf(phase === "idle" ? stage : phase));
  const active = steps[current] || steps[steps.length - 1];

  /* The feed follows the work. Newest at the bottom, like a log, because the
     examiner is watching the front of the run rather than reading a history. */
  const feed = useRef(null);
  useEffect(() => {
    if (feed.current) feed.current.scrollTop = feed.current.scrollHeight;
  }, [trace.length, notice]);

  const clock = (secs) => {
    const s = Math.max(0, Math.floor(secs));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  };
  const mark = { ok: "✓", warn: "!", start: "▸", info: "·" };

  return (
    <div className="qiq-run">
      <div className="qiq-run-head">
        <div className="qiq-orb" />
        <div style={{ minWidth: 0 }}>
          <div className="qiq-run-title">
            {active ? active.title : "Checking the paper"}
            <span className="qiq-run-ell" />
          </div>
          <div className="qiq-run-sub">{notice || (active ? active.sub : "")}</div>
        </div>
        <div className="qiq-run-clock">{clock(elapsed)}</div>
      </div>

      {/* the phases, as a rail: where the run is, and what is still to come */}
      <div className="qiq-rail">
        {steps.map((s, i) => (
          <div
            key={s.id}
            className={`qiq-rail-seg${i < current ? " is-done" : ""}${i === current ? " is-now" : ""}`}
            title={s.title}
          >
            <span className="qiq-rail-bar" />
            <span className="qiq-rail-label">{s.title.replace(/^(Reading|Detecting|Mapping|Evaluating|Preparing) /, "")}</span>
          </div>
        ))}
      </div>

      {/* what actually happened, as it happens */}
      <div className="qiq-trace" ref={feed}>
        {trace.length === 0 && <div className="qiq-trace-idle">Preparing to read the paper…</div>}

        {trace.map((e, i) => (
          <div key={i} className={`qiq-trace-row is-${e.kind}`}>
            <span className="qiq-trace-time">{clock(e.at)}</span>
            <span className="qiq-trace-mark">{mark[e.kind] || "·"}</span>
            <span className="qiq-trace-text">{e.text}</span>
            {e.detail && <span className="qiq-trace-detail">{e.detail}</span>}
          </div>
        ))}

        <div className="qiq-trace-row is-live">
          <span className="qiq-trace-time">{clock(elapsed)}</span>
          <span className="qiq-spinner" />
          <span className="qiq-trace-text">{notice || (active ? active.sub : "Working")}</span>
        </div>
      </div>

      {coverage && (
        <div className="qiq-proc-coverage">
          Questions: {coverage.total} · Answers detected: {coverage.detected} · Needs verification:{" "}
          {coverage.verify}
          {coverage.unanswered > 0 ? ` · No answer detected: ${coverage.unanswered}` : ""}
        </div>
      )}
    </div>
  );
}

/**
 * Pages an upload could not take. Shown next to the upload it belongs to and
 * kept there: a paper that lost its last pages is being marked on less than the
 * student wrote, and that is not a transient message.
 */
function TruncatedNotice({ items }) {
  if (!items.length) return null;
  return (
    <div className="qiq-cut">
      {items.map((x, i) => (
        <div key={i}>
          ⚠️ <strong>{x.name}</strong> has {x.total} pages — only the first {x.kept} were added.{" "}
          {x.scope === "question"
            ? "Questions past that point are not on the paper being marked."
            : x.scope === "reference"
            ? "Material past that point cannot be used when marking."
            : "Answers past that point will not be read or marked."}{" "}
          Split the file and upload the rest as a second file.
        </div>
      ))}
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
        <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: 0.7, textTransform: "uppercase", color: C.dim }}>
          Recent checks
        </span>
        <button className="qiq-mini-btn" style={{ marginLeft: "auto" }} onClick={onClear}>
          Clear
        </button>
      </div>

      <div style={{ display: "grid", gap: 7 }}>
        {history.map((h, i) => {
          /* Progress means one student against their own last paper. Comparing
             the row above — whoever it belonged to — turned two different
             students into a trend arrow that meant nothing. */
          const prev = history.slice(i + 1).find((o) => o.studentName === h.studentName);
          const delta = prev ? pct(h) - pct(prev) : 0;
          const trend = !prev ? "•" : delta > 1 ? "↑" : delta < -1 ? "↓" : "→";
          const trendColor = !prev ? C.faint : delta > 1 ? C.green : delta < -1 ? C.red : C.faint;

          return (
            <div key={h.id} className="qiq-hist">
              <span className="qiq-hist-trend" style={{ color: trendColor }} title={
                prev
                  ? `${delta > 0 ? "+" : ""}${delta.toFixed(0)}% vs this student's previous paper (${prev.score}/${prev.totalMarks}, ${prev.date})`
                  : `First recorded paper for ${h.studentName}`
              }>
                {trend}
              </span>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="qiq-ellipsis" style={{ fontSize: 14, fontWeight: 600, color: C.text }}>
                  {h.studentName}
                </div>
                <div className="qiq-ellipsis" style={{ fontSize: 12.5, color: C.faint }}>
                  {h.subject} · {h.date}
                  {h.pending > 0 && (
                    <span style={{ color: C.amber }} title="Some questions still need the teacher's mark">
                      {" "}
                      · {h.pending} pending
                    </span>
                  )}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: C.text, fontVariantNumeric: "tabular-nums" }}>
                  {h.overrides > 0 && <span title={`Teacher adjusted ${h.overrides} mark(s); the original suggestion was ${h.aiScore}/${h.totalMarks}`}>✎ </span>}
                  {h.score}/{h.totalMarks}
                </div>
                {h.overrides > 0 && Number.isFinite(h.aiScore) && h.aiScore !== h.score && (
                  <div style={{ fontSize: 12, color: C.faint }}>Suggested {h.aiScore}</div>
                )}
                <div style={{ fontSize: 12.5, fontWeight: 700, color: gradeColor(h.grade) }}>{h.grade}</div>
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
function GroundingBadge({ grounding, hasReference = true }) {
  const map = {
    [GROUNDING.REFERENCE]: ["is-ref", "Used reference material"],
    [GROUNDING.GENERAL]: ["is-general", "Used subject knowledge"],
    [GROUNDING.INSUFFICIENT]: [
      "is-insufficient",
      hasReference ? "Reference did not cover this question" : "No reference material was provided",
    ],
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
function ExamPanel({ exam, onMarks, onChoice }) {
  const [originalMissingIds] = useState(
    () => new Set(exam.questions.filter((q) => q.maxMarks === null).map((q) => q.id))
  );
  const correctionQuestions = exam.questions.filter((q) => originalMissingIds.has(q.id));
  const missingQuestions = correctionQuestions.filter((q) => q.maxMarks === null);
  const missing = missingQuestions.length;
  const completeQuestions = exam.questions.filter((q) => !originalMissingIds.has(q.id));
  const choice = Array.isArray(exam.choice) ? exam.choice : [];
  const missingGroups = [];
  for (const question of correctionQuestions) {
    const number = String(question.number || "");
    const base = (/^\s*(\d+)/.exec(number) || [null, number])[1];
    const last = missingGroups[missingGroups.length - 1];
    if (last && last.base === base) last.questions.push(question);
    else missingGroups.push({ base, questions: [question] });
  }
  const missingRows = missingGroups.flatMap((group) => {
    const rows = [];
    for (let i = 0; i < group.questions.length; i += 2) rows.push(group.questions.slice(i, i + 2));
    return rows;
  });

  return (
    <div className="qiq-exam">
      <div className="qiq-exam-head">
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.text }}>
            {exam.title || "Question paper"}
          </div>
          <div style={{ fontSize: 13.5, color: C.faint, marginTop: 2 }}>
            {exam.subject ? exam.subject + " · " : ""}
            {exam.questions.length} question{exam.questions.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="qiq-exam-total">
          <span style={{ fontSize: 17, fontWeight: 800, color: missing ? C.amber : C.green }}>
            {exam.totalMarks}
          </span>
          <span style={{ fontSize: 12.5, color: C.faint, display: "block" }}>
            {choice.length ? "marks to be earned" : "total marks"}
          </span>
          {choice.length > 0 && exam.printedMarks !== exam.totalMarks && (
            <span style={{ fontSize: 12, color: C.faint, display: "block" }}>
              {exam.printedMarks} printed
            </span>
          )}
        </div>
      </div>

      {/* What the paper's own rubric says about choice, read off the page. The
          number is editable because a misread rubric would otherwise score the
          whole paper out of the wrong denominator, and the examiner is the one
          who can see what is printed. */}
      {choice.map((g, i) => (
        <div key={i} className="qiq-choice">
          <span>Answer any</span>
          <input
            className="qiq-exam-marks"
            type="number"
            min="1"
            max={g.numbers.length}
            value={g.required}
            onChange={(e) => onChoice(i, e.target.value)}
            aria-label="Number of questions the student must attempt"
          />
          <span>
            of {g.numbers.length} — Q{g.numbers.join(", Q")}
          </span>
          <span className="qiq-choice-src" title={g.text}>
            read from the paper
          </span>
        </div>
      ))}

      {correctionQuestions.length > 0 && (
        <div className={`qiq-missing-marks${missing === 0 ? " is-complete" : ""}`} role="status">
          <div className="qiq-missing-marks-head">
            <span className="qiq-missing-icon">{missing === 0 ? "✓" : "!"}</span>
            <span>
              <strong>{missing === 0 ? "All question marks are ready" : "Marks needed before grading"}</strong>
              <small>
                {missing === 0
                  ? "Your marks have been saved. You can review them here or continue to check the paper."
                  : "These values were not visible on the question paper. Enter them here."}
              </small>
            </span>
          </div>
          <div className="qiq-missing-marks-fields">
            {missingRows.map((row, rowIndex) => (
              <div key={row.map((q) => q.id).join("-")} className="qiq-missing-mark-row">
                {row.map((q, questionIndex) => (
                  <label key={q.id} className="qiq-missing-mark-field">
                    <span className="qiq-missing-question" title={q.text || `Question ${q.number}`}>
                      <strong>Q{q.number}</strong>
                      <small>{q.text || "Question wording was not read"}</small>
                    </span>
                    <input
                      className="qiq-exam-marks"
                      type="number"
                      min="1"
                      value={q.maxMarks === null ? "" : q.maxMarks}
                      placeholder="Marks"
                      onChange={(e) => onMarks(q.id, e.target.value)}
                      aria-label={`Marks for question ${q.number}`}
                      autoFocus={rowIndex === 0 && questionIndex === 0}
                    />
                  </label>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="qiq-exam-rows">
        {completeQuestions.map((q) => (
          <div key={q.id} className={`qiq-exam-row${q.maxMarks === null ? " is-missing" : ""}`}>
            <span className="qiq-exam-num">{q.number}</span>
            <span className="qiq-exam-text" title={q.text}>
              {q.text || <em style={{ color: C.faint }}>no wording read</em>}
            </span>
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

      {exam.warnings.filter((w) => !/no printed marks/i.test(w)).length > 0 && (
        <div className="qiq-exam-warn">
          {exam.warnings.filter((w) => !/no printed marks/i.test(w)).map((w, i) => (
            <div key={i} className="qiq-exam-warnrow">
              <strong>Please review</strong>
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
  selectedQuestion = null,
  selectedPages = null,
  selectedRegions = [],
  selectedMark = null,
  selectedOverridden = false,
  hasUnassignedWriting = false,
  onClearSelection,
}) {
  const focus = (idx) => {
    setActiveAnn(idx);
    const el = document.getElementById(`qiq-note-${idx}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  /* A box on a page whose line measurement failed is a guess dressed up as a
     fact. "high" and "medium" placements are shown (medium admits it may sit a
     line off); "low" and "none" are withheld, and the marks live in the margin
     instead — where they cannot point at the wrong handwriting. */
  const pageTrusted = (pageNumber) =>
    geometryByPage[pageNumber] === "high" || geometryByPage[pageNumber] === "medium";

  /* When the examiner picked a question, only its annotations appear. The
     filter reads ann.questionId — stamped at marking time — so no component
     re-guesses which mark belongs to which question. */
  const belongsToSelection = (idx) =>
    !selectedQuestion || annotations[idx]?.questionId === selectedQuestion.questionId;

  /* One annotation can occupy boxes on more than one page, so the grouping is
     by page rather than by annotation. Top-to-bottom order keeps the tab key
     and the eye moving down the page together. */
  const boxesForPage = (pageNumber) => {
    const out = [];
    if (!pageTrusted(pageNumber)) return out;
    for (const [key, list] of Object.entries(annBoxes)) {
      if (!belongsToSelection(Number(key))) continue;
      for (const entry of list) {
        if (entry.page === pageNumber) out.push({ idx: Number(key), bbox: entry.bbox });
      }
    }
    return out.sort((a, b) => a.bbox.y - b.bbox.y);
  };

  const regionFor = (pageNumber) => selectedRegions.find((r) => r.page === pageNumber) || null;

  /* A question's region is a block of writing, not a phrase, so a weak line
     measurement is still worth drawing — being a line or two out around a whole
     answer still points the examiner at the right paragraph. It is labelled
     approximate rather than passed off as exact. Individual marks keep the
     stricter gate: pointing at the wrong *phrase* is a different kind of wrong. */
  const regionPlaceable = (pageNumber) =>
    !!geometryByPage[pageNumber] && geometryByPage[pageNumber] !== "none";
  const placedRegions = selectedRegions.filter((r) => regionPlaceable(r.page));

  /* Navigate to the answer itself where it was measured, and only fall back to
     the top of its first page when it was not. Derived from selectedQuestion —
     set once by the Evaluate tab — never from a copy. */
  useEffect(() => {
    if (!selectedQuestion) return;
    const region = selectedRegions.find((r) => regionPlaceable(r.page));
    const el = region
      ? document.getElementById(`qiq-region-${region.page}`)
      : selectedPages && selectedPages.length
      ? document.getElementById(`qiq-page-${selectedPages[0]}`)
      : null;
    if (el) el.scrollIntoView({ behavior: "smooth", block: region ? "center" : "start" });
  }, [selectedQuestion && selectedQuestion.questionId]);

  /* Pages can be removed after a grading run, and an edited transcript drops the
     geometry on purpose. Either way this view has nothing to stand on. */
  const note = pages.length === 0
    ? "The pages have been removed, so there is nothing left to mark on."
    : GEOMETRY_NOTE[geometryLevel] || "";

  /* "Placed" means placed somewhere worth trusting. Boxes withheld because
     their page failed the geometry check count as unplaced, so the header
     never claims a mark is on the handwriting when it is not. */
  const placedTrusted = annotations.filter(
    (_, i) => annBoxes[i] && annBoxes[i].some((b) => pageTrusted(b.page))
  ).length;
  const unplaceable = annotations.length - placedTrusted;

  /* Header counts follow the selection: with a question chosen they describe
     that question's marks, not the paper's. */
  const visibleIdx = annotations.map((_, i) => i).filter(belongsToSelection);
  const placedVisible = visibleIdx.filter(
    (i) => annBoxes[i] && annBoxes[i].some((b) => pageTrusted(b.page))
  ).length;

  return (
    <div className="qiq-annot-wrap qiq-marked-layout">
      <div>
        <div className="qiq-subhead">
          <div>
            <div style={{ fontSize: 16, fontWeight: 750 }}>
              {selectedQuestion
                ? `Q${selectedQuestion.number} on the page`
                : "The student's paper, marked"}
              {markingInProgress && <span className="qiq-marking-dot" />}
            </div>
            <div style={{ fontSize: 15, color: C.dim, marginTop: 3 }}>
              {markingInProgress
                ? `Marking… ${revealed} of ${annotations.length}`
                : selectedQuestion
                ? `${placedVisible} of ${visibleIdx.length} of this question's marks placed on the handwriting`
                : `${placedTrusted} of ${annotations.length} marks placed on the handwriting` +
                  (unplaceable > 0 ? ` · ${unplaceable} in the margin` : "")}
            </div>
          </div>
          {markingInProgress ? (
            <button className="qiq-mini-btn" onClick={onRevealAll}>
              Skip animation
            </button>
          ) : selectedQuestion ? (
            <button className="qiq-mini-btn" onClick={onClearSelection}>
              Show all questions
            </button>
          ) : (
            <Legend />
          )}
        </div>

        {note && !selectedQuestion && (
          <div className="qiq-geom-note">
            <span>
              {note}{" "}
              <button className="qiq-linkbtn" onClick={onShowText}>
                Open the text view
              </button>
            </span>
          </div>
        )}

        {/* What the examiner asked to see: the question itself, the mark it
            carries, and where on the paper that mark was earned. Without this
            the filtered page view is a set of highlights with no question
            attached to them. */}
        {selectedQuestion && (
          <div className="qiq-qcard">
            <div className="qiq-qcard-head">
              <span className="qiq-qcard-num">Question {selectedQuestion.number}</span>
              <span className="qiq-qcard-marks">
                {selectedMark === null || selectedMark === undefined ? "—" : selectedMark}
                <span style={{ color: C.faint, fontWeight: 600 }}> / {selectedQuestion.maxMarks ?? "—"}</span>
                {selectedOverridden && (
                  <span style={{ display: "block", fontSize: 12, color: C.blue, fontWeight: 600 }}>
                    ✎ set by teacher
                  </span>
                )}
              </span>
            </div>
            <div className="qiq-qcard-text">{selectedQuestion.questionText || "Question text unavailable"}</div>

            <div className="qiq-qcard-where">
              {(() => {
                const st = answerStatus(selectedQuestion, {
                  hasUnassignedWriting,
                  lowConfidence: LOW_CONFIDENCE,
                });
                if (st === ANSWER_STATUS.FAILED)
                  return <span style={{ color: C.red }}>This question still needs a mark.</span>;
                if (st === ANSWER_STATUS.UNANSWERED)
                  return <span style={{ color: C.faint }}>The student did not attempt this question.</span>;
                if (st === ANSWER_STATUS.NOT_DETECTED)
                  return (
                    <span style={{ color: C.amber }}>
                      No answer was matched to this question. Please check the answer sheet.
                    </span>
                  );
                if (placedRegions.length === 0)
                  return (
                    <span style={{ color: C.amber }}>
                      ~ The answer was read, but this page's writing could not be measured — the answer as read
                      is below, and this question's marks are listed in the margin.
                    </span>
                  );

                const exact = placedRegions.every((r) => pageTrusted(r.page));
                const where =
                  placedRegions.length === 1
                    ? `page ${placedRegions[0].page}`
                    : `pages ${placedRegions.map((r) => r.page).join(", ")}`;
                return (
                  <span style={{ color: exact ? C.green : C.amber }}>
                    {exact ? "✓" : "~"} Answer outlined on {where}
                    {exact ? "" : " — placement approximate, the outline may sit a line or two out"}
                    {" · "}
                    {placedVisible} of {visibleIdx.length} mark{visibleIdx.length === 1 ? "" : "s"} pinned to the
                    handwriting
                  </span>
                );
              })()}
            </div>

            {selectedQuestion.rationale && (
              <p className="qiq-qcard-why">
                <span style={{ fontWeight: 700, color: C.dim }}>Why this mark: </span>
                {selectedQuestion.rationale}
              </p>
            )}

            {/* The answer itself, as the machine read it. When the page could
                not be measured this is the only honest way to show "the answer
                is here" — and when it could, it is still what the marks were
                actually given for. */}
            {selectedQuestion.answerText && (
              <details className="qiq-qcard-answer" open={placedRegions.length === 0}>
                <summary>
                  Student’s answer
                  {Number.isFinite(selectedQuestion.confidence)
                    ? ` · ${selectedQuestion.confidence}% reading certainty`
                    : ""}
                </summary>
                <p>{selectedQuestion.answerText}</p>
              </details>
            )}

            {/* What was checked in it — the marks list on the right shows the
                comments, this shows the judgement behind them. */}
            {(selectedQuestion.correctPoints?.length ||
              selectedQuestion.incorrectPoints?.length ||
              selectedQuestion.missingPoints?.length) > 0 && (
              <ul className="qiq-qcard-points">
                {(selectedQuestion.correctPoints || []).map((p, j) => (
                  <li key={`c${j}`} style={{ color: C.green }}>
                    ✓ <span style={{ color: C.text }}>{p}</span>
                  </li>
                ))}
                {(selectedQuestion.incorrectPoints || []).map((p, j) => (
                  <li key={`e${j}`} style={{ color: C.red }}>
                    ✗ <span style={{ color: C.text }}>{p}</span>
                  </li>
                ))}
                {(selectedQuestion.missingPoints || []).map((p, j) => (
                  <li key={`m${j}`} style={{ color: "#A855F7" }}>
                    ! <span style={{ color: C.text }}>{p}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="qiq-pages">
          {pages.map((page, i) => {
            const pageNumber = i + 1;
            const boxes = boxesForPage(pageNumber);
            const region = regionFor(pageNumber);
            const shown = boxes.filter((b) => b.idx < revealed).length;
            const trusted = pageTrusted(pageNumber);
            const inSelection = !selectedPages || selectedPages.includes(pageNumber);

            return (
              <figure
                key={page.id}
                id={`qiq-page-${pageNumber}`}
                className="qiq-page-block"
                style={
                  selectedPages && !inSelection
                    ? { opacity: 0.35 } /* other pages stay for context, dimmed */
                    : undefined
                }
              >
                <div className="qiq-page-stage">
                  <img className="qiq-page-img" src={page.dataUrl} alt={`Page ${pageNumber}`} />

                  {/* The selected question's own writing, outlined from the same
                      line measurements the marks are placed with. Withheld on a
                      page whose measurement is untrusted rather than drawn
                      around the wrong paragraph. */}
                  {region && regionPlaceable(pageNumber) && (
                    <div
                      id={`qiq-region-${pageNumber}`}
                      className={`qiq-ans-region${trusted ? "" : " is-approx"}`}
                      style={{
                        left: `${Math.max(0, region.bbox.x - 0.015) * 100}%`,
                        top: `${Math.max(0, region.bbox.y - 0.01) * 100}%`,
                        width: `${Math.min(1, region.bbox.width + 0.03) * 100}%`,
                        height: `${Math.min(1, region.bbox.height + 0.02) * 100}%`,
                      }}
                    >
                      <span className="qiq-ans-region-tag">
                        Q{selectedQuestion.number}
                        {!trusted && " · approximate"}
                      </span>
                    </div>
                  )}

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
                          {hasMarks(ann) ? (
                            <span style={{ color: C.dim }}>
                              {" "}
                              · {ann.marks} mark{Math.abs(Number(ann.marks)) === 1 ? "" : "s"}
                            </span>
                          ) : (
                            <span style={{ color: C.faint }}> · marks not broken down</span>
                          )}
                          <span style={{ display: "block", marginTop: 5, color: C.text }}>
                            {ann.comment}
                          </span>
                          {unsure && (
                            <span
                              style={{ display: "block", marginTop: 6, color: C.amber, fontSize: 13.5 }}
                            >
                              ⚠ Please check this mark
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
                    {trusted
                      ? `${shown} mark${shown === 1 ? "" : "s"}` +
                        (geometryByPage[pageNumber] === "high" ? "" : " · approximate placement")
                      : "placement unreliable — this page's marks are listed in the margin instead"}
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
        isUnpinned={(idx) => !annBoxes[idx] || !annBoxes[idx].some((b) => pageTrusted(b.page))}
        filterQuestionId={selectedQuestion ? selectedQuestion.questionId : null}
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
            <div style={{ fontSize: 15, fontWeight: 700 }}>
              Student's answer, marked
              {markingInProgress && <span className="qiq-marking-dot" />}
            </div>
            <div style={{ fontSize: 13.5, color: C.faint, marginTop: 2 }}>
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
                  {hasMarks(seg.ann) ? (
                    <span style={{ color: C.dim }}>
                      {" "}
                      · {seg.ann.marks} mark{Math.abs(Number(seg.ann.marks)) === 1 ? "" : "s"}
                    </span>
                  ) : (
                    <span style={{ color: C.faint }}> · marks not broken down</span>
                  )}
                  <span style={{ display: "block", marginTop: 5, color: C.text }}>{seg.ann.comment}</span>
                  {unsure && (
                    <span style={{ display: "block", marginTop: 6, color: C.amber, fontSize: 13.5 }}>
                      ⚠ Please check this mark
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
function MarginNotes({ annotations, revealed, activeAnn, setActiveAnn, isUnpinned, filterQuestionId = null }) {
  const visible = filterQuestionId
    ? annotations.filter((a) => a.questionId === filterQuestionId)
    : annotations;
  return (
    <div className="qiq-margin">
      <div
        style={{
          fontSize: 13,
          fontWeight: 700,
          letterSpacing: 0.7,
          textTransform: "uppercase",
          color: C.dim,
          marginBottom: 10,
        }}
      >
        Teacher's margin notes
        {filterQuestionId && (
          <span style={{ color: C.faint, fontWeight: 500, textTransform: "none" }}>
            {" "}
            — this question only
          </span>
        )}
      </div>

      {visible.length === 0 && (
        <div style={{ fontSize: 14, color: C.faint }}>
          {filterQuestionId ? "No inline comments for this question." : "No inline comments were returned."}
        </div>
      )}

      {annotations.map((ann, idx) => {
        if (idx >= revealed) return null; // slides in when its highlight lands
        if (filterQuestionId && ann.questionId !== filterQuestionId) return null;

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
                  fontSize: 13,
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
              <div style={{ fontSize: 13.5, color: C.faint, fontStyle: "italic", marginBottom: 4 }}>
                “{String(ann.text).slice(0, 90)}
                {String(ann.text).length > 90 ? "…" : ""}”
              </div>
            )}
            <div className="qiq-handwrite" style={{ fontSize: 16, lineHeight: 1.55, color: C.text }}>
              {ann.comment}
            </div>
            {unsure && (
              <div style={{ marginTop: 6, fontSize: 13, color: C.amber, lineHeight: 1.5 }}>
                Please check this mark
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------- report card: score arc -- */

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

/* ---------------------------------------------------------- Tab: evaluate -- */

/**
 * The examiner's workbench. The AI has already read the paper; this screen is
 * where the human decides. Every question gets a card with the proposed mark
 * and — more importantly — the rationale for it, the points that earned or
 * cost marks, and the student's answer as the machine read it. Anything the
 * pipeline could not settle (a skipped answer, a failed marking, writing it
 * could not place, low confidence) is called out rather than smoothed over.
 *
 * The final mark belongs to the examiner: editing a mark here updates the
 * report card, the totals and the grade everywhere else in the app.
 */
function ReviewPanel({
  evaluation,
  keyPoints,
  markOverrides,
  onOverride,
  markNotice,
  awarded,
  total,
  grade,
  hasPages,
  hasReference,
  geometryByPage = {},
  onViewOnPage,
  onRetryEval,
  onShowRaw,
}) {
  const warnings =
    evaluation.paper && Array.isArray(evaluation.paper.warnings) ? evaluation.paper.warnings : [];
  const questions =
    evaluation.paper && Array.isArray(evaluation.paper.questions) ? evaluation.paper.questions : [];
  const hasUnassignedWriting = !!evaluation.hasUnassignedWriting;

  /* One status vocabulary for the whole tab. "unanswered" is a claim about the
     student and needs the strong case; anything weaker is a detection problem
     the examiner must look at, not a zero. */
  const statusOf = (k) => {
    const s = answerStatus(k, { hasUnassignedWriting, lowConfidence: LOW_CONFIDENCE });
    if (s === ANSWER_STATUS.FAILED) return { label: "Marking incomplete", color: C.red };
    if (s === ANSWER_STATUS.NOT_DETECTED) return { label: "Check answer sheet", color: C.amber };
    if (s === ANSWER_STATUS.UNANSWERED) return { label: "Question not attempted", color: C.faint };
    if (s === ANSWER_STATUS.UNCERTAIN) return { label: "Needs review", color: C.amber };
    if (k.marksTotal > 0 && k.marksAwarded >= k.marksTotal) return { label: "Full marks", color: C.green };
    if (k.marksAwarded > 0) return { label: "Partial", color: C.amber };
    return { label: "Zero", color: C.red };
  };

  /* Answer coverage, the workbench's opening fact: how much of the paper the
     system could actually see before any mark was proposed. */
  const statusKey = (k) => answerStatus(k, { hasUnassignedWriting, lowConfidence: LOW_CONFIDENCE });
  const required = keyPoints.filter((k) => k.counted !== false);
  const notCounted = keyPoints.filter((k) => k.counted === false);
  const coverage = {
    detected: keyPoints.filter((k) => statusKey(k) === ANSWER_STATUS.DETECTED).length,
    uncertain: keyPoints.filter((k) => statusKey(k) === ANSWER_STATUS.UNCERTAIN).length,
    notDetected: keyPoints.filter((k) => statusKey(k) === ANSWER_STATUS.NOT_DETECTED).length,
    unanswered: keyPoints.filter((k) => statusKey(k) === ANSWER_STATUS.UNANSWERED).length,
    failed: keyPoints.filter((k) => statusKey(k) === ANSWER_STATUS.FAILED).length,
  };

  /* A question the paper's choice excluded is not a problem to solve: the
     student was never required to answer it, so it must not appear in the list
     of things needing the examiner's attention. */
  const attention = keyPoints
    .map((k, i) => ({ k, i }))
    .filter(({ k }) => k.counted !== false)
    .filter(({ k }) => statusKey(k) !== ANSWER_STATUS.DETECTED && statusKey(k) !== ANSWER_STATUS.UNANSWERED);

  const jump = (i) =>
    document.getElementById(`qiq-rev-${i}`)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="qiq-rev">
      {/* ----------------------------------------------------- review summary */}
      {keyPoints.length > 0 && (
        <div className="qiq-rev-coverage">
          <div className="qiq-rev-summary-main">
            <strong>Review summary</strong>
            {markNotice && <span className="qiq-mark-saved">{markNotice}</span>}
            {notCounted.length > 0 && (
              <span>{required.length} counted · {notCounted.length} not required</span>
            )}
          </div>
          <div className="qiq-rev-summary-counts">
            <span><strong>{coverage.detected}</strong> answers found</span>
            {coverage.unanswered > 0 && (
              <span><strong>{coverage.unanswered}</strong> not attempted</span>
            )}
            {(coverage.uncertain + coverage.notDetected + coverage.failed) > 0 && (
              <span className="needs-check">
                <strong>{coverage.uncertain + coverage.notDetected + coverage.failed}</strong> need checking
              </span>
            )}
          </div>
          {(warnings.length > 0 || attention.length > 0) && (
            <div className="qiq-rev-summary-action">
              Check questions listed as not attempted or unclear before finalising.
              <button className="qiq-rev-link" onClick={onShowRaw}>Check answer text →</button>
            </div>
          )}
        </div>
      )}

      {/* --------------------------------------------------- question index */}
      {keyPoints.length > 1 && (
        <div className="qiq-rev-index-wrap">
          <span className="qiq-rev-index-label">Questions</span>
          <div className="qiq-rev-index">
            {keyPoints.map((k, i) => {
              const s = statusOf(k);
              return (
                <button
                  key={i}
                  className="qiq-rev-chip"
                  style={{ borderColor: `${s.color}88`, color: s.color }}
                  title={`${k.questionNumber ? "Question " + k.questionNumber : "Point " + (i + 1)} — ${s.label}, ${
                    k.marksAwarded ?? 0
                  }/${k.marksTotal ?? "—"}`}
                  onClick={() => jump(i)}
                >
                  {k.questionNumber || i + 1}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- per-question */}
      {keyPoints.map((k, i) => {
        const s = statusOf(k);
        const st = statusKey(k);
        const q = questions[i] || {};
        const overridden = Number.isFinite(markOverrides[i]);
        const max = Number(k.marksTotal);
        const editable = Number.isFinite(max) && max > 0;
        const title = (k.point || "").replace(/^Q\S*\.\s*/, "");
        const aiMark = Number(k.aiMarks !== undefined ? k.aiMarks : k.marksAwarded) || 0;

        /* Where this answer physically lives, and how much to trust it. */
        const pageRange =
          Number.isFinite(k.pageStart) && k.pageStart > 0
            ? k.pageEnd && k.pageEnd !== k.pageStart
              ? `pages ${k.pageStart}–${k.pageEnd}`
              : `page ${k.pageStart}`
            : null;
        const canShowOnPage = hasPages && k.questionId && pageRange && st !== ANSWER_STATUS.NOT_DETECTED && st !== ANSWER_STATUS.UNANSWERED;

        return (
          <div
            key={i}
            id={`qiq-rev-${i}`}
            className={`qiq-rev-card${k.counted === false ? " is-uncounted" : ""}`}
            style={{ borderLeftColor: k.counted === false ? C.faint : s.color }}
          >
            <div className="qiq-rev-head">
              <div style={{ minWidth: 0, flex: 1 }}>
                <span className="qiq-rev-qnum" style={{ color: s.color }}>
                  {k.questionNumber ? `Q${k.questionNumber}` : `Point ${i + 1}`}
                </span>
                <span className="qiq-rev-qtext" title={title}>
                  {title || "—"}
                </span>
              </div>

              <span
                className="qiq-chip"
                style={{ color: s.color, borderColor: `${s.color}55`, background: `${s.color}14` }}
              >
                {s.label}
              </span>

              {/* Marked, shown, and not counted. The student chose more answers
                  than the paper asked for, or this one was not among their best
                  — either way it is not a failure and must not read like one. */}
              {k.counted === false && (
                <span
                  className="qiq-chip"
                  style={{ color: C.blue, borderColor: `${C.blue}55`, background: `${C.blue}14` }}
                  title="The paper's choice means this answer does not count towards the total. Raise its mark above a counted one and it will."
                >
                  Not counted
                </span>
              )}

              <span className="qiq-rev-markbox">
                <span className="qiq-rev-marklabel">Final mark</span>
                {/* A failed evaluation is "no mark yet", never a zero. The box
                    stays empty until the examiner enters one. */}
                <input
                  className="qiq-exam-marks"
                  type="number"
                  min="0"
                  max={editable ? max : undefined}
                  step="0.5"
                  disabled={!editable}
                  placeholder={st === ANSWER_STATUS.FAILED ? "—" : undefined}
                  value={overridden ? markOverrides[i] : st === ANSWER_STATUS.FAILED ? "" : k.marksAwarded ?? 0}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") {
                      onOverride(i, null);
                      return;
                    }
                    const v = Number(raw);
                    if (!Number.isFinite(v)) return;
                    const clamped = Math.max(0, Math.min(editable ? max : v, v));
                    /* Typing the AI's own proposal back in is not a decision —
                       store nothing, so a later re-grade is not pinned to it. */
                    onOverride(i, st !== ANSWER_STATUS.FAILED && clamped === aiMark ? null : clamped);
                  }}
                  aria-label={`Final mark for ${k.questionNumber ? "question " + k.questionNumber : "point " + (i + 1)}`}
                />
                <span style={{ color: C.faint, fontSize: 14 }}>/ {k.marksTotal ?? "—"}</span>
              </span>
            </div>

            {q.answerText && (
              <details className="qiq-rev-answer qiq-rev-answer-first" open>
                <summary><span className="qiq-answer-icon" aria-hidden="true">“</span>Student’s answer</summary>
                <p>{q.answerText}</p>
              </details>
            )}

            {/* Explain the mark after the teacher has seen the student's answer. */}
            {(st === ANSWER_STATUS.UNANSWERED || st === ANSWER_STATUS.NOT_DETECTED || k.rationale || q.rationale) && (
              <p className="qiq-rev-rationale">
                <span className="qiq-rev-rationale-label">Why this mark</span>
                {st === ANSWER_STATUS.UNANSWERED
                  ? `Question not attempted.${hasReference ? "" : " No reference material was provided."}`
                  : st === ANSWER_STATUS.NOT_DETECTED
                  ? `No answer was matched to this question. Please check the answer sheet before awarding zero marks.${hasReference ? "" : " No reference material was provided."}`
                  : k.rationale || q.rationale}
              </p>
            )}

            {/* Keep exceptional notes after the reason so every card starts consistently. */}
            {k.questionId && (st === ANSWER_STATUS.UNCERTAIN || st === ANSWER_STATUS.FAILED) && (
              <div className="qiq-rev-statusline">
                {st === ANSWER_STATUS.UNCERTAIN ? (
                  <span>Answer may need checking{pageRange ? ` (${pageRange})` : ""}</span>
                ) : null}
                {st === ANSWER_STATUS.FAILED && (
                  <span>Marking incomplete. Enter a mark or try again.</span>
                )}
              </div>
            )}

            {overridden && (
              <div className="qiq-rev-changed">
                Mark changed to {markOverrides[i]} —{" "}
                {st === ANSWER_STATUS.FAILED ? "no original mark was available." : `the original suggestion was ${aiMark}.`}
                <button className="qiq-rev-link" onClick={() => onOverride(i, null)}>
                  {st === ANSWER_STATUS.FAILED ? "Clear" : "Restore suggested mark"}
                </button>
              </div>
            )}

            {st !== ANSWER_STATUS.UNANSWERED && st !== ANSWER_STATUS.NOT_DETECTED && Boolean(k.correctPoints?.length || k.missingPoints?.length || k.incorrectPoints?.length) && (
              <div className="qiq-rev-lists">
                {k.correctPoints?.length > 0 && (
                  <ul className="qiq-rev-list">
                    {k.correctPoints.filter((p) => typeof p === "string" && p.trim()).map((p, j) => (
                      <li key={`c${j}`} style={{ color: C.green }}>
                        <span style={{ color: C.text }}>{p}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {k.incorrectPoints?.length > 0 && (
                  <ul className="qiq-rev-list">
                    {k.incorrectPoints.filter((p) => typeof p === "string" && p.trim()).map((p, j) => (
                      <li key={`e${j}`} style={{ color: C.red }}>
                        <span style={{ color: C.text }}>{p}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {k.missingPoints?.length > 0 && (
                  <ul className="qiq-rev-list">
                    {k.missingPoints.filter((p) => typeof p === "string" && p.trim()).map((p, j) => (
                      <li key={`m${j}`} style={{ color: "#A855F7" }}>
                        <span style={{ color: C.text }}>{p}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {st !== ANSWER_STATUS.UNANSWERED && (
            <div className="qiq-rev-foot">
              {!hasReference && (st === ANSWER_STATUS.DETECTED || st === ANSWER_STATUS.UNCERTAIN) ? (
                <span className="qiq-reference-note">
                  Answer found. No reference material was provided, so this mark was based on subject knowledge.
                </span>
              ) : k.grounding && st !== ANSWER_STATUS.NOT_DETECTED ? (
                <GroundingBadge grounding={k.grounding} hasReference={hasReference} />
              ) : null}
              {Number.isFinite(k.confidence) && isLowConfidence(k) && st !== ANSWER_STATUS.FAILED && st !== ANSWER_STATUS.UNANSWERED && st !== ANSWER_STATUS.NOT_DETECTED && (
                <span style={{ fontSize: 12.5, color: isLowConfidence(k) ? C.amber : C.faint }}>
                  Please check this reading
                </span>
              )}
              {st === ANSWER_STATUS.FAILED && (
                <button className="qiq-rev-link" onClick={onRetryEval}>
                  Try marking again
                </button>
              )}
              {(st === ANSWER_STATUS.NOT_DETECTED || st === ANSWER_STATUS.UNCERTAIN) && (
                <button className="qiq-rev-link" onClick={onShowRaw}>
                  Check possible writing
                </button>
              )}
              {canShowOnPage ? (
                <button className="qiq-rev-link" onClick={() => onViewOnPage(k)}>
                  View on answer sheet
                </button>
              ) : null}
            </div>
            )}
          </div>
        );
      })}

      {keyPoints.length === 0 && (
        <p style={{ color: C.faint, fontSize: 14 }}>
          No per-question breakdown was produced for this paper — see the Report Card tab.
        </p>
      )}

      {/* --------------------------------------------------- examiner total */}
      {keyPoints.length > 0 && (
        <div className="qiq-rev-total">
          <span style={{ color: C.dim, fontSize: 14 }}>Examiner's total</span>
          <span style={{ fontWeight: 800, fontSize: 18, color: gradeColor(grade) }}>
            {awarded} / {total}
          </span>
          <span style={{ fontSize: 14, color: C.faint }}>
            grade {grade}
            {notCounted.length > 0 &&
              ` · out of the ${required.length} question${required.length === 1 ? "" : "s"} this paper required`}
            {Object.keys(markOverrides).length > 0 &&
              ` · ${Object.keys(markOverrides).length} mark(s) adjusted by you`}
            {coverage.failed > 0 &&
              ` · ${coverage.failed} question${coverage.failed === 1 ? "" : "s"} still need a mark (counted as 0 above until you enter one)`}
          </span>
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------- Tab 4: report card -- */

function ReportCard({
  evaluation,
  keyPoints,
  awarded,
  total,
  grade,
  overrideCount = 0,
  runKey,
  studentName,
  setStudentName,
  subject,
  setSubject,
  reportDate,
  setReportDate,
  hasReference,
  studentDetails,
  setStudentDetails,
}) {
  const g = gradeColor(grade || evaluation.grade);
  const positives = Array.isArray(evaluation.thingsWellDone) ? evaluation.thingsWellDone : [];
  const improvements = Array.isArray(evaluation.improvementAreas) ? evaluation.improvementAreas : [];
  const remark = String(evaluation.overallRemark || "");
  const typed = useTypewriter(remark, TYPE_MS, runKey);
  const speech = useSpeech();

  /* The printed report speaks the same status vocabulary as the Evaluate tab.
     It must never print "not covered" where the workbench said "we could not
     read it" — the report is the copy that reaches the student. */
  const hasUnassignedWriting = !!evaluation.hasUnassignedWriting;
  const rows = keyPoints.map((k) => {
    const st = answerStatus(k, { hasUnassignedWriting, lowConfidence: LOW_CONFIDENCE });
    /* A failed evaluation is "no mark yet", not a zero. Only an
       examiner-entered mark turns the dash into a number. */
    return { k, st, unevaluated: st === ANSWER_STATUS.FAILED && !k.overridden, counted: k.counted !== false };
  });
  const notCounted = rows.filter((r) => !r.counted);

  const unevaluated = rows.filter((r) => r.unevaluated && r.counted);
  const pending = unevaluated.reduce((sum, r) => sum + (Number(r.k.marksTotal) || 0), 0);

  /* PASS/FAIL is a verdict, and a verdict cannot be issued while marks are
     still missing. If the pending marks alone could carry the paper over the
     line, the honest stamp is "pending" — a FAIL there would be the report
     card asserting something nobody has actually marked. */
  const passed = total > 0 && awarded / total >= PASS_THRESHOLD;
  const undecided = !passed && total > 0 && pending > 0 && (awarded + pending) / total >= PASS_THRESHOLD;
  const verdict = passed ? "pass" : undecided ? "pending" : "fail";

  const chipFor = ({ k, st, unevaluated: pendingRow, counted }) => {
    /* Marked, printed, and not part of the score. The paper let the student
       choose, and this answer was not among the ones that count. */
    if (!counted) return { label: "Not counted", color: C.blue };
    if (pendingRow) return { label: "Mark needed", color: C.red };
    if (st === ANSWER_STATUS.NOT_DETECTED) return { label: "Check answer sheet", color: C.amber };
    if (st === ANSWER_STATUS.UNANSWERED) return { label: "Not attempted", color: C.faint };
    if (st === ANSWER_STATUS.UNCERTAIN) return { label: "Needs review", color: C.amber };
    if (k.quality === "well" || (k.covered && !k.quality)) return { label: "Covered well", color: C.green };
    if (k.quality === "partially") return { label: "Partial", color: C.amber };
    return { label: "Not covered", color: C.red };
  };

  return (
    <div>
      <div className="qiq-report-actions qiq-noprint">
        <button className="qiq-btn qiq-btn-sm" onClick={() => window.print()}>
          Print report
        </button>
        <span style={{ fontSize: 13.5, color: C.faint }}>
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
              <div className="qiq-report-sub">QIQ · Teacher-reviewed descriptive assessment</div>
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
          {studentDetails.map((row) => (
            <label className="qiq-rfield" key={row.id}>
              <input
                className="qiq-rlabel-input qiq-noprint"
                value={row.label}
                onChange={(e) =>
                  setStudentDetails((rows) =>
                    rows.map((item) => item.id === row.id ? { ...item, label: e.target.value } : item)
                  )
                }
                placeholder="Detail name"
                aria-label="Report detail name"
              />
              <span className="qiq-print-only">{row.label || "Additional detail"}</span>
              <input
                className="qiq-rinput"
                value={row.value}
                onChange={(e) =>
                  setStudentDetails((rows) =>
                    rows.map((item) => item.id === row.id ? { ...item, value: e.target.value } : item)
                  )
                }
                placeholder="—"
              />
            </label>
          ))}
        </div>

        <div className="qiq-report-ready qiq-noprint">
          <strong>Ready for teacher review</strong>
          <span>Check the final marks and student details before printing or sharing.</span>
        </div>

        {/* ------------------------------------------------- score + stamp */}
        <div className="qiq-report-score">
          <ScoreArc awarded={awarded} total={total} runKey={runKey} />

          <div className="qiq-grade-block">
            <div className="qiq-grade-stamp" key={`g${runKey}`} style={{ borderColor: g, color: g }}>
              {grade || evaluation.grade || "—"}
            </div>
            <div className="qiq-grade-caption">Grade awarded</div>
          </div>

          <div
            className={`qiq-passmark is-${verdict}`}
            key={`p${runKey}`}
            title={
              verdict === "pending"
                ? `Pass mark is ${Math.round(PASS_THRESHOLD * 100)}%. ${pending} mark(s) have not been entered yet and could change this result.`
                : `Pass mark is ${Math.round(PASS_THRESHOLD * 100)}%`
            }
          >
            {verdict === "pending" ? "PENDING" : verdict === "pass" ? "PASS" : "FAIL"}
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
                {rows.map((row, i) => {
                  const { k, unevaluated } = row;
                  const chip = chipFor(row);
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
                        <strong style={{ color: chip.color, opacity: row.counted ? 1 : 0.6 }}>
                          {unevaluated ? "—" : k.marksAwarded ?? 0}
                        </strong>
                        <span style={{ color: C.faint }}> / {k.marksTotal ?? "—"}</span>
                        {!row.counted && (
                          <div style={{ fontSize: 12, color: C.blue, whiteSpace: "nowrap" }}>
                            not in the total
                          </div>
                        )}
                        {k.overridden && (
                          <div
                            style={{ fontSize: 12, color: C.blue, whiteSpace: "nowrap" }}
                            title={`The suggested mark was ${k.aiMarks ?? 0}; the teacher set ${k.marksAwarded}.`}
                          >
                            ✎ set by teacher
                          </div>
                        )}
                      </td>
                      <td style={{ color: C.dim, lineHeight: 1.55 }}>
                        {row.st === ANSWER_STATUS.UNANSWERED
                          ? `Question not attempted.${hasReference ? "" : " No reference material was provided."}`
                          : row.st === ANSWER_STATUS.NOT_DETECTED
                          ? `No answer was matched to this question. Please check the answer sheet.${hasReference ? "" : " No reference material was provided."}`
                          : k.teacherNote}
                        {/* Where the judgement came from. Present only for the
                            question-paper pipeline; the older flow has no
                            reference material to attribute to. */}
                        {row.st !== ANSWER_STATUS.UNANSWERED && row.st !== ANSWER_STATUS.NOT_DETECTED && (
                          <div style={{ marginTop: 6, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                            {!hasReference && (row.st === ANSWER_STATUS.DETECTED || row.st === ANSWER_STATUS.UNCERTAIN) ? (
                              <span className="qiq-reference-note">
                                Answer found. No reference material was provided, so this mark was based on subject knowledge.
                              </span>
                            ) : k.grounding ? (
                              <GroundingBadge grounding={k.grounding} hasReference={hasReference} />
                            ) : null}
                            {k.evidence && k.evidence.length > 0 && (
                              <span style={{ fontSize: 12.5, color: C.faint }}>
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
                      No mark-by-mark details are available.
                    </td>
                  </tr>
                )}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2} style={{ fontWeight: 700 }}>
                    Total
                    {pending > 0 && (
                      <div style={{ fontSize: 12, fontWeight: 500, color: C.red }}>
                        {pending} mark{pending === 1 ? "" : "s"} across {unevaluated.length} question
                        {unevaluated.length === 1 ? "" : "s"} still need to be entered and count as 0 here
                      </div>
                    )}
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
            <div style={{ fontSize: 14, fontWeight: 700, color: C.green, marginBottom: 10 }}>
              What was done well
            </div>
            {positives.length === 0 && <div style={{ fontSize: 14, color: C.faint }}>No positives recorded.</div>}
            <ul className="qiq-list">
              {positives.map((t, i) => (
                <li key={i}>
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="qiq-listcard" style={{ borderTop: `2px solid ${C.amber}` }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: C.amber, marginBottom: 10 }}>
              Areas to improve
            </div>
            {improvements.length === 0 && (
              <div style={{ fontSize: 14, color: C.faint }}>No improvement areas recorded.</div>
            )}
            <ul className="qiq-list">
              {improvements.map((t, i) => (
                <li key={i}>
                  <span>
                    {t}{" "}
                    <a
                      className="qiq-study qiq-noprint"
                      href={`https://www.google.com/search?q=${encodeURIComponent(t)}+explanation+for+students`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Study this topic
                    </a>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="qiq-report-foot">
          Prepared with QIQ · Please review any mark labelled “Needs review” before sharing this report.
          {notCounted.length > 0 &&
            ` · This paper let the student choose: ${rows.length - notCounted.length} of ${rows.length} ` +
              `question(s) count, and the total is out of those. The rest are marked for the record.`}
          {unevaluated.length > 0 &&
            ` · ${unevaluated.length} question(s) still need teacher marking — their ${pending} mark(s) are pending, not zero.`}
          {verdict === "pending" &&
            " · The pass/fail verdict is withheld until those marks are entered: the paper can still pass."}
          {overrideCount > 0 &&
            ` · ${overrideCount} mark${overrideCount === 1 ? "" : "s"} on this report were set by the teacher instead of using the original suggestion.`}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- Tab 3: raw -- */

function RawView({ pageTexts, setPageTexts, geometryByPage = {}, dirty, onReEvaluate }) {
  const text = pageTexts.join("\n\n");
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  /* One box per page, not one box for the paper. The page a line sits on is what
     its ink measurement is attached to, and a single merged box gave the user no
     way to correct a word without putting those boundaries at risk. */
  const editPage = (i, value) =>
    setPageTexts(pageTexts.map((t, j) => (j === i ? value : t)));

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="qiq-subhead">
        <div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>Answer text read from the paper</div>
          <div style={{ fontSize: 13.5, color: C.faint, marginTop: 2 }}>
            {words} words · {text.length} characters
            {pageTexts.length > 1 ? ` · ${pageTexts.length} pages, edited separately` : ""} · fix any
            unclear or misread words before marking again
          </div>
        </div>
        {dirty && (
          <button className="qiq-btn qiq-btn-sm" onClick={onReEvaluate}>
            Mark corrected text again
          </button>
        )}
      </div>

      {pageTexts.length <= 1 ? (
        <textarea
          className="qiq-mono"
          value={pageTexts[0] || ""}
          onChange={(e) => editPage(0, e.target.value)}
          spellCheck={false}
        />
      ) : (
        pageTexts.map((pageText, i) => {
          const level = geometryByPage[i + 1] || "none";
          return (
            <div key={i} className="qiq-rawpage">
              <div className="qiq-rawpage-head">
                <span>Page {i + 1}</span>
                <span style={{ color: C.faint }}>
                  {pageText.trim() ? pageText.trim().split(/\s+/).length : 0} words ·{" "}
                  {level === "high" || level === "medium"
                    ? "marks can be placed on this page"
                    : "this page's writing could not be measured"}
                </span>
              </div>
              <textarea
                className="qiq-mono"
                style={{ minHeight: 160 }}
                value={pageText}
                onChange={(e) => editPage(i, e.target.value)}
                spellCheck={false}
              />
            </div>
          );
        })
      )}

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
  font-size: 16px;
}
.qiq-root *, .qiq-root *::before, .qiq-root *::after { box-sizing: border-box; }

.qiq-header {
  display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
  background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(15,23,42,.7));
  border:1px solid ${C.border}; border-radius:16px; padding:14px 18px; margin-bottom:16px;
  backdrop-filter: blur(8px);
}
.qiq-brand-title { font-size:19px; font-weight:750; letter-spacing:-.25px; }
.qiq-brand-subtitle { font-size:15px; color:${C.dim}; margin-top:3px; line-height:1.4; }
.qiq-logo {
  width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
  background: linear-gradient(135deg, ${C.blue}, ${C.purple});
  font-weight:800; font-size:19px; color:#fff; box-shadow:0 6px 20px rgba(37,99,235,.38);
}

.qiq-pipeline { display:flex; align-items:center; flex-wrap:wrap; }
.qiq-pipe-item { display:flex; align-items:center; gap:8px; }
.qiq-pipe-dot {
  width:24px; height:24px; border-radius:50%; display:grid; place-items:center;
  font-size:13px; font-weight:700; background:#0B1220; color:${C.faint};
  border:1px solid ${C.border}; transition:.3s;
}
.qiq-pipe-dot.is-done { background:${C.blue}; border-color:${C.blue}; color:#fff; }
.qiq-pipe-dot.is-active {
  background: linear-gradient(135deg, ${C.blue}, ${C.purple}); border-color:transparent; color:#fff;
  animation: qiq-pulse 2s infinite;
}
.qiq-pipe-line { width:34px; height:2px; margin:0 10px; border-radius:2px; transition:.4s; }
@keyframes qiq-pulse { 0%,100%{box-shadow:0 0 0 4px rgba(37,99,235,.18)} 50%{box-shadow:0 0 0 8px rgba(37,99,235,.06)} }

.qiq-grid { display:grid; grid-template-columns: minmax(420px, 470px) minmax(0,1fr); gap:18px; align-items:start; }
@media (max-width: 1080px) { .qiq-grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) {
  .qiq-root { padding:10px; }
  .qiq-header { padding:13px; }
  .qiq-grid > aside.qiq-panel { padding:16px; }
  .qiq-panel { border-radius:13px; }
  .qiq-brand-title { font-size:16px; }
  .qiq-brand-subtitle { font-size:14px; }
  .qiq-pipeline { width:100%; justify-content:space-between; }
  .qiq-pipe-line { width:14px; margin:0 5px; }
  .qiq-pipe-item span { font-size:13px !important; }
  .qiq-empty { padding:36px 8px; }
  .qiq-empty-title { font-size:21px; }
}
.qiq-panel {
  background: ${C.card}; border:1px solid ${C.border}; border-radius:16px; padding:18px;
  box-shadow: 0 12px 40px rgba(0,0,0,.28);
  min-width: 0; /* a grid child must be allowed to shrink, or wide content escapes */
  overflow: hidden;
}
.qiq-grid > aside.qiq-panel { padding:22px; }
.qiq-right { min-height: 560px; display:flex; flex-direction:column; overflow:visible; }

.qiq-step-num {
  width:24px; height:24px; border-radius:7px; display:grid; place-items:center;
  background: rgba(37,99,235,.18); color:#B8CCFF; font-size:14px; font-weight:800;
  border:1px solid rgba(37,99,235,.3);
}
.qiq-input {
  width:100%; background:#0B1220; border:1px solid ${C.border}; border-radius:10px;
  color:${C.text}; padding:13px 14px; font-size:15px; font-family:inherit; outline:none;
  transition: border-color .18s, box-shadow .18s;
}
.qiq-input:focus { border-color:${C.blue}; box-shadow:0 0 0 3px rgba(37,99,235,.16); }
/* Placeholders are deliberately italic and dim so they can never be mistaken
   for text the teacher actually typed. */
.qiq-input::placeholder { color:#455066; font-style:italic; }
.qiq-textarea { min-height:150px; resize:vertical; line-height:1.65; }
.qiq-collapsible {
  margin-top:20px; border:1px solid ${C.border}; border-radius:12px; background:#0B1220;
  overflow:hidden;
}
.qiq-collapsible-summary {
  display:flex; align-items:center; gap:10px; padding:13px 14px; cursor:pointer;
  list-style:none; user-select:none; transition:background .16s;
}
.qiq-collapsible-summary::-webkit-details-marker { display:none; }
.qiq-collapsible-summary:hover { background:rgba(37,99,235,.07); }
.qiq-collapsible-summary:focus-visible { outline:2px solid ${C.blue}; outline-offset:-2px; }
.qiq-collapsible-title { color:${C.text}; font-size:15px; font-weight:750; white-space:nowrap; }
.qiq-collapsible-value {
  margin-left:auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  color:${C.dim}; font-size:13px;
}
.qiq-collapsible-chevron {
  margin-left:auto; color:#93B4FF; font-size:18px; font-weight:800; transition:transform .18s;
}
.qiq-collapsible-value + .qiq-collapsible-chevron { margin-left:4px; }
.qiq-collapsible[open] .qiq-collapsible-chevron { transform:rotate(180deg); }
.qiq-collapsible-body {
  display:grid; gap:9px; padding:0 14px 14px; border-top:1px solid ${C.border}; padding-top:13px;
}

.qiq-invalid { border-color:${C.red} !important; box-shadow:0 0 0 3px rgba(239,68,68,.14) !important; }
.qiq-hint { margin-top:7px; font-size:14px; color:${C.dim}; line-height:1.55; }
.qiq-optional-note { padding:0 2px; }
.qiq-mini-btn {
  background: rgba(37,99,235,.14); border:1px solid rgba(37,99,235,.34); color:#93B4FF;
  font-size:13px; font-weight:700; padding:3px 9px; border-radius:99px; cursor:pointer;
  font-family:inherit; transition:.15s; letter-spacing:.2px; white-space:nowrap;
}
.qiq-mini-btn:hover { background: rgba(37,99,235,.26); color:#C7D8FF; }

.qiq-drop {
  display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
  border:1.5px dashed #385071; border-radius:14px; padding:28px 20px; cursor:pointer;
  background: linear-gradient(180deg, rgba(37,99,235,.04), transparent); transition:.2s;
  overflow:hidden; max-width:100%;
}
.qiq-drop:hover, .qiq-drop.is-dragging {
  border-color:${C.blue}; background: rgba(37,99,235,.09); transform: translateY(-1px);
}
.qiq-upload-icon {
  width:30px; height:30px; border-radius:9px; display:grid; place-items:center; margin-bottom:9px;
  color:#DCE8FF; background:rgba(37,99,235,.24); border:1px solid rgba(96,165,250,.45);
  font-size:15px; font-weight:800;
}
.qiq-upload-icon.is-optional { background:rgba(148,163,184,.12); border-color:${C.borderSoft}; color:${C.dim}; }
.qiq-upload-title { font-size:15px; line-height:1.35; font-weight:750; color:${C.text}; }
.qiq-upload-title.is-large { font-size:16px; }
.qiq-upload-body { width:100%; text-align:center; }
.qiq-upload-copy { max-width:360px; margin-left:auto; margin-right:auto; font-size:15px; color:${C.dim}; line-height:1.55; margin-top:6px; }
.qiq-upload-action {
  display:inline-flex; margin-top:12px; padding:7px 13px; border-radius:8px;
  background:${C.blue}; color:#fff; font-size:14px; font-weight:750;
}
.qiq-upload-action.is-secondary { background:rgba(37,99,235,.15); color:#B8CCFF; border:1px solid rgba(96,165,250,.3); }
.qiq-upload-confirm {
  display:flex; align-items:flex-start; gap:8px; margin:9px 0; padding:9px 11px;
  border-radius:9px; background:rgba(34,197,94,.07); border:1px solid rgba(34,197,94,.25);
  color:#BBF7D0; font-size:14px; line-height:1.45;
}
.qiq-upload-confirm > span:first-child {
  width:18px; height:18px; flex:0 0 18px; display:grid; place-items:center; border-radius:50%;
  background:rgba(34,197,94,.2); color:#86EFAC; font-size:13px; font-weight:900;
}
.qiq-drop-icon {
  width:40px; height:40px; border-radius:12px; display:grid; place-items:center; margin-bottom:10px;
  background: linear-gradient(135deg, rgba(37,99,235,.22), rgba(124,58,237,.22));
  border:1px solid rgba(37,99,235,.32); font-size:17px; color:#A9C3FF;
}
.qiq-file {
  display:flex; align-items:center; gap:10px; background:#0B1220;
  border:1px solid ${C.border}; border-radius:10px; padding:8px 10px;
  min-width:0; max-width:100%; overflow:hidden;
}
.qiq-thumb {
  width:34px; height:44px; object-fit:contain; border-radius:6px;
  border:1px solid ${C.border}; flex-shrink:0; background:#fff; max-width:100%;
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
  border-radius:11px; padding:15px 16px; font-size:15px; font-weight:750; cursor:pointer;
  font-family:inherit; box-shadow:0 8px 24px rgba(37,99,235,.3); transition:.18s;
}
.qiq-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow:0 12px 30px rgba(37,99,235,.42); }
.qiq-btn:disabled { opacity:.65; cursor:progress; box-shadow:none; }
.qiq-btn-ghost {
  margin-top:0; background:#0B1220; border:1px solid ${C.border}; color:${C.dim};
  box-shadow:none; font-weight:600; font-size:14px; padding:10px 12px;
}
.qiq-btn-ghost:hover:not(:disabled) { color:${C.text}; border-color:${C.blue}; transform:none; box-shadow:none; }
.qiq-btn-sm { width:auto; margin-top:0; padding:9px 15px; font-size:14px; }
.qiq-link {
  background:none; border:none; color:#93B4FF; font-size:14px; cursor:pointer;
  padding:0; font-family:inherit; text-decoration:underline; text-underline-offset:3px;
}

.qiq-error {
  margin-top:16px; background: rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.32);
  color:#FCA5A5; border-radius:10px; padding:11px 13px; font-size:14px; line-height:1.6;
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
  text-align:center; padding:52px 24px;
}
.qiq-empty-icon {
  width:58px; height:58px; border-radius:18px; display:grid; place-items:center; font-size:24px;
  background: linear-gradient(135deg, rgba(37,99,235,.16), rgba(124,58,237,.16));
  border:1px solid ${C.border}; margin-bottom:18px;
}
.qiq-empty-kicker { color:#93B4FF; font-size:14px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:10px; }
.qiq-empty-title { font-size:24px; font-weight:800; color:${C.text}; letter-spacing:-.4px; }
.qiq-empty-copy { font-size:16px; color:${C.dim}; max-width:540px; line-height:1.7; margin:10px 0 22px; }
.qiq-setup-list { width:min(100%, 560px); display:grid; gap:10px; text-align:left; }
.qiq-setup-item {
  display:flex; align-items:center; gap:13px; padding:13px 15px; border:1px solid ${C.border};
  border-radius:12px; background:#0B1220; color:${C.text};
}
.qiq-setup-item.is-done { border-color:rgba(34,197,94,.32); background:rgba(34,197,94,.055); }
.qiq-setup-status {
  width:30px; height:30px; flex:0 0 30px; border-radius:50%; display:grid; place-items:center;
  border:1px solid ${C.borderSoft}; color:${C.dim}; font-size:14px; font-weight:800;
}
.qiq-setup-item.is-done .qiq-setup-status { background:rgba(34,197,94,.18); border-color:rgba(34,197,94,.4); color:#86EFAC; }
.qiq-setup-item strong { display:block; font-size:16px; }
.qiq-setup-item small { display:block; color:${C.dim}; font-size:14px; margin-top:3px; line-height:1.4; }
.qiq-privacy-note { margin-top:18px; color:${C.faint}; font-size:13.5px; }
.qiq-teacher-review {
  display:flex; gap:11px; align-items:flex-start; margin-bottom:14px; padding:12px 14px;
  border-radius:11px; background:rgba(37,99,235,.07); border:1px solid rgba(96,165,250,.26);
}
.qiq-teacher-review > span {
  width:24px; height:24px; flex:0 0 24px; display:grid; place-items:center; border-radius:50%;
  background:rgba(37,99,235,.2); color:#BFDBFE; font-weight:900;
}
.qiq-teacher-review strong { display:block; color:${C.text}; font-size:14px; }
.qiq-teacher-review small { display:block; color:${C.dim}; font-size:13px; line-height:1.5; margin-top:2px; }
.qiq-orb {
  width:38px; height:38px; border-radius:50%; flex-shrink:0;
  background: conic-gradient(from 0deg, ${C.blue}, ${C.purple}, ${C.blue});
  mask: radial-gradient(circle 13px at center, transparent 98%, #000 100%);
  -webkit-mask: radial-gradient(circle 13px at center, transparent 98%, #000 100%);
  animation: qiq-spin 1.1s linear infinite;
}
.qiq-rawpage { display:grid; gap:6px; }
.qiq-rawpage-head {
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:13px; font-weight:700; color:${C.dim}; letter-spacing:.4px;
}

.qiq-cut {
  margin-top:10px; display:grid; gap:6px; background: rgba(245,158,11,.09);
  border:1px solid rgba(245,158,11,.3); border-radius:10px; padding:9px 11px;
  font-size:13.5px; line-height:1.6; color:#FCD34D;
}
.qiq-cut strong { color:#FDE68A; }

.qiq-notice {
  margin-top:16px; max-width:440px; background: rgba(245,158,11,.1);
  border:1px solid rgba(245,158,11,.32); color:#FCD34D; border-radius:10px;
  padding:10px 13px; font-size:14px; line-height:1.6; animation: qiq-fade .25s ease;
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
  border-bottom:2px solid transparent; color:${C.faint}; font-size:15px; font-weight:600;
  padding:10px 14px; cursor:pointer; font-family:inherit; transition:.18s; margin-bottom:-3px;
}
.qiq-tab:hover { color:${C.dim}; }
.qiq-tab.is-active { color:${C.text}; border-bottom-color:${C.blue}; }
.qiq-tab-badge {
  font-size:12.5px; font-weight:800; color:#04121F; border-radius:99px; padding:2px 7px;
}
.qiq-tab-warn {
  font-size:12.5px; font-weight:800; color:#FCD34D; background: rgba(245,158,11,.16);
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
.qiq-marked-layout { grid-template-columns:minmax(0, 1fr); }
.qiq-marked-layout > .qiq-margin {
  max-height:none; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));
  gap:12px; overflow:visible; padding-right:0;
}
.qiq-marked-layout > .qiq-margin > :first-child { grid-column:1 / -1; margin-bottom:0 !important; }
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
  font-size:12px; font-weight:800; color:#06101F; vertical-align:super; line-height:1.5;
  letter-spacing:.2px; box-shadow:0 2px 8px rgba(0,0,0,.4);
  animation: qiq-badge-pop .34s cubic-bezier(.34,1.56,.64,1) both .12s;
}
@keyframes qiq-badge-pop {
  0% { transform: scale(0) translateY(4px); opacity:0; }
  100% { transform: scale(1) translateY(0); opacity:1; }
}
.qiq-warn {
  display:inline-flex; align-items:center; margin-left:6px; padding:2px 7px; border-radius:999px;
  border:1px solid rgba(245,158,11,.25); background:rgba(245,158,11,.07);
  color:#D9B46D; font-size:12px; font-weight:700; cursor:help; vertical-align:middle;
}
.qiq-warn.is-compact { font-size:11.5px; }

.qiq-pop {
  position:absolute; bottom:calc(100% + 9px); left:0; z-index:30; width:280px;
  background:#111C33; border:1px solid ${C.borderSoft}; border-radius:10px; padding:10px 12px;
  font-size:14px; line-height:1.6; white-space:normal; color:${C.dim};
  box-shadow:0 14px 40px rgba(0,0,0,.55); opacity:0; visibility:hidden;
  transform: translateY(4px); transition:.16s; pointer-events:none;
}
.qiq-hl:hover .qiq-pop { opacity:1; visibility:visible; transform:none; }

/* ------------------------------------------------- question paper + refs */
.qiq-drop-sm { padding:18px 18px; min-height:0; text-align:center; }

.qiq-exam { border:1px solid ${C.border}; border-radius:12px; background:#0B1220; padding:12px 13px; }
.qiq-choice {
  display:flex; align-items:center; gap:7px; flex-wrap:wrap; margin-top:10px;
  padding:8px 10px; border-radius:9px; font-size:13.5px; color:${C.text};
  background: rgba(37,99,235,.08); border:1px solid rgba(37,99,235,.28);
}
.qiq-choice .qiq-exam-marks { width:52px; }
.qiq-choice-src { margin-left:auto; color:${C.faint}; font-size:12px; }
.qiq-exam-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.qiq-exam-total { text-align:right; flex-shrink:0; line-height:1.2; }
.qiq-exam-rows { display:grid; gap:4px; margin-top:11px; max-height:230px; overflow:auto; }
.qiq-exam-row {
  display:grid; grid-template-columns: 56px minmax(0,1fr) 64px;
  column-gap:0; row-gap:7px; align-items:center; font-size:13.5px;
  padding:8px 10px; border-radius:8px; background:#0F172A;
}
.qiq-exam-row.is-missing { background: rgba(245,158,11,.10); }
.qiq-exam-num { font-weight:800; color:${C.text}; }
.qiq-exam-text {
  color:${C.dim}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  margin-left:7px; margin-right:20px;
}
.qiq-exam-marks {
  width:64px; background:#0A0F1E; border:1px solid ${C.borderSoft}; border-radius:7px;
  color:${C.text}; font-size:13.5px; padding:3px 5px; text-align:center; font-family:inherit;
}
.qiq-exam-row.is-missing .qiq-exam-marks { border-color:${C.amber}; }
.qiq-missing-marks {
  margin-top:12px; padding:11px; border-radius:10px; background:rgba(245,158,11,.08);
  border:1px solid rgba(245,158,11,.38);
}
.qiq-missing-marks.is-complete { background:rgba(34,197,94,.07); border-color:rgba(34,197,94,.32); }
.qiq-missing-marks.is-complete .qiq-missing-icon { background:${C.green}; color:#052E16; }
.qiq-missing-marks.is-complete .qiq-missing-marks-head strong { color:#BBF7D0; }
.qiq-missing-marks-head { display:flex; align-items:flex-start; gap:9px; color:${C.text}; }
.qiq-missing-marks-head strong { display:block; font-size:14px; color:#FDE68A; }
.qiq-missing-marks-head small { display:block; margin-top:2px; color:${C.dim}; font-size:13.5px; line-height:1.45; }
.qiq-missing-icon {
  width:20px; height:20px; flex:0 0 20px; display:grid; place-items:center; border-radius:50%;
  background:${C.amber}; color:#241603; font-size:14px; font-weight:900;
}
.qiq-missing-marks-fields { display:grid; gap:7px; margin-top:10px; }
.qiq-missing-mark-row { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:7px; }
.qiq-missing-mark-field {
  display:flex; align-items:center; justify-content:space-between; gap:8px; min-width:0;
  padding:7px 8px 7px 10px; border-radius:8px;
  background:#0B1220; border:1px solid rgba(245,158,11,.28); color:${C.text};
  font-size:14px; font-weight:700;
}
.qiq-missing-question { min-width:0; flex:1; }
.qiq-missing-question strong { display:block; white-space:nowrap; font-size:14px; }
.qiq-missing-question small {
  display:-webkit-box; margin-top:2px; overflow:hidden; -webkit-box-orient:vertical;
  -webkit-line-clamp:2; color:${C.dim}; font-size:12.5px; font-weight:500; line-height:1.3;
}
.qiq-missing-mark-field .qiq-exam-marks {
  width:72px; padding:6px 7px; font-size:15px; border-color:${C.amber};
}
@media (max-width: 520px) {
  .qiq-missing-mark-row { grid-template-columns:1fr; }
}
.qiq-exam-warn { margin-top:10px; display:grid; gap:6px; }
.qiq-exam-warnrow {
  display:flex; gap:7px; align-items:flex-start; font-size:13.5px; line-height:1.55;
  color:${C.text}; background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.28);
  border-radius:8px; padding:7px 9px;
}

/* ---- evaluate tab: the examiner's workbench ---- */
.qiq-rev { display:grid; gap:12px; }
.qiq-rev-alert {
  border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.07);
  border-radius:12px; padding:12px 14px;
}
.qiq-rev-alertrow {
  display:flex; gap:8px; align-items:flex-start; font-size:14px; line-height:1.6;
  color:${C.text}; margin-top:5px;
}
.qiq-rev-link {
  background:none; border:none; padding:0; color:${C.blue}; font-size:inherit; font-family:inherit;
  cursor:pointer; text-decoration:underline; text-underline-offset:2px;
}
.qiq-rev-index-wrap {
  display:flex; align-items:center; gap:14px; padding:11px 14px; border:1px solid ${C.border};
  border-radius:11px; background:#0B1220;
}
.qiq-rev-index-label {
  color:${C.text}; font-size:14px; font-weight:800; flex-shrink:0;
}
.qiq-rev-index { display:flex; flex-wrap:wrap; gap:9px; }
.qiq-rev-chip {
  min-width:48px; min-height:38px; padding:7px 12px; border-radius:9px; border:1px solid ${C.borderSoft};
  background:#111B30; font-size:15px; font-weight:800; font-family:inherit; cursor:pointer;
  box-shadow:0 1px 0 rgba(255,255,255,.035); transition:background .15s, transform .15s;
}
.qiq-rev-chip:hover { background:#17243D; transform:translateY(-1px); }
.qiq-rev-card {
  border:1px solid ${C.border}; border-left:3px solid ${C.borderSoft}; border-radius:12px;
  background:${C.card}; padding:17px 18px; scroll-margin-top:12px;
}
.qiq-rev-head {
  display:grid; grid-template-columns:minmax(0,1fr) auto auto; align-items:center; gap:14px;
}
.qiq-rev-qnum { font-weight:800; font-size:15px; margin-right:7px; }
.qiq-rev-qtext {
  color:${C.text}; font-size:15px; line-height:1.55; white-space:normal;
  max-width:100%; vertical-align:top;
}
.qiq-rev-markbox {
  display:inline-flex; align-items:center; gap:7px; width:max-content;
  margin-left:auto; padding:0; background:transparent; border:0;
}
.qiq-rev-marklabel {
  color:${C.dim}; font-size:11.5px; font-weight:800; letter-spacing:.35px; text-transform:uppercase;
  white-space:nowrap;
}
.qiq-rev-markbox .qiq-exam-marks { width:52px; font-size:14px; padding:6px 7px; }
.qiq-rev-rationale {
  font-size:15px; line-height:1.7; color:${C.text}; margin:16px 0 0;
  padding:12px 14px; border-radius:9px; background:#0B1220; border:1px solid ${C.border};
}
.qiq-rev-rationale-label {
  display:block; margin-bottom:5px; color:#C7D8FF; font-size:12.5px;
  font-weight:800; letter-spacing:.5px; text-transform:uppercase;
}
.qiq-rev-lists { display:grid; gap:4px; margin-top:9px; }
.qiq-rev-list { list-style:none; margin:0; padding:0; display:grid; gap:3px; font-size:14px; line-height:1.55; }
.qiq-rev-answer { margin-top:10px; font-size:13.5px; color:${C.faint}; }
.qiq-rev-answer-first { margin-top:16px; }
.qiq-rev-answer summary { cursor:pointer; color:#C7D8FF; font-weight:700; }
.qiq-rev-answer p {
  margin:7px 0 0; padding:9px 11px; border-radius:8px; background:#0A0F1E;
  color:${C.dim}; line-height:1.65; white-space:pre-wrap; max-height:220px; overflow:auto;
}
.qiq-rev-foot { display:flex; align-items:center; gap:10px; margin-top:11px; flex-wrap:wrap; }
.qiq-rev-foot .qiq-rev-link { margin-left:auto; font-size:13.5px; }
.qiq-reference-note {
  color:${C.dim}; font-size:13px; line-height:1.5;
}
.qiq-rev-total {
  display:flex; align-items:baseline; gap:12px; justify-content:flex-end;
  border-top:1px solid ${C.border}; padding-top:11px;
}
.qiq-rev-card.is-uncounted { opacity:.72; }

.qiq-rev-coverage {
  border:1px solid ${C.border}; border-radius:12px; background:#0B1220; padding:14px 16px;
}
.qiq-rev-summary-main {
  display:flex; align-items:baseline; justify-content:space-between; gap:12px; color:${C.text};
}
.qiq-rev-summary-main strong { font-size:15px; }
.qiq-rev-summary-main span { color:${C.faint}; font-size:13px; }
.qiq-rev-summary-counts {
  display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:9px; color:${C.dim}; font-size:14px;
}
.qiq-rev-summary-counts strong { color:${C.text}; font-size:16px; margin-right:3px; }
.qiq-rev-summary-counts .needs-check { color:#D9B46D; }
.qiq-rev-summary-action {
  display:flex; justify-content:space-between; align-items:center; gap:16px; margin-top:11px;
  padding-top:10px; border-top:1px solid ${C.border}; color:${C.dim}; font-size:13.5px;
}
.qiq-rev-statusline {
  margin-top:10px; padding:8px 11px; border-radius:8px;
  background:rgba(245,158,11,.055); border:1px solid rgba(245,158,11,.18);
  color:#D9B46D; font-size:13.5px; line-height:1.5;
}
.qiq-rev-changed {
  margin-top:10px; color:${C.blue}; font-size:13px;
}
.qiq-rev-changed .qiq-rev-link { margin-left:6px; }
@media (max-width: 760px) {
  .qiq-rev-head { grid-template-columns:1fr auto; }
  .qiq-rev-head > div:first-child { grid-column:1 / -1; }
  .qiq-rev-summary-action { align-items:flex-start; flex-direction:column; gap:7px; }
}
/* ---- the run screen: a pipeline reporting on itself, not a loading bar ---- */
.qiq-run {
  display:grid; gap:14px; padding:22px 20px; border:1px solid ${C.border};
  border-radius:14px; background:${C.card}; width:100%;
}
.qiq-run-head { display:flex; align-items:center; gap:14px; }
.qiq-run-title { font-size:15px; font-weight:700; color:${C.text}; display:flex; align-items:baseline; gap:2px; }
.qiq-run-sub { font-size:13.5px; color:${C.faint}; margin-top:3px; line-height:1.5; }
.qiq-run-clock {
  margin-left:auto; font-size:15px; font-weight:700; color:${C.dim};
  font-variant-numeric: tabular-nums; letter-spacing:.5px;
}
.qiq-run-ell::after { content:"…"; animation: qiq-ell 1.4s steps(4,end) infinite; }
@keyframes qiq-ell { 0% { opacity:.2; } 50% { opacity:1; } 100% { opacity:.2; } }

.qiq-rail { display:flex; gap:6px; }
.qiq-rail-seg { flex:1; min-width:0; }
.qiq-rail-bar {
  display:block; height:3px; border-radius:2px; background:${C.border}; transition: background .3s;
}
.qiq-rail-seg.is-done .qiq-rail-bar { background:${C.green}; }
.qiq-rail-seg.is-now .qiq-rail-bar {
  background: linear-gradient(90deg, ${C.blue}, ${C.purple}, ${C.blue});
  background-size:200% 100%; animation: qiq-rail-run 1.6s linear infinite;
}
@keyframes qiq-rail-run { from { background-position:0 0; } to { background-position:200% 0; } }
.qiq-rail-label {
  display:block; margin-top:6px; font-size:12px; letter-spacing:.4px; text-transform:uppercase;
  color:${C.faint}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.qiq-rail-seg.is-now .qiq-rail-label { color:${C.text}; }

.qiq-trace {
  max-height:min(46vh, 340px); overflow:auto; border:1px solid ${C.border}; border-radius:10px;
  background:${C.navy}; padding:9px 4px 9px 10px; display:grid; gap:2px; align-content:start;
}
.qiq-trace-idle { font-size:13.5px; color:${C.faint}; padding:4px 2px; }
.qiq-trace-row {
  display:grid; grid-template-columns: 40px 14px minmax(0, auto) 1fr; gap:8px; align-items:baseline;
  font-size:13.5px; line-height:1.7; padding:1px 0;
  animation: qiq-trace-in .22s ease-out both;
}
@keyframes qiq-trace-in { from { opacity:0; transform: translateY(3px); } to { opacity:1; transform:none; } }
.qiq-trace-time { color:${C.faint}; font-variant-numeric: tabular-nums; font-size:12.5px; }
.qiq-trace-mark { font-weight:800; color:${C.dim}; }
.qiq-trace-row.is-ok .qiq-trace-mark { color:${C.green}; }
.qiq-trace-row.is-warn .qiq-trace-mark { color:${C.amber}; }
.qiq-trace-row.is-start .qiq-trace-mark { color:${C.blue}; }
.qiq-trace-text { color:${C.text}; font-weight:600; }
.qiq-trace-detail { color:${C.faint}; min-width:0; overflow-wrap:anywhere; }
.qiq-trace-row.is-live .qiq-trace-text { color:${C.dim}; font-weight:500; }
.qiq-trace-row.is-live { opacity:.9; }

.qiq-proc-coverage {
  margin-top:20px; font-size:14px; color:${C.text};
  border:1px solid ${C.borderSoft}; border-radius:10px; padding:9px 14px; background:#0B1220;
}

.qiq-reflist { display:grid; gap:5px; margin-top:8px; }
.qiq-refitem {
  display:flex; gap:8px; align-items:center; font-size:13.5px; color:${C.dim};
  background:#0B1220; border:1px solid ${C.border}; border-radius:8px; padding:6px 9px;
}
.qiq-reficon { flex-shrink:0; }

.qiq-ground {
  display:inline-block; font-size:12.5px; font-weight:750; letter-spacing:.1px;
  padding:3px 8px; border-radius:99px; vertical-align:middle;
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

/* the selected question's answer, outlined where it actually sits */
.qiq-ans-region {
  position:absolute; border:2px dashed ${C.blue}; border-radius:8px;
  background: rgba(96,165,250,.07); pointer-events:none; line-height:normal;
  box-shadow: 0 0 0 9999px rgba(2,6,23,.42);
  animation: qiq-region-in .35s ease-out both;
}
.qiq-ans-region-tag {
  position:absolute; top:-10px; left:8px; padding:1px 7px; border-radius:6px;
  background:${C.blue}; color:#F8FAFC; font-size:12px; font-weight:800; letter-spacing:.4px;
}
.qiq-ans-region.is-approx { border-style:dotted; border-color:${C.amber}; background: rgba(245,158,11,.06); }
.qiq-ans-region.is-approx .qiq-ans-region-tag { background:${C.amber}; color:#1A1206; }
@keyframes qiq-region-in { from { opacity:0; } to { opacity:1; } }

.qiq-qcard-answer { margin-top:8px; font-size:14px; }
.qiq-qcard-answer summary { cursor:pointer; color:${C.dim}; font-size:13.5px; }
.qiq-qcard-answer p {
  margin:6px 0 0; padding:8px 10px; border-radius:8px; background:${C.navy};
  border:1px solid ${C.border}; color:${C.text}; line-height:1.65; white-space:pre-wrap;
  max-height:150px; overflow:auto;
}
.qiq-qcard-points { list-style:none; margin:9px 0 0; padding:0; display:grid; gap:4px; font-size:13.5px; }
.qiq-qcard-points li { display:flex; gap:7px; line-height:1.55; }

/* the question behind the highlights, above the pages */
.qiq-qcard {
  border:1px solid #2D405D; border-left:4px solid ${C.blue}; border-radius:12px;
  background:#111B30; padding:16px 18px; margin-bottom:16px;
}
.qiq-qcard-head { display:flex; gap:12px; align-items:flex-start; justify-content:space-between; }
.qiq-qcard-num {
  display:inline-flex; padding:5px 9px; border-radius:7px; background:rgba(37,99,235,.16);
  border:1px solid rgba(96,165,250,.3); font-size:15px; font-weight:800; color:#93B4FF;
}
.qiq-qcard-text {
  margin-top:12px; max-width:1100px; font-size:16px; font-weight:600;
  color:#F1F5F9; line-height:1.65;
}
.qiq-qcard-marks {
  flex-shrink:0; text-align:right; font-size:18px; font-weight:800; color:${C.text};
  font-variant-numeric: tabular-nums;
}
.qiq-qcard-where {
  font-size:15px; margin-top:13px; line-height:1.55; padding:9px 11px;
  border-radius:8px; background:#0B1426;
}
.qiq-qcard-why {
  font-size:16px; color:#CBD5E1; line-height:1.75; margin:12px 0 0;
  padding:12px 14px; border-radius:9px; background:rgba(148,163,184,.06);
  border:1px solid ${C.border};
}
.qiq-page-cap {
  display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;
  font-size:13px; color:${C.dim}; padding:7px 3px 0; line-height:1.5;
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
  padding:1px 6px; border-radius:99px; font-size:12px; font-weight:800;
  color:#06101F; line-height:1.5; letter-spacing:.2px; white-space:nowrap;
  box-shadow:0 2px 8px rgba(0,0,0,.45);
  animation: qiq-badge-pop .34s cubic-bezier(.34,1.56,.64,1) both .12s;
}
/* Writing that runs to the page edge leaves no room outside it, so the badge
   tucks back inside rather than being clipped by the stage. */
.qiq-ann-box.is-tight .qiq-ann-badge { left:auto; right:3px; transform:translate(0,-50%); }
.qiq-ann-warn {
  position:absolute; right:100%; top:50%; transform:translate(-5px,-50%);
  font-size:13px; line-height:1; pointer-events:none;
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
  border-radius:10px; padding:10px 12px; font-size:14px; line-height:1.6; color:${C.text};
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
  font-size:12px; font-weight:800; color:#06101F;
}
.qiq-note-marks {
  padding:1px 7px; border-radius:99px; font-size:12px; font-weight:800; color:#06101F;
}
/* The teacher's comment. This used to be set in a cursive "handwriting" face to
   suggest a marked-up page, but the comment is the single most important thing
   a student reads here, and a script face at 14px on a dark background is hard
   work — especially for the long, specific feedback the new per-question marking
   produces. The marked-up feel is carried by the colour, the left rule and the
   badges instead; the words themselves are plain. */
.qiq-handwrite {
  font-family: inherit;
  font-size:15px;
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
.qiq-report-sub { font-size:13.5px; color:${C.faint}; margin-top:2px; }
.qiq-report-serial {
  font-size:13px; color:${C.faint}; letter-spacing:1.4px; text-transform:uppercase;
  font-variant-numeric: tabular-nums;
}

.qiq-report-fields {
  display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:16px; margin:20px 0 4px;
}
@media (max-width: 780px) { .qiq-report-fields { grid-template-columns: 1fr; } }
.qiq-rfield { display:block; }
.qiq-rfield > span {
  display:block; font-size:12px; letter-spacing:1.1px; text-transform:uppercase;
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
.qiq-arc-pct { font-size:13px; color:${C.faint}; margin-top:2px; letter-spacing:.6px; }

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
  font-size:12px; letter-spacing:1.2px; text-transform:uppercase;
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
/* the third state: marks are missing, so no verdict has been earned yet */
.qiq-passmark.is-pending {
  color:${C.amber}; background: rgba(245,158,11,.07);
  font-size:15px; letter-spacing:1.5px;
}
@keyframes qiq-stamp-in {
  0%   { transform: scale(0) rotate(-24deg); opacity:0; }
  60%  { transform: scale(1.1) rotate(1deg); opacity:1; }
  100% { transform: scale(1) rotate(-3deg); opacity:.92; }
}

.qiq-report-section { margin-top:26px; }
.qiq-report-h {
  display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  font-size:13px; letter-spacing:1.2px; text-transform:uppercase;
  color:${C.dim}; font-weight:800; margin-bottom:12px;
}
.qiq-voice { display:inline-flex; align-items:center; gap:6px; margin-left:auto; }
.qiq-speaking {
  font-size:12.5px; color:${C.blue}; text-transform:none; letter-spacing:0;
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
  margin-top:14px; text-align:right; font-size:15px; color:${C.faint}; font-style:italic;
}
.qiq-caret {
  display:inline-block; width:2px; height:1em; background:${C.purple};
  margin-left:2px; vertical-align:-2px; animation: qiq-blink .8s steps(1) infinite;
}

.qiq-table-wrap { overflow-x:auto; border:1px solid ${C.border}; border-radius:12px; }
.qiq-table { width:100%; border-collapse:collapse; font-size:15px; min-width:640px; }
.qiq-table th {
  text-align:left; font-size:12.5px; letter-spacing:.8px; text-transform:uppercase;
  color:${C.dim}; font-weight:700; padding:11px 14px; background:#0B1220;
  border-bottom:1px solid ${C.border};
}
.qiq-table td { padding:13px 14px; border-bottom:1px solid rgba(30,41,59,.7); vertical-align:top; }
.qiq-table tbody tr:hover { background: rgba(37,99,235,.05); }
.qiq-table tfoot td { background:#0B1220; border-bottom:none; font-size:15px; }
.qiq-chip {
  display:inline-block; font-size:13px; font-weight:700; padding:3px 9px;
  border-radius:99px; border:1px solid; white-space:nowrap;
}

.qiq-two-col { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:26px; }
@media (max-width: 760px) { .qiq-two-col { grid-template-columns: 1fr; } }
.qiq-listcard { background:#0B1220; border:1px solid ${C.border}; border-radius:12px; padding:16px 18px; }
.qiq-list { list-style:none; padding:0; margin:0; display:grid; gap:9px; }
.qiq-list li { display:flex; gap:9px; font-size:15px; line-height:1.6; color:${C.dim}; }
.qiq-study {
  color:#93B4FF; text-decoration:none; font-size:13.5px; font-weight:700;
  border-bottom:1px dotted #93B4FF; white-space:nowrap; margin-left:2px;
}
.qiq-study:hover { color:#C7D8FF; border-bottom-style:solid; }

.qiq-report-foot {
  margin-top:24px; padding-top:14px; border-top:1px solid ${C.border};
  font-size:12.5px; color:${C.faint}; line-height:1.6;
}

.qiq-mono {
  width:100%; min-height:420px; background:#0B1220; border:1px solid ${C.border};
  border-radius:12px; color:#CBD5E1; padding:18px 20px; resize:vertical; outline:none;
  font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size:15px; line-height:1.8; white-space:pre-wrap;
}
.qiq-mono:focus { border-color:${C.blue}; box-shadow:0 0 0 3px rgba(37,99,235,.16); }
.qiq-pre {
  margin-top:10px; background:#0B1220; border:1px solid ${C.border}; border-radius:10px;
  padding:14px; font-size:13.5px; color:${C.faint}; max-height:280px; overflow:auto;
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
