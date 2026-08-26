# QIQ — Descriptive Paper Checker

A browser application that reads a scanned or photographed exam answer sheet,
works out which answer belongs to which question, marks each answer against the
teacher's own reference material, and draws the marks back onto the student's
handwriting — then hands the whole thing to a human examiner who can override
any of it.

This document describes every part of the system and, more importantly, **why
each part is built the way it is**. The rules in [Design rules](#design-rules)
are the ones that shape everything else; if you only read one section, read that
one.

---

## Contents

1. [What the product actually does](#what-the-product-actually-does)
2. [Design rules](#design-rules)
3. [The stack and how to run it](#the-stack-and-how-to-run-it)
4. [The pipeline, end to end](#the-pipeline-end-to-end)
5. [Module reference](#module-reference)
   - [identity.js — which question is this?](#identityjs--which-question-is-this)
   - [ocr.js — text into a structured document](#ocrjs--text-into-a-structured-document)
   - [segment.js — splitting a document into questions](#segmentjs--splitting-a-document-into-questions)
   - [exam.js — the question paper as a schema](#examjs--the-question-paper-as-a-schema)
   - [choice.js — "answer any 3 of the following 5"](#choicejs--answer-any-3-of-the-following-5)
   - [match.js — which answer belongs to which question](#matchjs--which-answer-belongs-to-which-question)
   - [reference.js — local BM25 retrieval](#referencejs--local-bm25-retrieval)
   - [assess.js — marking one question at a time](#assessjs--marking-one-question-at-a-time)
   - [geometry.js — where the ink is](#geometryjs--where-the-ink-is)
   - [marking.js — putting marks back on the page](#markingjs--putting-marks-back-on-the-page)
   - [llm.js / json.js / groq.js — the model seam](#llmjs--jsonjs--groqjs--the-model-seam)
   - [keypool.js + vite.config.js — the key-scheduling proxy](#keypooljs--viteconfigjs--the-key-scheduling-proxy)
   - [pdf.js / docx.js / text.js / types.js](#pdfjs--docxjs--textjs--typesjs)
6. [The UI](#the-ui)
7. [Confidence, status, grounding and counting](#confidence-status-grounding-and-counting)
8. [Token budget and rate limits](#token-budget-and-rate-limits)
9. [Failure modes and how each is reported](#failure-modes-and-how-each-is-reported)
10. [Tests](#tests)
11. [Known limits](#known-limits)
12. [Extending the system](#extending-the-system)

---

## What the product actually does

The teacher supplies up to four things:

| Input | Required? | What it is used for |
|---|---|---|
| **Answer paper** | yes | The student's work. PDF, Word, or images. Rasterised to page images and read by a vision model. |
| **Question paper** | optional | Becomes the *schema*: how many questions exist, what each asks, what each is worth, and **how many the student had to attempt**. Without it the app falls back to a simpler single-scheme flow. |
| **Reference material** | optional | Textbook, notes, or model answers. Indexed locally and searched per question; a scan is transcribed first. Model answers filed by question number are paired directly. |
| **Marking scheme / answer key** | optional in exam mode, required otherwise | The teacher's own authority on what a correct answer says. |

The output is four views of one result: a **marked page image**, an **annotated
transcript**, an **examiner's review workbench** where every mark can be
overridden, and a **printable report card**.

---

## Design rules

These are not style preferences. Every one of them exists because violating it
produced a specific, damaging bug.

### 1. A model proposes; it never decides

Marks are recomputed in code from clamped values. A model that returns 8/5, or
whose parts do not add up to its own total, cannot corrupt the result.
`assessQuestion` clamps to the question's ceiling independently of the schema
that already clamped it, and `assessPaper` sums the paper total itself.

### 2. Never fabricate a number and present it as a measurement

This is the rule that has been broken most often, always the same way: a schema
default filling in for something the model did not say.

- `annotation.marks` defaulted to `0` → every correct highlight was stamped
  `+0`, a zero the student never earned.
- `annotation.confidence` defaulted to `70` → **every question on every paper
  reported exactly 70% confidence**, and it looked like a measurement.

Both fields are now `optional` with no default. Where nothing was reported,
nothing is claimed — the UI says *"marks not broken down"* rather than showing a
number nobody produced.

The one allocation that *is* permitted: if a question has one highlight and one
mark, that highlight carries the mark, because only one allocation is possible.
Two highlights and one total stay unmarked — choosing a split would be
inventing a breakdown.

### 3. A detection failure is never reported as a student failure

"We could not find this answer" and "the student did not write one" are
different claims. `answerStatus()` will only say `UNANSWERED` when nothing was
matched **and** there is no unassigned writing anywhere on the paper that could
be the missing answer. Otherwise the honest answer is `NOT_DETECTED`, which the
UI shows as *"Answer not detected — this does NOT mean the question was
unanswered"*.

### 4. A failed evaluation is not a zero

When marking a question throws, it is recorded as `failed: true` with a
placeholder mark of 0 — but every surface treats that as **pending**, not
awarded. The review list shows an empty mark box, the report card prints `—`,
the total states how many marks are still unevaluated, and the PASS/FAIL stamp
goes to a third state, **PENDING**, when the pending marks could still carry the
paper over the pass line.

### 5. Placement is withheld rather than guessed

A mark drawn on the wrong handwriting is worse than a mark in the margin.
`alignBandsToLines` reports `high`/`medium`/`low`/`none`, and per-annotation
boxes are drawn only for `high` and `medium`. A question's *region* — a whole
block, not a phrase — is drawn at `low` too, because being a line out around an
entire answer still points at the right paragraph; it is labelled
**approximate** and never passed off as exact.

### 6. Nothing is truncated silently

Every cap says what it dropped, where the user can see it:

- Uploads past the page cap raise a persistent notice naming the file, its true
  page count, and what the loss costs.
- A long answer key is addressed **by question number**, not cut at a character
  count — the old `slice(0, 4000)` meant the second half of a long paper was
  marked with no key at all and nobody was told.
- A matching call that fails names the questions it cost.
- A scanned reference is transcribed rather than refused, and labelled *"read as
  a scan"* so a weak retrieval result can be traced to its cause.

### 7. Question identity is resolved in exactly one place

See [identity.js](#identityjs--which-question-is-this). Three modules used to
parse question labels with their own regexes, so agreement between them was a
coincidence of formatting. The same rule now covers the rubric: what "answer any
three" means lives in [choice.js](#choicejs--answer-any-3-of-the-following-5)
and nowhere else.

The general form of the rule: **anything that depends on how a paper is written
gets one owner, and every other module asks it.** A shape that only works for
`1.` is a bug, not a limitation.

### 8. A student is scored out of what the paper required

Not out of what it printed. See [choice.js](#choicejs--answer-any-3-of-the-following-5).
A question excluded by the paper's own choice is still marked and still shown —
it is simply not part of the total, and is never displayed as a failure.

### 9. Every claim of "the reference says so" is verified

A model may label a mark `REFERENCE_SUPPORTED`, but that survives only if the
annotation cites an evidence id that was actually retrieved. Citations that name
nothing real are stripped and the grounding is downgraded — otherwise "the
textbook says so" becomes the easiest sentence in the world to hallucinate.

---

## The stack and how to run it

- **React 19 + Vite 7.** No UI framework, no state library, no CSS file — all
  styling is one template-literal stylesheet inside `PaperChecker.jsx`.
- **No runtime dependencies beyond React.** Retrieval, geometry, segmentation,
  schema validation and the key pool are all hand-written and dependency-free.
  pdf.js is loaded from a CDN at runtime, on demand.
- **Models** (Groq): `qwen/qwen3.6-27b` for vision/OCR, `openai/gpt-oss-120b`
  for every reasoning stage.

```bash
cp .env.example .env      # add GROQ_API_KEY (or GROQ_API_KEYS=a,b,c)
npm install
npm run dev               # vite dev server + the /api/groq proxy
npm test                  # all seven suites, ~330 assertions, no network
npm run build             # production bundle to dist/
```

**API keys never reach the browser.** The client posts to the same-origin
`/api/groq/chat/completions`; the Vite middleware attaches the `Authorization`
header server-side. `GROQ_API_KEY`, `GROQ_API_KEY_2…10`, or a comma-separated
`GROQ_API_KEYS` are all accepted; duplicates are dropped.

### Layout

```
index.html            mounts #root
main.jsx              React entry
PaperChecker.jsx      the entire UI: state, pipeline orchestration, styles (~4.5k lines)
vite.config.js        dev server + /api/groq proxy + /api/groq/stats
src/engine/           the pipeline — pure modules, no React, no DOM except where marked
  identity.js           what a question label means, and which question a block is
  choice.js             "answer any 3 of the following 5"
  exam.js               the question paper as a schema
  ocr.js segment.js     text into lines, lines into questions
  match.js              which answer belongs to which question
  reference.js          BM25 over the teacher's own material
  assess.js             marking, mark control, grounding, totals
  geometry.js marking.js  where the ink is, and putting marks back on it
  llm.js json.js groq.js  the model seam
  pdf.js docx.js text.js types.js
src/server/keypool.js key scheduling; server-side only, keys never leave the process
test/*.test.mjs       plain node scripts, no framework
```

---

## The pipeline, end to end

Two flows exist. **Exam mode** runs when a question paper has been read;
otherwise the original single-scheme flow runs. Exam mode is the real product.

```
                ┌── question paper (optional) ──────────────────────┐
                │  rasterise → vision OCR → extractExamWithLlm      │
                │  → structuralMarks   (independent mark cross-check)│
                │  → detectChoice      ("answer any 3 of the next 5")│
                │  → validateExam ──────────────────► exam schema    │
                └────────────────────────────────────────────────────┘
                                                          │
                ┌── reference material (optional) ─────────┼─────────┐
                │  text PDF  → chunkDocument → BM25 index            │
                │  scanned   → rasterise → vision OCR → same index   │
                │  raw text  → numberedSections → pairReferenceAnswers│
                │              ──────────────► answers by question id │
                └───────────────────────────────────────┬───────────┘
                                                        │
  answer paper                                          │
      │                                                 │
      ├─ rasterise (PDF/image → JPEG data URLs)          │
      ├─ runOcr        one vision call per page          │
      ├─ measurePages  deskew → ink projection per page  │  ← local, no model
      ├─ structureOcr  text → lines with page + offsets  │
      ├─ attachGeometry lines ← measured boxes           │
      │                                                  │
      ├─ matchAnswers ─────────────────────────────────► │
      │    route 1: align labelled blocks to the exam    │
      │    route 2: ask the model, in bounded batches    │
      │    route 3: report unmatched, never guess        │
      │                                                  │
      ├─ assessPaper   one model call per question ──────┘
      │    evidence = paired reference answer + BM25 hits
      │    marks clamped, grounding verified, totals summed in code
      │    applyChoice — only the required attempts reach the total
      │
      ├─ summarisePaper       the closing remark
      ├─ toEvaluation         project onto the legacy view shape
      ├─ anchorAnnotations    quoted phrase → character span
      └─ resolveAnnotationBoxes  span → lines → boxes on the page
```

Every stage writes to a **live trace** the user watches while it runs:

```
00:03  ✓ Read page 2 of 4          389 characters · 16 lines
00:09  ✓ Measured page 2           31 ink bands vs 34 lines — placement medium
                                   · page was 1.7° crooked, corrected
00:11  ▸ Applying the paper's choice  any 3 of 5 (Q1…Q5) — the total is out of
                                     15, not 25
00:14  ✓ Q4                        found by its written number · 95% confidence
                                   · page 2
00:41  ✓ Marked Q4                 3/5 marks · supported by the reference
                                   · 82% confidence
```

The trace is not decoration — it is how a user diagnoses which stage lost their
answer.

---

## Module reference

### identity.js — which question is this?

**The most important module in the system**, and the newest. Every stage has to
answer the same question: the answer sheet says `3(a)`, the question paper says
`Q.3 (a)`, the model answers say `Ans 3a`. Each stage used to answer it with its
own regular expression, so a paper numbered any other way — `Question Five`,
`1.1`, `Q. No. 7`, a Roman `iv` — parsed differently in each place and silently
failed to line up.

The fix is to **stop parsing labels in the abstract**. The question paper is the
schema: it states the exact set of questions that exist. Every other document is
*aligned to that set* rather than parsed independently and compared as strings.
"Which of these twelve known questions is this block" is a constrained problem
with a right answer; "what does this label syntax mean" is not.

```js
readLabel(line)   → {label, rest, bracketed, lead} | null
labelParts(raw)   → ["3","a","ii"]        the trail of levels named
labelKey(raw)     → "3.a.r2"              one comparable form
canonicalLabel()  → "3(a)(ii)"            the display form
alignToQuestions(blocks, questions) → {placed: Map<questionId, …>, unplaced: []}
```

**`readLabel` is a scanner, not a pattern table**, because label shapes compose:
stacked lead words, then levels, bracketed or dotted, to any depth. It reads
`1.` `1)` `Q1` `Q.1` `Q 1:` `Ans 1.` `Answer Two:` `Sol. 9` `Q. No. 7` `No. 4`
`Question Five` `3(a)` `3 (a)` `2 a)` `2a` `Question 12 (b)` `Q.3 (a) (ii)`
`(c)` `iii)` `ii.` — and **refuses** `A cell divides…`, `Five plants were
grown…`, `3.5 kg of water…`, `Now…`, `Note…`.

Prose defends itself with two rules:

- A token with nothing marking it as a label — no lead word, no bracket, no
  terminator — is not a label. `3a` survives (a digit followed by a letter is
  not how anyone writes a quantity); `1.1` does not, because `3.5 kg` is —
  **unless the paper being marked actually asks a question 1.1**. `readLabel`
  takes an optional `known` set of the exam's own label keys, and that closes the
  one genuinely ambiguous case: detection in isolation is ambiguous, detection
  against a known set is not.
- After a full stop, only a **digit or a bracket** continues the label. A bare
  letter after a stop is the first word of the answer far more often than a
  sub-level — `2. k = 3` and `1. A cell divides` were being read as labels
  `2.k` and `1.a`, mangling the label *and* eating the student's first word.

**`alignToQuestions` runs four layers, strongest first, each declaring what it
is worth:**

| Layer | Confidence | When |
|---|---|---|
| `label` | 95 | The labels agree once written in the same key space |
| `label-loose` | 88 | They agree once notation and depth are resolved (a key files `5`, the paper asks `5(a)`) |
| `label-partial` | 76 | Only one exam question could carry this label at all |
| `content` | 62 | No usable label; the text matches one question clearly better than any other |

One block claims one question. A block two questions fit equally well is left
**unplaced** — "we could not find this answer" is true, and picking one is not.
Roman numerals and letters are deliberately never merged: `(i)` and `(a)` are
different sub-parts on papers that use both.

### ocr.js — text into a structured document

`structureOcr(pageTexts)` turns per-page text into `{lines, pageCount, text,
quality}`. Each line carries `id`, `page`, `index`, `text`, `start`/`end`
offsets into the joined text, and a legibility `confidence`.

The offsets matter more than they look: `text` keeps the blank lines that
`lines` drops, so without exact offsets a character span could never be resolved
back to a line — and the whole annotation-placement chain depends on that hop.

The vision model returns no geometry, so `bbox` is left `undefined` rather than
fabricated; geometry.js measures it from the image instead.

The **OCR system prompt** is written for three downstream consumers, each of
which was previously left to chance:

1. the student's own question labels — the strongest matching route (95 vs ~70);
2. **one output line per written line** — what the ink measurement is aligned
   against; reflowed prose is why placement kept coming out `low`;
3. `[illegible]` markers — which `lineConfidence()` scores on, and which nothing
   was asking the transcriber to produce.

### segment.js — splitting a document into questions

`detectQuestionsStructural(lines)` finds heads, slices bodies between them, and
returns `ParsedQuestion[]` with `questionNumber`, `questionText`, `answerText`,
`answerLineIds`, `maxMarks`, `pageStart`/`pageEnd`, `parentId`.

It no longer owns any notation knowledge — it calls `readLabel` and asks
identity.js what each label means. It keeps the raw label alongside the
canonical one so the label can be stripped off the front of the answer text it
introduces. A bare `(c)` attaches to the number the last head *opened with*:
after `3(b)`, the parent is 3, not 3(b).

`extractMarks` reads `(5 marks)`, `[5]`, `5 M`. `allocateMarks` spreads a
declared total across questions that printed none.

### exam.js — the question paper as a schema

Two passes, and **the deterministic one is the cross-check rather than the
fallback**: the model reads the paper (numbering styles vary far too much for
regex alone), and `structuralMarks` confirms numbering and printed marks
independently. Disagreements become warnings, never silent corrections.

`validateExam` repairs nothing. A missing per-question mark stays missing and
blocks grading; a sum that disagrees with the declared total is reported as
both numbers. Inventing a plausible mark would mean marking against it for the
rest of the run.

`deriveMarks` recomputes everything mark-dependent whenever the teacher edits a
value — previously the "these questions have no printed marks" warning was
generated once and then stood forever, still on screen after every mark had
been entered. It also settles the paper's **maximum**, which is not the sum of
its questions when the paper offers a choice: only the best N of each group can
ever be counted, so only those N belong in the denominator. `printedMarks` is
kept alongside so a report can say "out of 15, from a paper printing 25" rather
than looking like it lost ten marks.

`validateExam` takes the paper's own OCR lines, not just the extracted
questions, because the rubric that decides the total — "answer any 3 of the
following 5" — is printed on the page. See
[choice.js](#choicejs--answer-any-3-of-the-following-5).

`classifyQuestion` returns a `QuestionType` (definition, compare, numerical, …),
and `criteriaFor` turns that into what the examiner should attend to.
Deliberately shallow: it steers attention, it is not a scoring framework.

### choice.js — "answer any 3 of the following 5"

**A paper is not always the sum of its questions.** Exam papers routinely print
more questions than a student is meant to answer, and the mark a student is owed
is out of what they were *required* to attempt. Scoring an any-3-of-5 paper out
of five questions invents two questions' worth of failure and hands it to the
student.

Nothing is configured per paper. The rubric is read off the page the same way
the questions are, so `Answer any 3 of the following 5 questions:`, `Attempt any
five questions.`, `Answer any THREE questions from Section B.` and `Do any 2 of
the following.` all work, in digits or words, without anyone describing the
paper to the app in advance.

- `parseAttemptRules(lines)` finds the rules and their counts. `of the following
  N` is read from the text *after* the count, so "any 3 of the following 5"
  cannot read its own 3.
- `detectChoice(lines)` scopes each rule to the questions that follow it,
  bounded by the next rule and by its own `of N`. That is how a compulsory
  Section A and a choice in Section B come out right with no notion of
  "sections" in the code. A rule governing fewer questions than it asks for is
  dropped rather than trusted.
- `applyChoice(questions, groups)` decides what counts.

Two decisions are made explicit rather than buried:

- **Which attempts count: the best ones.** A student who answered six when five
  were asked for is credited with their best five — the student-favourable
  reading, and the one an examiner would have to justify departing from. A
  question that could not be *evaluated* outranks a genuine zero for the last
  slot, because a pipeline failure is not evidence of a bad attempt.
- **The denominator is always N questions' worth**, even when the student
  attempted fewer. Attempting four of five loses the fifth: that is the rule the
  paper set, not a detection failure.

The choice is re-applied **live in the UI**, not just at marking time — if the
examiner raises the fourth attempt above the third, the fourth is what now
counts. `ExamPanel` shows the detected rule with the number editable, because a
misread rubric would otherwise score the whole paper out of the wrong
denominator and the examiner is the one who can see what is printed.

A side effect worth naming: the old *"the questions add up to 56 but the paper
states 40"* warning was often not a mistake at all. The paper meant 40; the sum
of its printed questions was never its total.

### match.js — which answer belongs to which question

Getting this wrong is worse than getting it uncertain: a confidently misassigned
answer marks a student down for something they answered correctly elsewhere. So
every route carries a confidence and the weak routes say so.

**Route 1 — alignment.** Blocks from `detectQuestionsStructural` are aligned
against `exam.questions` via `alignToQuestions`. Confidence and method come from
the layer that placed it.

**Route 2 — the model, in bounded batches.** This used to be *one* request
carrying the entire answer sheet plus every unmatched question, with a fixed
`maxTokens: 1800`. Groq charges prompt + completion budget against the ceiling
**up front**, so on a long paper that single call was refused outright (413) and
took every unlabelled answer with it — the user saw a half-checked paper with no
explanation. It is now:

- `unmatchedRuns()` — consecutive unmatched questions are searched in the
  stretch of the sheet *between the answers around them*; an empty gap falls
  back to everything still unclaimed rather than writing the run off;
- `chunkQuestions()` / `chunkLines()` — both listings capped by a character
  budget (≤8 questions, ~6000 chars of lines), with overlap between line windows
  so an answer straddling a cut is still visible;
- failures are per batch and name the questions they cost.

Only line ids that exist and are still unclaimed are accepted, so a hallucinated
id is dropped rather than believed.

**Route 3 — no match.** Reported as skipped. Never quietly given someone else's
text.

**How short is "too short to be an answer"** depends on what was asked. A flat
ten-character floor called `k = 3`, `42` and `Mitochondria` blank, scored them
zero and reported "no answer detected" with the student's correct answer sitting
on the page. The floor now follows the marks: 1 char at 1 mark, 4 at 2 marks, 10
above. The word-overlap sanity check likewise applies only where there are
enough words for overlap to mean anything — a two-word answer shares nothing
with its question by nature.

### reference.js — local BM25 retrieval

The evaluator should mark against the teacher's textbook, not against whatever
the model happens to remember. That needs search, but not a vector database: the
corpus is a handful of documents and the queries are exam questions full of
distinctive technical nouns — exactly the shape lexical search is strong on.

BM25 (`k1=1.5`, `b=0.75`) over the stemmer already in text.js. A few dozen lines,
no dependency, no service, runs in the browser. `chunkDocument` splits on
sentence boundaries at ~110 words with one sentence of overlap, so a definition
split across a boundary is still whole in one chunk. It satisfies the
`KnowledgeRetriever` contract in types.js, so swapping in embeddings later is a
change of file, not a change of pipeline.

`retrieve` also returns a **coverage** score — how well the corpus covers the
question at all — which is what lets a mark say "the reference material was not
enough" instead of pretending it was.

### assess.js — marking one question at a time

One model call per question, run `concurrency` at a time where `concurrency` is
the number of API keys in the pool.

**What goes into a question's prompt:**

1. **The answer key section for this question.** A long key used to be
   `slice(0, 4000)` and handed to every question alike, so the later half of a
   paper was marked with no key at all. `answerKeySection()` addresses it by
   question number, falling back to retrieval when the key carries no numbering.
2. **The paired reference answer**, if the reference material files a model
   answer under the same question — injected as real, citable evidence
   (`refq-3`) ahead of the BM25 hits, with coverage set to full. The prompt says
   to compare point by point and that **wording does not have to match**: the
   same idea in the student's own words earns the mark, and a point the model
   answer does not make cannot cost one.
3. Retrieved passages, each labelled with its source.
4. The question, its marks, its type and its criteria.
5. The student's answer, with an explicit warning that a leading `2)` is the
   *label the student wrote*, not their choice of option — observed marking a
   correct MCQ answer wrong on a real paper.

**What comes back is distrusted in five ways:** marks clamped to the ceiling
again in code; evidence ids checked against what was actually retrieved;
`REFERENCE_SUPPORTED` downgraded when nothing real is cited; per-annotation
marks dropped when they add up to more than the mark awarded; confidence
averaged only over annotations that actually reported one.

`pairReferenceAnswers(text, questions)` aligns the reference's numbered sections
to *this* paper's questions through the same alignment everything else uses.
Model answers are matched by label only (`minContent: 2` disables the content
layer) — placing a model answer by topic similarity would be a guess about which
question it answers, and marking a student against the wrong model answer is
worse than marking them without one.

`toEvaluation(paper, remark)` projects the per-question assessment onto the
older single-evaluation shape the views were built for. It must carry
`answerText` through: `answerStatus()` classifies on that field, and when the
projection dropped it **every answered question reported as unanswered**.

### geometry.js — where the ink is

Horizontal ink projection: binarise the page (Otsu), count dark pixels per row,
read the peaks as lines of writing. Decades old, and the right tool here
precisely because it is boring — it fails predictably, and we can measure how
far to trust it.

`alignBandsToLines(bands, lines)` compares the number of measured bands with the
number of OCR lines and reports the disagreement as confidence:

| Bands vs lines | Confidence | Consequence |
|---|---|---|
| equal | `high` | boxes drawn as measured |
| within 20% | `medium` | boxes drawn, labelled approximate |
| worse | `low` | per-mark boxes withheld; question region still drawn, labelled approximate |
| no bands | `none` | nothing drawn; marks live in the margin |

**Skew is corrected before projecting.** A page held crooked in front of a phone
smears every line across several rows: the peaks flatten, bands merge, and a 4°
tilt collapsed eight lines of writing into **one band** — reported `low`, which
withheld every mark on the page. The projection now searches ±3.4° of shear and
keeps the angle whose row profile is peakiest (sum of squares; spreading the same
ink over more rows can only lower it). Ties go to the straighter reading, so a
square page is never shorn on rounding noise.

The shear separates the lines; it is not a claim about where they sit. Each band
is reported as the **bounding box of the ink that belongs to it, in the page's
own coordinates** — on a crooked page a line occupies a diagonal ribbon and the
highlight has to cover the ribbon. The measured tilt is reported in the run trace
(*"page was 1.7° crooked, corrected"*).

All coordinates are fractions of page width/height, so they survive the JPEG
re-encode and any CSS scaling.

### marking.js — putting marks back on the page

The chain the whole product turns on:

```
verbatim phrase the examiner quoted
  → findSpan()           character span in the answer text
  → linesForSpan()       the OCR lines carrying that span
  → line.bbox            boxes measured from the page image
  → unionBoxes()         one box per page the span crosses
```

Every hop is reversible and every hop fails cleanly. `findSpan` tries an exact
hit, then the normalised form, then progressively shorter word prefixes,
skipping spans already claimed by an earlier annotation. `missing` annotations
describe content the student never wrote, so there is nothing to point at — they
go straight to the margin. Quotes under three characters are refused outright:
they would happily land inside an unrelated word.

### llm.js / json.js / groq.js — the model seam

**`llm.js`** is the single JSON round-trip: ask, recover the JSON from whatever
wrapping came back, validate against a declared schema, hand the caller a
coerced value. A stage never sees a raw model string. Vision calls take a
different path because vision models on this account reject `reasoning_effort`
alongside images and have no JSON mode.

**`json.js`** — `extractJson` handles fenced, prefaced and truncated replies;
`validate` is a ~150-line schema validator that clamps ranges, strips unknown
keys, applies defaults, and reports every violation with a path. The order is
always: parse strictly → recover structurally → validate → only then use.

**`groq.js`** — one call site for the whole app. Automatic back-off on 429 using
Groq's own `retry-after` header, with `onRetry(seconds)` so the UI can say why
it is waiting instead of looking frozen. `GroqError.failedGeneration` carries
the raw attempt when JSON mode rejects its own output, so a recoverable draft
beats losing the stage.

### keypool.js + vite.config.js — the key-scheduling proxy

Plain failover is not enough. Groq charges prompt + `max_completion_tokens`
against a per-minute ceiling **before the request runs**, and returns 413 when a
single request costs more than the window has left. Retrying that same request
blindly on the next key burns four keys to earn the same error four times.

So the pool tracks what each key has left — Groq reports it on every response —
and routes each request to a key that can afford it, waiting for the soonest
refill when none can. Only a request too large for any key's *full* allowance is
rejected outright, reported as its own distinct problem since no number of keys
will fix it.

`estimateCost` counts prompt chars ÷ 4, plus a measured flat 1900 tokens per
image (a 2000px page costs about the same as a 1400px one), plus the completion
budget in full, plus envelope slack.

Rerouted statuses: 429, 413 (this key is spent), 401, 403 (this key is invalid).
Everything else is a property of the request and would fail identically
everywhere. `/api/groq/stats` exposes pool size so the client can size its
marking concurrency to the number of keys.

### pdf.js / docx.js / text.js / types.js

- **pdf.js** — loads pdf.js from a CDN on demand; `extractPdfText` pulls the text
  layer and reports `hasText: false` for scans. A scanned reference is **not**
  refused: it is rasterised and read with the vision model, one call per page,
  announced while it happens and labelled *"read as a scan"* in the file list.
  Refusing it meant the paper was then marked with no reference material at all —
  the exact failure the retrieval stage exists to prevent.
- **docx.js** — reads `.docx` by unzipping `word/document.xml` in the browser.
  Legacy `.doc` is detected and refused with an explanation.
- **text.js** — `normalize` (with an offset map back to the original),
  `tokenize` + `stem` (a compact Porter-ish stemmer), `splitSentences`,
  `findSpan`, `tokenOverlap`, `hashString`.
- **types.js** — the JSDoc contracts for every structure that crosses a module
  boundary. No runtime cost; it is where the shapes are documented.

---

## The UI

`PaperChecker.jsx` is one file: state, orchestration, every component and the
whole stylesheet. It is large on purpose — the pipeline logic lives in
`src/engine/`, and what remains is presentation plus the orchestration that
binds it.

**Left rail (inputs):** question paper → reference material → answer paper →
student details → marking scheme → *Check Paper*.

Once a question paper is read, the exam panel shows every question with its
marks editable, and — when the paper offers one — **the choice it printed**, with
the count editable: `Answer any [3] of 5 — Q1, Q2, Q3, Q4, Q5 · read from the
paper`. A misread rubric would otherwise score the whole paper out of the wrong
denominator, and the examiner is the one who can see what is printed.

**Right pane (results), five tabs:**

| Tab | What it is for |
|---|---|
| **Evaluate** | The examiner's workbench. Answer-coverage summary (*"12 questions · 3 counted, 2 not required"*), an attention list of everything that did not settle cleanly — excluding questions the paper's choice made optional, which are not problems to solve — then one card per question: status, editable mark, a *Not counted* chip where it applies, "why this mark", correct/missing/incorrect points, grounding badge, confidence, and *View answer on page*. |
| **Marked Paper** | The page images with marks drawn on the handwriting. With a question selected: its answer outlined, the rest of the page dimmed, a card carrying the question, its mark, its rationale, the answer as read, and what was checked in it. |
| **Annotated Text** | The transcript with the same marks inline — the fallback when geometry could not place them. |
| **Report Card** | The printable artifact. Score arc, grade stamp, PASS/FAIL/**PENDING**, marks breakdown — excluded questions printed greyed with *"not in the total"* and a footnote saying how many of how many count — teacher's remark (typed out, and readable aloud via `speechSynthesis`), positives and improvement areas. |
| **Raw OCR Text** | The transcript, editable **one box per page**. Fixing a misread word and re-grading skips the vision pass entirely, and because the transcript is held per page the boundaries the ink measurements hang off survive the edit — a multi-page correction used to drop the overlay completely. Each box says whether marks can be placed on that page. |

**The examiner is the authority.** Any mark can be overridden; overrides
recompute the total and the grade, are marked `✎ examiner-set` everywhere, keep
the AI's proposal visible beside them, and are written back into the history row
so the log never claims the machine scored what a human decided. Typing the AI's
own number back in stores nothing — that is not a decision.

Overrides also re-run the paper's choice. Raise an excluded answer above a
counted one and the two swap: which attempts count follows the marks on screen,
not the marks the model first proposed.

**The processing screen** is a live pipeline trace, not a spinner: a phase rail,
and a feed where every line is something that actually happened with the number
it returned. Watching it is how you find out *which stage* lost an answer.

**History** keeps the last five papers in `localStorage`. The trend arrow
compares a student against **their own** previous paper — comparing against the
row above, whoever it belonged to, made the arrow meaningless.

---

## Confidence, status, grounding and counting

**Four** independent facts about a question, which fail independently and are
therefore reported separately. Collapsing any two of them is how a detection
problem starts reading as a student's failure.

**Detection** — `answerStatus(q, {hasUnassignedWriting, lowConfidence})`:

| Status | Meaning |
|---|---|
| `DETECTED` | An answer was found and linked confidently |
| `UNCERTAIN` | Found, but below the confidence floor — verify |
| `NOT_DETECTED` | Not found, **and there is unassigned writing** that could be it |
| `UNANSWERED` | Not found, and no unassigned writing anywhere — the strong case |
| `FAILED` | Marking itself failed; no mark exists yet |

**Confidence** — the weakest link wins: `min(model confidence, detection
confidence)`, capped at 65 when the reference material was insufficient and 55
when the model judged the answer irrelevant. If the model reported no confidence
at all, the detection number stands alone; nothing is invented to fill the gap.
Below 60 the mark is flagged for manual review.

**Grounding** — `REFERENCE_SUPPORTED` / `GENERAL_KNOWLEDGE` /
`INSUFFICIENT_REFERENCE`, verified against real citations. A student is entitled
to know whether they were marked on the textbook or on the model's memory.

**Counted** — whether this question is part of the total at all. A paper that
said "answer any 3 of the following 5" excludes two of them, and an excluded
question is *marked, shown, and not a failure*: it carries a blue "Not counted"
chip rather than a red one, it is kept out of the attention list and out of the
pending-marks tally, and the report card prints it greyed with "not in the
total". It is recomputed live from the marks on screen, so raising an excluded
answer above a counted one swaps which of them counts.

---

## Token budget and rate limits

The free tier meters ~8000 tokens/minute across every model, charged up front.
Measured costs: a full page image ≈ 1870 prompt tokens; an evaluation prompt
1000–2000; an answer at medium reasoning effort ≈ 2300 completion tokens.

Consequences that shaped the design:

- OCR runs **one page per request**, sequentially, with a lean budget and a
  larger retry only for a page dense enough to exhaust it.
- Answer matching is **batched by gap and by character budget** (above).
- Marking runs one question per request, `concurrency` = number of keys.
- The answer key is **sectioned per question** rather than repeated whole.
- A **scanned** reference costs one vision call per page, on top of the answer
  paper's own. It is announced page by page while it happens, because it is the
  one place the app spends the user's budget on something they did not
  explicitly ask for.
- Deskewing, ink projection, retrieval, segmentation and the choice rubric are
  all **local** — they cost nothing and cannot fail the run.
- Every wait is surfaced: *"Groq's per-minute token limit was reached — resuming
  in 14s…"*

---

## Failure modes and how each is reported

| What fails | What the user sees |
|---|---|
| Upload exceeds the page cap | Persistent notice: file name, true page count, pages kept, what the loss costs |
| A page's OCR returns nothing | Trace line marked warn; the page contributes no lines |
| Ink measurement fails on a page | Trace line with band/line counts; per-mark boxes withheld; marks go to the margin |
| Page is crooked | Corrected by the shear search; the trace names the angle. Beyond ±3.4° it stays `low` and says so |
| Reference PDF is a scan | Read with the vision model, one call per page, announced; labelled "read as a scan" in the file list |
| Question paper has no printed marks | Blocking warning naming the questions; grading refuses until they are entered |
| Declared total ≠ sum of questions | Both numbers reported; nothing adjusted. If the paper offers a choice the total is stated out of the required questions, with the printed sum named alongside |
| The rubric is misread | The detected "answer any N" is shown in the exam panel with N editable; the total and the grade follow it |
| An answer cannot be matched | Reported per question as `NOT_DETECTED` or `UNANSWERED`, with the distinction preserved |
| A student answers more than the paper asked for | The best N count; the extras are marked, shown with a *Not counted* chip, and never scored as failures |
| A student answers fewer than the paper asked for | The empty slots keep their marks in the denominator — that is the rule the paper set, not a detection failure |
| A one-word answer to a one-mark question | Treated as an answer: the "too short" floor follows the marks, not a fixed character count |
| Writing matches no question at all | *"Possible missed answer: N characters could not be assigned"* + a coverage warning |
| A matching batch errors | Warning naming those questions: *"reported as undetected rather than guessed"* |
| Marking one question throws | `failed: true`; mark box empty, report prints `—`, totals state the pending marks, PASS/FAIL becomes PENDING |
| The model returns bad JSON | `extractJson` recovers what it can; otherwise that question fails alone |
| Every key is spent | Proxy waits for the soonest refill, then surfaces 429 to the browser's own back-off |
| A request is too large for any key | Rejected immediately as `request_too_large` — no number of keys fixes it |

---

## Tests

Plain node scripts, no framework, no network. `npm test` runs all seven.

| Suite | Covers |
|---|---|
| `identity.test.mjs` | Label reading across ~20 notations; prose rejection; the paper's own numbering settling ambiguous shapes; one key space; all four alignment layers; ambiguity left unplaced; one block one question |
| `geometry.test.mjs` | Otsu thresholding, band detection and band↔line confidence across 18 synthetic pages — spacing, faint pencil, grey paper, noise, borders, and three tilts. It reports the table **and asserts on it**: every case but the touching-lines one must find its lines exactly, all three skew cases must separate, stay usable, cover the writing and report their tilt, and a square page must not be shorn |
| `marking.test.mjs` | Span anchoring, segment slicing, unanchored annotations, box resolution |
| `pipeline.test.mjs` | Exam validation, question typing, BM25 retrieval, matching, mark control, grounding, batched matching, answer-key sectioning, reference pairing, per-annotation marks, confidence, short answers, the view projection |
| `choice.test.mjs` | Rubric wordings in digits and words; rule scoping across sections; bounded groups; best-N selection; under-attempted papers; unevaluated attempts winning the last slot; a whole paper scored end to end out of 15 rather than 25 |
| `keypool.test.mjs` | Cost estimation, reset parsing, key selection, budget tracking |
| `proxy.test.mjs` | Failover on 429/413, dead-key retirement, oversized rejection, and that a key is never echoed to the client |

The suites are written to pin **behaviour that was once wrong**, so each
assertion names the bug it prevents rather than the function it calls.

---

## Known limits

Deliberate, and not going to change:

- **A parent question does not pair with its sub-parts.** If the reference
  answers `3(a)` and `3(b)` but the paper asks a whole `3`, no pairing is made —
  marking a compare-and-contrast answer against half of one is worse than
  searching by topic.
- **Model answers are never placed by content similarity.** Only labels pair
  them. Guessing which question a model answer belongs to and then marking a
  student against it is a compounding error.
- **A block two questions fit equally well stays unplaced.** "We could not find
  this answer" is true; picking one is not.
- **The best attempts fill a choice, not the first ones.** Some boards mark the
  first N attempted instead. `applyChoice` takes `{prefer: "first"}` for that,
  but the default is the student-favourable reading, because an examiner should
  have to choose to be stricter rather than choose to be fair.

Real, and bounded:

- **Skew beyond ±3.4°** is outside the shear search. The page still measures,
  but the confidence report is the honest answer rather than a correction.
- **Touching lines still merge.** Handwriting whose ascenders and descenders
  overlap the neighbouring line has no gap in the ink profile to find. It is
  reported as `low` and the marks go to the margin.
- **Multi-column layouts read as single lines** spanning both columns.
- **A scanned reference costs one vision call per page** and is only as good as
  that transcription. The file list labels it *"read as a scan"* so a weak
  retrieval result can be traced to its cause.
- **An unrecognised rubric wording** falls back to no choice, and the paper is
  scored out of everything printed. The detected rule is shown in the exam panel
  with its count editable, so the examiner can correct it in one field — but
  they have to notice.
- **Choice is detected from printed rubric only.** A paper whose optionality is
  conveyed by layout alone (an "OR" between two questions with no instruction
  line) is not detected. Internal `OR` choices within a single question are not
  modelled at all.
- **History is per-browser** (`localStorage`, last five papers). There is no
  server-side store, and no cross-device view of a student.

---

## Extending the system

**Add a new question type** → `TYPE_RULES` (the instruction verb that selects it)
and `CRITERIA` (what it is marked on) in `exam.js`, plus the `QuestionType` union
in `types.js`.

**Swap BM25 for embeddings** → implement the `KnowledgeRetriever` contract in
`types.js`. Nothing else changes; `assessPaper` only calls `.search()`.

**Support a new label notation** → `identity.js` only. Add the shape to
`readLabel`, add a case to `identity.test.mjs`. No other module knows or cares
what a label looks like.

**Support a new rubric wording** → `choice.js` only. Extend `ANY_RULE`, add a
row to the wordings table in `choice.test.mjs`. Scoping, selection and the
denominator are already generic.

**Change models** → `OCR_MODEL` / `EVAL_MODEL` in `groq.js`. Check
`estimateCost`'s per-image constant if the vision model changes.

**Add a pipeline stage** → emit `step(text, detail, kind)` from it so it appears
in the live trace. A stage the user cannot watch is a stage they cannot debug.

**The rule to keep**: if you find yourself writing a default for something a
model was supposed to report, stop. That default will be displayed as a
measurement, and someone will make a decision about a student on it.
