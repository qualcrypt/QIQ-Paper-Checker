/**
 * Reference material, indexed locally.
 *
 * The evaluator should mark against the teacher's own textbook and notes rather
 * than against whatever the model happens to remember. That means the reference
 * PDFs have to be searchable, but it does not mean a vector database: the corpus
 * is a handful of documents, the queries are exam questions full of distinctive
 * technical nouns, and lexical search is very strong on exactly that shape. This
 * is BM25 over the stemmer already in text.js — a few dozen lines, no
 * dependency, no service, and it runs in the browser.
 *
 * It satisfies the KnowledgeRetriever contract in types.js, so swapping in
 * embeddings later is a change of file, not a change of pipeline.
 */

import { tokenize, normalizeText, splitSentences, hashString } from "./text.js";

/* Long enough to carry a complete idea, short enough that three of them fit in
   a prompt without crowding out the student's answer. */
const TARGET_WORDS = 110;
const OVERLAP_SENTENCES = 1;

const K1 = 1.5; // term-frequency saturation
const B = 0.75; // length normalisation

/**
 * Split a document into retrievable chunks on sentence boundaries, with a
 * one-sentence overlap so a definition split across a boundary is still whole in
 * one of them.
 *
 * @param {string} text
 * @param {{source?: string, page?: number}} meta
 * @returns {import("./types.js").KnowledgeDoc[]}
 */
export function chunkDocument(text, meta = {}) {
  const sentences = splitSentences(String(text || ""));
  const chunks = [];
  let buffer = [];
  let words = 0;

  const flush = () => {
    if (buffer.length === 0) return;
    const body = buffer.map((s) => s.text).join(" ").trim();
    if (normalizeText(body).length > 0) {
      chunks.push({
        id: `${meta.source || "ref"}#${chunks.length + 1}-${hashString(body)}`,
        text: body,
        metadata: { sourceType: "reference", ...meta },
      });
    }
    buffer = buffer.slice(Math.max(0, buffer.length - OVERLAP_SENTENCES));
    words = buffer.reduce((n, s) => n + s.text.split(/\s+/).length, 0);
  };

  for (const s of sentences) {
    buffer.push(s);
    words += s.text.split(/\s+/).length;
    if (words >= TARGET_WORDS) flush();
  }
  // The tail carries no overlap duplicate, so only keep it if it added something.
  if (buffer.length > OVERLAP_SENTENCES || chunks.length === 0) {
    buffer = buffer.slice(0);
    const body = buffer.map((s) => s.text).join(" ").trim();
    if (normalizeText(body).length > 0) {
      chunks.push({
        id: `${meta.source || "ref"}#${chunks.length + 1}-${hashString(body)}`,
        text: body,
        metadata: { sourceType: "reference", ...meta },
      });
    }
  }

  return chunks;
}

/**
 * Build a BM25 index over chunks.
 * @param {import("./types.js").KnowledgeDoc[]} docs
 */
export function buildIndex(docs) {
  const postings = new Map(); // term -> Map(docIndex -> frequency)
  const lengths = [];

  docs.forEach((doc, i) => {
    const terms = tokenize(doc.text);
    lengths[i] = terms.length;
    const seen = new Map();
    for (const t of terms) seen.set(t, (seen.get(t) || 0) + 1);
    for (const [t, f] of seen) {
      if (!postings.has(t)) postings.set(t, new Map());
      postings.get(t).set(i, f);
    }
  });

  const avgLength = lengths.length ? lengths.reduce((a, b) => a + b, 0) / lengths.length : 0;
  return { docs, postings, lengths, avgLength, size: docs.length };
}

/**
 * Retrieve the chunks most likely to answer a question.
 *
 * `coverage` is reported alongside the score because it is the honest signal for
 * "does the reference material actually cover this?" — the fraction of the
 * question's own content words that appear anywhere in what came back. A high
 * BM25 score only says a chunk was the best of a bad set; low coverage is what
 * tells the evaluator to fall back to general knowledge and label it as such.
 *
 * @param {object} index from buildIndex
 * @param {import("./types.js").RetrievalContext} ctx
 * @returns {{evidence: import("./types.js").RetrievedEvidence[], coverage: number}}
 */
export function retrieve(index, { query, topK = 3 } = {}) {
  if (!index || index.size === 0) return { evidence: [], coverage: 0 };

  const terms = tokenize(query);
  if (terms.length === 0) return { evidence: [], coverage: 0 };

  const N = index.size;
  const scores = new Map();

  for (const term of new Set(terms)) {
    const posting = index.postings.get(term);
    if (!posting) continue;

    const df = posting.size;
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));

    for (const [i, f] of posting) {
      const norm = 1 - B + (B * index.lengths[i]) / (index.avgLength || 1);
      const tf = (f * (K1 + 1)) / (f + K1 * norm);
      scores.set(i, (scores.get(i) || 0) + idf * tf);
    }
  }

  const ranked = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([i, score]) => ({
      id: index.docs[i].id,
      text: index.docs[i].text,
      score,
      lexicalScore: score,
      metadata: index.docs[i].metadata,
    }));

  // Coverage over the retrieved set, not the whole corpus.
  const present = new Set();
  for (const r of ranked) for (const t of tokenize(r.text)) present.add(t);
  const wanted = new Set(terms);
  let hit = 0;
  for (const t of wanted) if (present.has(t)) hit++;
  const coverage = wanted.size ? hit / wanted.size : 0;

  return { evidence: ranked, coverage };
}

/** Below this, the reference material does not really address the question. */
export const WEAK_COVERAGE = 0.45;

/**
 * A KnowledgeRetriever (types.js) over a set of reference documents.
 * @param {import("./types.js").KnowledgeDoc[]} docs
 */
export function createRetriever(docs) {
  const index = buildIndex(docs);
  return {
    size: index.size,
    retrieve: async (ctx) => retrieve(index, ctx).evidence,
    /** Retrieval plus the coverage signal the evaluator needs. */
    search: (ctx) => retrieve(index, ctx),
  };
}
