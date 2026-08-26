/**
 * Optional questions: "Answer any 3 of the following 5".
 *
 * A paper is not always the sum of its questions. Exam papers routinely print
 * more questions than a student is meant to answer, and the mark a student is
 * owed is out of what they were *required* to attempt — not out of everything
 * printed. Scoring a paper out of all seven questions when the rubric said
 * "attempt any five" invents two questions' worth of failure and hands it to
 * the student.
 *
 * Nothing here is configured per paper. The rubric is read off the paper the
 * same way the questions are, so any wording, any counts, any number of
 * sections work without anyone telling the app about them in advance. Where the
 * wording is one this parser does not know, the examiner sets the number by hand
 * and the rest of the machinery is identical.
 *
 * Two decisions are made explicit rather than buried:
 *
 *   · **Which attempts count.** The best ones. A student who answered six when
 *     five were asked for is credited with their best five, which is the
 *     student-favourable reading and the one an examiner would have to defend
 *     departing from. It is overridable per paper.
 *   · **What the denominator is.** Always N questions' worth, even when the
 *     student attempted fewer. Attempting four of five loses the fifth — that
 *     is the rule the paper set, not a detection failure.
 */

import { NUMBER_WORDS } from "./identity.js";
import { detectQuestionsStructural } from "./segment.js";
import { canonicalLabel } from "./identity.js";

/** "5", "five", "FIVE" → 5. Anything else → null. */
export function countFromWord(raw) {
  const s = String(raw || "").trim().toLowerCase();
  if (/^\d{1,2}$/.test(s)) {
    const n = Number(s);
    return n > 0 ? n : null;
  }
  return NUMBER_WORDS[s] !== undefined ? NUMBER_WORDS[s] : null;
}

const COUNT = "\\d{1,2}|" + Object.keys(NUMBER_WORDS).join("|");

/* "Answer any THREE of the following five questions", "Attempt any 5",
   "Do any two questions from Section B". The verb, then "any", then the count;
   an optional "of … <count>" gives the size of the group it is choosing from. */
const ANY_RULE = new RegExp(
  `\\b(?:answer|attempt|do|solve|write|complete)\\b[^.\\n]{0,30}?\\bany\\b\\s*(?:${COUNT})\\b` +
    `(?:[^.\\n]{0,40}?\\bof\\b[^.\\n]{0,25}?\\b(?:${COUNT})\\b)?`,
  "i"
);
const ANY_COUNT = new RegExp(`\\bany\\b\\s*(${COUNT})\\b`, "i");
const OF_COUNT = new RegExp(`\\bof\\b(?:[^.\\n]{0,25}?)\\b(${COUNT})\\b`, "i");

/* A paper that says everything is compulsory is saying there is no choice, and
   saying it explicitly is worth recording: it stops a stray "any" elsewhere on
   the page from being read as a rule. */
const ALL_COMPULSORY = /\b(?:all|every)\b[^.\n]{0,30}\b(?:compulsory|mandatory|required)\b|\banswer\s+all\b/i;

/**
 * Read the attempt rules printed on a paper.
 *
 * @param {import("./types.js").OCRLine[]} lines  the question paper's own lines
 * @returns {{index: number, required: number, of: number|null, text: string}[]}
 */
export function parseAttemptRules(lines) {
  const rules = [];
  for (const line of lines || []) {
    const text = String(line.text || "");
    if (!ANY_RULE.test(text)) continue;

    const any = ANY_COUNT.exec(text);
    const required = any ? countFromWord(any[1]) : null;
    if (!required) continue;

    /* "of the following five" is the group's size. It is read from the text
       after the count, so "any 3 of the following 5" cannot read its own 3. */
    const tail = text.slice(any.index + any[0].length);
    const of = OF_COUNT.test(tail) ? countFromWord(OF_COUNT.exec(tail)[1]) : null;

    rules.push({ index: line.index, required, of, text: text.trim() });
  }
  return rules;
}

/** Whether the paper states outright that there is no choice. */
export function saysAllCompulsory(lines) {
  return (lines || []).some((l) => ALL_COMPULSORY.test(String(l.text || "")));
}

/**
 * Turn the printed rubric into groups of question numbers.
 *
 * A rule governs the questions that follow it, bounded by the next rule and by
 * its own "of N" where one was printed. That is how a paper with a compulsory
 * Section A and a choice in Section B comes out right without anyone describing
 * its sections to the app.
 *
 * @param {import("./types.js").OCRLine[]} lines  the question paper
 * @returns {{required: number, numbers: string[], text: string}[]}
 */
export function detectChoice(lines) {
  const rules = parseAttemptRules(lines);
  if (rules.length === 0) return [];

  const { questions } = detectQuestionsStructural(lines);
  const heads = questions
    .map((q) => ({ number: canonicalLabel(q.questionNumber), index: q.headIndex }))
    .filter((h) => h.number && Number.isFinite(h.index))
    .sort((a, b) => a.index - b.index);

  const groups = [];
  rules.forEach((rule, i) => {
    const next = rules[i + 1] ? rules[i + 1].index : Infinity;
    let following = heads.filter((h) => h.index > rule.index && h.index < next).map((h) => h.number);

    /* "of the following 5" bounds the group even when more questions follow;
       without it the group runs to the next rule or the end of the paper. */
    if (rule.of && rule.of > 0) following = following.slice(0, rule.of);

    /* A rule that governs fewer questions than it asks for is describing
       something this parser did not scope correctly — a footer repeated on
       every page, a rule printed after its own section. Dropping it is safer
       than scoring a paper out of a group that cannot satisfy it. */
    if (following.length < rule.required || rule.required <= 0) return;
    if (following.length === rule.required) return; // no choice at all

    groups.push({ required: rule.required, numbers: following, text: rule.text });
  });

  return groups;
}

/**
 * Decide which questions count towards the paper's total.
 *
 * @param {{questionId: string, number: string, maxMarks: number, marksAwarded?: number,
 *          skipped?: boolean, failed?: boolean}[]} questions
 * @param {{required: number, numbers: string[]}[]} groups
 * @param {{prefer?: "best"|"first"}} [opts]
 * @returns {{counted: Set<string>, dropped: {questionId: string, number: string, reason: string}[],
 *            maximumMarks: number, totalMarks: number}}
 */
export function applyChoice(questions, groups, opts = {}) {
  const prefer = opts.prefer === "first" ? "first" : "best";
  const list = Array.isArray(questions) ? questions : [];
  const counted = new Set(list.map((q) => q.questionId));
  const dropped = [];

  const byNumber = new Map(list.map((q) => [canonicalLabel(q.number), q]));

  for (const group of groups || []) {
    const members = group.numbers.map((n) => byNumber.get(canonicalLabel(n))).filter(Boolean);
    if (members.length <= group.required) continue;

    /* Attempted first — an unattempted question can only ever fill a slot no
       attempt is left to fill. Among attempts, the best ones count. */
    const attempted = members.filter((q) => !q.skipped);
    const unattempted = members.filter((q) => q.skipped);

    const ranked =
      prefer === "first"
        ? attempted
        : attempted.slice().sort((a, b) => {
            const aw = (Number(b.marksAwarded) || 0) - (Number(a.marksAwarded) || 0);
            if (aw !== 0) return aw;
            /* A question that could not be evaluated is not evidence of a bad
               attempt, so it outranks a genuine zero for the last slot. */
            if (!!a.failed !== !!b.failed) return a.failed ? -1 : 1;
            return (Number(b.maxMarks) || 0) - (Number(a.maxMarks) || 0);
          });

    const keep = ranked.slice(0, group.required);
    /* Short of attempts, the empty slots are still slots: they keep their marks
       in the denominator so an under-attempted paper scores what it earned. */
    if (keep.length < group.required) {
      keep.push(...unattempted.slice(0, group.required - keep.length));
    }

    const kept = new Set(keep.map((q) => q.questionId));
    for (const q of members) {
      if (kept.has(q.questionId)) continue;
      counted.delete(q.questionId);
      dropped.push({
        questionId: q.questionId,
        number: q.number,
        reason: q.skipped ? "not attempted, and not required" : "an extra attempt beyond the choice",
      });
    }
  }

  let maximumMarks = 0;
  let totalMarks = 0;
  for (const q of list) {
    if (!counted.has(q.questionId)) continue;
    maximumMarks += Number(q.maxMarks) || 0;
    totalMarks += Number(q.marksAwarded) || 0;
  }

  return { counted, dropped, maximumMarks, totalMarks };
}

/**
 * One line an examiner can read, describing what the choice did.
 * @returns {string} "" when the paper had no choice to apply
 */
export function describeChoice(groups) {
  if (!groups || groups.length === 0) return "";
  return groups
    .map((g) => `answer any ${g.required} of ${g.numbers.length} (Q${g.numbers.join(", Q")})`)
    .join("; ");
}
