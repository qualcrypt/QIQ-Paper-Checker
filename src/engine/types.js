/**
 * Shared contracts for the QIQ assessment engine.
 *
 * The repository is plain JavaScript, so the interfaces from the design spec are
 * expressed as JSDoc typedefs rather than TypeScript. Editors and `tsc --checkJs`
 * understand these, and no new build step is introduced.
 *
 * Ownership boundaries this engine enforces:
 *   the APPLICATION owns structure, validation, scoring, totals and thresholds;
 *   the KNOWLEDGE layer owns evidence, marking schemes and concepts;
 *   the LLM owns interpretation, claim extraction, comparison and prose.
 * No model-generated number is ever trusted as a final mark.
 */

/* ------------------------------------------------------------------ OCR -- */

/**
 * @typedef {Object} BBox
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 */

/**
 * One physical line of the paper. `id` is stable and is what annotations point
 * at, so the UI can always trace a mark back to the exact text that caused it.
 * @typedef {Object} OCRLine
 * @property {string} id
 * @property {number} page
 * @property {number} index        Reading order within the whole document.
 * @property {string} text
 * @property {number} [start]      Char offset of this line inside OCRDocument.text.
 * @property {number} [end]
 * @property {BBox} [bbox]         Fractions of the page, measured by geometry.js.
 * @property {number} [confidence] 0..1
 */

/**
 * @typedef {Object} OCRDocument
 * @property {OCRLine[]} lines
 * @property {number} pageCount
 * @property {string} text          Reconstructed plain text (backwards compatible).
 * @property {OCRQuality} quality
 */

/**
 * @typedef {Object} OCRQuality
 * @property {number} score          0..1 overall legibility estimate.
 * @property {number} illegibleRatio Share of lines carrying illegibility markers.
 * @property {number} shortLineRatio
 * @property {string[]} flags
 */

/* ------------------------------------------------------------ questions -- */

/** @typedef {"definition"|"explain"|"describe"|"compare"|"differentiate"|"discuss"|"analyze"|"evaluate"|"justify"|"cause_effect"|"short_answer"|"long_answer"|"numerical"|"diagram"|"essay"} QuestionType */

/**
 * @typedef {Object} ParsedQuestion
 * @property {string} id
 * @property {string} questionNumber
 * @property {string} questionText
 * @property {string} answerText
 * @property {string[]} answerLineIds
 * @property {number} [maxMarks]
 * @property {number} [pageStart]
 * @property {number} [pageEnd]
 * @property {string} [parentId]     Set for sub-questions such as (a) or (i).
 */

/**
 * @typedef {Object} QuestionUnderstanding
 * @property {string} [subject]
 * @property {string} [topic]
 * @property {string} [chapter]
 * @property {QuestionType} questionType
 * @property {"easy"|"medium"|"hard"} difficulty
 * @property {number} maxMarks
 * @property {"recall"|"understanding"|"application"|"analysis"} expectedDepth
 * @property {string[]} requiredConcepts
 * @property {string[]} dimensions   Which rubric dimensions this type needs.
 */

/* ------------------------------------------------------------ knowledge -- */

/**
 * @typedef {Object} EvidenceMetadata
 * @property {string} [subject]
 * @property {string} [klass]        "class" is reserved; stored as klass.
 * @property {string} [board]
 * @property {string} [chapter]
 * @property {string} [topic]
 * @property {string} [questionType]
 * @property {"marking_scheme"|"textbook"|"syllabus"|"model_answer"|"rubric"|"question_bank"|"reference"} [sourceType]
 * @property {number} [year]
 */

/**
 * @typedef {Object} KnowledgeDoc
 * @property {string} id
 * @property {string} text
 * @property {EvidenceMetadata} metadata
 */

/**
 * @typedef {Object} RetrievedEvidence
 * @property {string} id
 * @property {string} text
 * @property {number} score          Final reranked score.
 * @property {number} [lexicalScore]
 * @property {number} [vectorScore]
 * @property {EvidenceMetadata} metadata
 */

/**
 * @typedef {Object} RetrievalContext
 * @property {string} query
 * @property {string[]} [keywords]
 * @property {EvidenceMetadata} [filter]
 * @property {number} [topK]
 */

/**
 * The retrieval seam. Any implementation — in-memory, SQLite FTS, pgvector,
 * Pinecone — satisfies this without the pipeline changing.
 * @typedef {Object} KnowledgeRetriever
 * @property {(ctx: RetrievalContext) => Promise<RetrievedEvidence[]>} retrieve
 */

/* --------------------------------------------------------------- rubric -- */

/**
 * @typedef {Object} RubricCriterion
 * @property {string} id
 * @property {string} description
 * @property {number} marks             Maximum for this criterion.
 * @property {string[]} [concepts]      Concepts that satisfy it.
 * @property {string} [dimension]
 */

/**
 * @typedef {Object} Rubric
 * @property {string} questionId
 * @property {number} maxMarks
 * @property {RubricCriterion[]} criteria
 * @property {string} source            "llm" | "scheme" | "fallback"
 */

/* --------------------------------------------------------------- claims -- */

/** @typedef {"SUPPORTED"|"PARTIALLY_SUPPORTED"|"CONTRADICTED"|"UNSUPPORTED"|"IRRELEVANT"} ClaimStatus */

/**
 * @typedef {Object} Claim
 * @property {string} id
 * @property {string} text          Verbatim student text.
 * @property {string} concept
 * @property {string[]} lineIds
 * @property {number} start         Character offset into the answer text.
 * @property {number} end
 */

/**
 * @typedef {Object} ClaimAssessment
 * @property {string} claimId
 * @property {string} text
 * @property {string} concept
 * @property {string[]} lineIds
 * @property {ClaimStatus} status
 * @property {string[]} evidenceIds
 * @property {string} [rubricCriterion]
 * @property {string} reason
 * @property {boolean} isFactualError  CONTRADICTED means wrong, not merely absent.
 * @property {number} confidence      0..100
 */

/* ----------------------------------------------------------- evaluation -- */

/**
 * @typedef {Object} CriterionResult
 * @property {string} criterionId
 * @property {string} description
 * @property {number} maxMarks
 * @property {number} marksAwarded      Model proposal, before deterministic clamping.
 * @property {string} justification
 * @property {string[]} claimIds
 * @property {string[]} evidenceIds
 * @property {string[]} missingConcepts
 * @property {boolean} [criticAdjusted]
 * @property {number} [criticDelta]
 */

/**
 * @typedef {Object} ConfidenceResult
 * @property {number} score        0..100
 * @property {string[]} reasons
 * @property {string[]} flags
 * @property {Object.<string, number>} [signals]
 */

/**
 * @typedef {Object} Annotation
 * @property {"correct"|"partial"|"wrong"|"missing"} type
 * @property {string} text
 * @property {string} comment
 * @property {number} marks
 * @property {number} confidence
 * @property {string} [claimId]
 * @property {string} [lineId]
 * @property {string} [rubricCriterion]
 * @property {string} [reason]
 * @property {string} [questionId]
 */

/**
 * @typedef {Object} QuestionAssessment
 * @property {string} questionId
 * @property {string} questionNumber
 * @property {string} questionText
 * @property {string} answerText
 * @property {number} maxMarks
 * @property {number} finalMarks
 * @property {string} questionType
 * @property {QuestionUnderstanding} understanding
 * @property {ClaimAssessment[]} claims
 * @property {RubricCriterion[]} rubric
 * @property {CriterionResult[]} criteria
 * @property {RetrievedEvidence[]} evidence
 * @property {Annotation[]} annotations
 * @property {ConfidenceResult} confidence
 * @property {string[]} factualErrors
 * @property {string[]} missingConcepts
 */

/**
 * @typedef {Object} PaperAssessment
 * @property {string} paperId
 * @property {{name?: string}} [student]
 * @property {string} [subject]
 * @property {QuestionAssessment[]} questions
 * @property {number} totalMarks
 * @property {number} maximumMarks
 * @property {number} percentage
 * @property {string} grade
 * @property {boolean} passed
 * @property {number} overallConfidence
 * @property {{strengths: string[], weaknesses: string[], recommendations: string[], studyTopics: string[], remark: string}} summary
 * @property {string[]} warnings       Non-fatal degradations worth surfacing.
 */

export {};
