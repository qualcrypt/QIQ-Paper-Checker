/**
 * The JSON seam between the pipeline and the model.
 *
 * Every reasoning stage in this engine asks for JSON and refuses to proceed on
 * anything else. `callJson` is the single place that enforces it: ask, recover
 * the JSON from whatever wrapping came back, validate it against a declared
 * schema, and hand the caller a coerced value it can rely on. A stage never
 * sees a raw model string.
 *
 * segment.js has expected exactly this interface since it was written
 * (`llm.callJson({stage, system, user, schema, maxTokens})`) and has been
 * silently falling back to its regex-only path for want of an implementation.
 *
 * Nothing here trusts a number from the model. Schemas clamp ranges, the
 * validator strips unknown keys, and marks are recomputed in assess.js.
 */

import { extractJson, SchemaError, validate } from "./json.js";
import { groqChat, EVAL_MODEL } from "./groq.js";

/**
 * @param {object} opts
 * @param {(seconds: number) => void} [opts.onRetry]  Surface rate-limit waits.
 * @param {(stage: string) => void} [opts.onStage]    Progress for the UI.
 * @returns {{callJson: Function, callText: Function}}
 */
export function createLlm({ onRetry, onStage } = {}) {
  /**
   * One JSON round-trip.
   *
   * @param {object} req
   * @param {string} req.stage      Label for progress and error messages.
   * @param {string} req.system
   * @param {string} req.user
   * @param {object} req.schema     A json.js schema; the reply is validated against it.
   * @param {number} req.maxTokens
   * @param {string} [req.model]
   * @param {string} [req.effort]   Groq reasoning_effort.
   * @param {object[]} [req.images] data URLs to attach as image_url parts.
   */
  async function callJson({
    stage,
    system,
    user,
    schema,
    maxTokens,
    model = EVAL_MODEL,
    effort = "medium",
    images = [],
    temperature = 0.2,
  }) {
    if (onStage) onStage(stage);

    const content = images.length
      ? [{ type: "text", text: user }].concat(
          images.map((url) => ({ type: "image_url", image_url: { url } }))
        )
      : user;

    const payload = {
      model,
      max_completion_tokens: maxTokens,
      temperature,
      messages: [
        { role: "system", content: system },
        { role: "user", content },
      ],
    };

    /* Vision models on this account reject reasoning_effort together with
       images, and JSON mode is unavailable there too — hence the split. */
    if (images.length === 0) {
      payload.response_format = { type: "json_object" };
      payload.reasoning_effort = effort;
    } else {
      payload.reasoning_effort = "none";
      payload.reasoning_format = "hidden";
    }

    let out;
    try {
      out = await groqChat(payload, onRetry);
    } catch (e) {
      /* JSON mode can reject its own generation and still return the attempt.
         A recoverable draft beats losing the whole stage. */
      if (e && e.failedGeneration) out = { text: e.failedGeneration, reasoning: "" };
      else throw e;
    }

    let parsed;
    try {
      parsed = extractJson(out.text);
    } catch (e) {
      throw new SchemaError(`The ${stage} step did not return usable JSON. ${e.message}`);
    }

    if (!schema) return parsed;

    try {
      return validate(parsed, schema);
    } catch (e) {
      throw new SchemaError(`The ${stage} step returned the wrong shape. ${e.message}`, e.details);
    }
  }

  /** Plain text, for OCR where there is no structure to validate. */
  async function callText({ stage, system, user, maxTokens, model, images = [], temperature = 0 }) {
    if (onStage) onStage(stage);

    const content = images.length
      ? [{ type: "text", text: user }].concat(
          images.map((url) => ({ type: "image_url", image_url: { url } }))
        )
      : user;

    const { text } = await groqChat(
      {
        model,
        max_completion_tokens: maxTokens,
        temperature,
        reasoning_format: "hidden",
        reasoning_effort: "none",
        messages: [
          { role: "system", content: system },
          { role: "user", content },
        ],
      },
      onRetry
    );
    return text;
  }

  return { callJson, callText };
}
