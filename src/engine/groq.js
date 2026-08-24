/**
 * The Groq HTTP client.
 *
 * One call site for the whole application. The key is never here: requests go
 * to the same-origin /api/groq route, which attaches the Authorization header
 * server-side (see the groqProxy plugin in vite.config.js).
 *
 * The free tier meters roughly 8000 tokens per minute across every model, and
 * Groq charges prompt + max_completion_tokens against that ceiling *up front* —
 * an oversized budget is rejected outright even when the model would never have
 * spent it. That is why every caller passes a measured maxTokens, and why the
 * 429 back-off below matters: a multi-question paper will hit the limit and has
 * to wait it out rather than fail.
 */

export const GROQ_BASE = (typeof window !== "undefined" && window.__GROQ_BASE__) || "/api/groq";

export const OCR_MODEL = "qwen/qwen3.6-27b";
export const EVAL_MODEL = "openai/gpt-oss-120b";

export const RATE_LIMIT_RETRIES = 3;

export class GroqError extends Error {
  constructor(message, failedGeneration) {
    super(message);
    this.failedGeneration = failedGeneration || "";
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Groq reports the exact wait in a header, or failing that inside the message. */
function retryDelaySeconds(res, err) {
  const header = Number(res.headers.get("retry-after"));
  if (Number.isFinite(header) && header > 0) return Math.min(header, 60);
  const m = /try again in ([\d.]+)s/i.exec(err.message || "");
  if (m) return Math.min(Math.ceil(Number(m[1])) + 1, 60);
  return 20;
}

/**
 * One chat completion, with automatic back-off on the free tier's per-minute
 * token limit. `onRetry(seconds)` lets the UI say why it is waiting instead of
 * looking frozen.
 */
export async function groqChat(payload, onRetry) {
  for (let attempt = 0; ; attempt++) {
    let res;
    try {
      res = await fetch(`${GROQ_BASE}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new GroqError("Could not reach the Groq API. Check your internet connection.");
    }

    const body = await res.json().catch(() => null);

    if (res.ok) {
      const choice = body && body.choices && body.choices[0];
      const text = ((choice && choice.message && choice.message.content) || "").trim();

      if (choice && choice.finish_reason === "length") {
        throw new GroqError(
          "The model hit its output limit before finishing. Try a shorter paper or a tighter marking scheme."
        );
      }
      if (!text) throw new GroqError("The model returned an empty response. Please try again.");

      return { text, reasoning: (choice && choice.message && choice.message.reasoning) || "" };
    }

    const err = (body && body.error) || {};

    if (res.status === 429 && attempt < RATE_LIMIT_RETRIES) {
      const wait = retryDelaySeconds(res, err);
      if (onRetry) onRetry(wait);
      await sleep(wait * 1000);
      continue;
    }

    if (res.status === 401)
      throw new GroqError("Groq rejected the API key (401). Check GROQ_API_KEY in .env, then restart the dev server.");
    if (res.status === 429)
      throw new GroqError("Groq's per-minute token limit is still exhausted. Wait a minute and try again.");
    if (res.status === 413)
      throw new GroqError(
        "The request exceeded this Groq tier's per-minute token allowance. Upload fewer pages at once, or upgrade the account."
      );
    if (err.code === "json_validate_failed" && !err.failed_generation)
      throw new GroqError(
        "The examiner used its whole token budget before writing the verdict. Shorten the marking scheme, or raise EVAL_MAX_TOKENS if the account allows it."
      );

    throw new GroqError(`Groq error ${res.status}: ${err.message || "unknown error"}`, err.failed_generation);
  }
}
