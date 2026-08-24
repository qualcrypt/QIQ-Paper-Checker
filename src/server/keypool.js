/**
 * A pool of Groq keys, scheduled against their actual remaining budget.
 *
 * Plain failover is not enough. Groq charges prompt + max_completion_tokens
 * against a per-minute ceiling *up front*, and returns 413 when a single request
 * costs more than the window has left. Retrying that same request on the next
 * key blindly is how you burn four keys to earn the same error four times.
 *
 * So the pool tracks what each key has left — Groq reports it on every response
 * — and routes each request to a key that can actually afford it. When no key
 * can, it waits for the soonest window to refill rather than failing, because
 * the caller's alternative is to fail anyway. Only a request too large for any
 * key's *full* allowance is rejected outright, and that is reported as its own
 * distinct problem, since no number of keys will ever fix it.
 *
 * Runs server-side only. Keys never leave this process.
 */

/** Groq reports resets as "7.66s", "1m2.5s", "500ms". */
export function parseReset(value) {
  const s = String(value || "").trim();
  if (!s) return null;

  let ms = 0;
  let matched = false;
  for (const [re, mult] of [
    [/([\d.]+)\s*m(?!s)/, 60000],
    [/([\d.]+)\s*s/, 1000],
    [/([\d.]+)\s*ms/, 1],
  ]) {
    const m = re.exec(s);
    if (m) {
      ms += parseFloat(m[1]) * mult;
      matched = true;
    }
  }
  if (!matched) {
    const n = parseFloat(s);
    if (!Number.isFinite(n)) return null;
    ms = n * 1000; // bare numbers are seconds
  }
  return Math.max(0, Math.round(ms));
}

/**
 * What a request will cost against the token ceiling.
 *
 * Groq bills the completion budget whether or not it is spent, so the ceiling
 * sees max_completion_tokens in full. The prompt is estimated at 4 characters
 * per token, and images are charged at a measured flat rate rather than by their
 * base64 length — a 2000px page costs about the same as a 1400px one.
 */
export function estimateCost(body) {
  let payload;
  try {
    payload = JSON.parse(body.toString("utf8"));
  } catch {
    return 2000; // unparseable: assume a middling request
  }

  const completion = Number(payload.max_completion_tokens) || Number(payload.max_tokens) || 1000;

  let promptChars = 0;
  let images = 0;
  for (const msg of payload.messages || []) {
    const c = msg.content;
    if (typeof c === "string") promptChars += c.length;
    else if (Array.isArray(c)) {
      for (const part of c) {
        if (part.type === "text") promptChars += (part.text || "").length;
        else if (part.type === "image_url") images++;
      }
    }
  }

  return Math.ceil(promptChars / 4) + images * 1900 + completion + 200; // 200 = envelope slack
}

const DEFAULT_LIMIT = 8000; // free-tier tokens per minute, until a header says otherwise

class Key {
  constructor(value, index) {
    this.value = value;
    this.index = index;
    this.limit = DEFAULT_LIMIT;
    this.remaining = DEFAULT_LIMIT; // optimistic until the first response
    this.resetAt = 0;
    this.inFlight = 0;
    this.reserved = 0; // cost of requests sent but not yet answered
    this.calls = 0;
    this.rateLimited = 0;
    /* A revoked or mistyped key never recovers. Cooling it down would have the
       pool keep hopefully retrying it once a minute forever. */
    this.dead = false;
  }

  /** Budget refills when the window rolls over. */
  available(now) {
    if (this.dead) return 0;
    if (this.resetAt && now >= this.resetAt) {
      this.remaining = this.limit;
      this.resetAt = 0;
    }
    return Math.max(0, this.remaining - this.reserved);
  }

  waitMs(now) {
    return this.resetAt ? Math.max(0, this.resetAt - now) : 0;
  }
}

export class KeyPool {
  constructor(keys, { log = () => {} } = {}) {
    this.keys = keys.map((k, i) => new Key(k, i));
    this.log = log;
  }

  get size() {
    return this.keys.length;
  }

  /** The largest single request any key could ever afford. */
  get ceiling() {
    return this.keys.reduce((m, k) => Math.max(m, k.limit), 0);
  }

  /**
   * The best key for a request of this cost, or null if none can afford it now.
   * Prefers the most headroom, then the least busy, so concurrent requests
   * spread across the pool instead of queueing behind one key.
   */
  pick(cost, now = Date.now()) {
    let best = null;
    let bestScore = -Infinity;

    for (const k of this.keys) {
      const avail = k.available(now);
      if (avail < cost) continue;
      const score = avail - k.inFlight * 1000;
      if (score > bestScore) {
        bestScore = score;
        best = k;
      }
    }
    return best;
  }

  /** How long until some key could afford `cost`. Infinity if none ever could. */
  soonest(cost, now = Date.now()) {
    let wait = Infinity;
    for (const k of this.keys) {
      if (k.dead || k.limit < cost) continue; // this key could never serve it
      wait = Math.min(wait, k.available(now) >= cost ? 0 : k.waitMs(now) || 1000);
    }
    return wait;
  }

  reserve(key, cost) {
    key.reserved += cost;
    key.inFlight++;
    key.calls++;
  }

  release(key, cost) {
    key.reserved = Math.max(0, key.reserved - cost);
    key.inFlight = Math.max(0, key.inFlight - 1);
  }

  /**
   * Learn this key's true budget from the response Groq just sent.
   * These headers are authoritative; the local estimate is only a stand-in
   * between calls.
   */
  observe(key, headers, status, cost) {
    const now = Date.now();
    const num = (h) => {
      const v = headers.get(h);
      const n = v == null ? NaN : Number(v);
      return Number.isFinite(n) ? n : null;
    };

    const limit = num("x-ratelimit-limit-tokens");
    const remaining = num("x-ratelimit-remaining-tokens");
    const reset = parseReset(headers.get("x-ratelimit-reset-tokens"));

    if (limit !== null) key.limit = limit;
    if (remaining !== null) key.remaining = remaining;
    else if (status < 400) key.remaining = Math.max(0, key.remaining - cost);
    if (reset !== null) key.resetAt = now + reset;

    if (status === 401 || status === 403) {
      key.dead = true;
      key.remaining = 0;
      return;
    }

    if (status === 429 || status === 413) {
      key.rateLimited++;
      // Believe the retry-after over anything else.
      const retry = parseReset(headers.get("retry-after"));
      if (retry !== null) key.resetAt = now + retry;
      else if (!key.resetAt) key.resetAt = now + 20000;
      // Whatever it said, this key cannot serve this request now.
      if (remaining === null) key.remaining = 0;
    }
  }

  stats() {
    const now = Date.now();
    return this.keys.map((k) => ({
      key: k.index + 1,
      remaining: k.available(now),
      limit: k.limit,
      calls: k.calls,
      dead: k.dead,
      rateLimited: k.rateLimited,
      resetInMs: k.waitMs(now),
    }));
  }
}
