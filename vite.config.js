import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

import { KeyPool, estimateCost } from "./src/server/keypool.js";

const GROQ_UPSTREAM = "https://api.groq.com/openai/v1/chat/completions";

/* Errors raised by the proxy itself are shaped like Groq's own error body, so
   groqChat() in the browser can report them through its existing paths instead
   of needing a second, proxy-specific error format. */
function errorBody(message, code) {
  return JSON.stringify({ error: { message, code: code || "proxy_error" } });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* How long the proxy will hold a request waiting for a key's window to refill
   before giving up and letting the browser's own back-off take over. A minute
   covers a full Groq token window; beyond that the caller deserves an answer. */
const MAX_WAIT_MS = 60000;
const MAX_ATTEMPTS = 6;

/* Statuses where another key is worth trying. 429/413 mean this key is spent;
   401/403 mean it is invalid. Everything else is a property of the request and
   would fail identically on every key. */
const REROUTABLE = new Set([429, 413, 401, 403]);

/**
 * Same-origin proxy for the Groq API, scheduling across a pool of keys.
 *
 * The keys never reach the browser: the client posts to /api/groq and this
 * middleware attaches the Authorization header server-side.
 *
 * Scheduling, not just failover. Each request is costed up front (Groq bills
 * prompt + max_completion_tokens against a per-minute ceiling before it runs)
 * and routed to a key with room for it. With four keys that is roughly four
 * times the per-minute headroom, and — more usefully — a request that would
 * have been rejected as 413 against a nearly-spent key simply goes to a fresh
 * one instead.
 *
 * When no key has room the request waits for the soonest refill rather than
 * failing, because failing is what the caller would do anyway. The one thing no
 * pool can fix is a request larger than a single key's entire allowance; that is
 * detected before any call is made and reported as its own problem.
 */
export function groqProxy(pool, { maxWaitMs = MAX_WAIT_MS, maxAttempts = MAX_ATTEMPTS } = {}) {
  const handler = async (req, res, next) => {
    const path = (req.url || "").split("?")[0];
    if (path !== "/api/groq/chat/completions") return next();

    res.setHeader("content-type", "application/json");

    if (req.method !== "POST") {
      res.statusCode = 405;
      return res.end(errorBody("The Groq proxy only accepts POST.", "method_not_allowed"));
    }
    if (pool.size === 0) {
      res.statusCode = 500;
      return res.end(
        errorBody(
          "No GROQ_API_KEY is set on the server. Copy .env.example to .env, add a key, and restart.",
          "missing_api_key"
        )
      );
    }

    let body;
    try {
      body = await readBody(req);
    } catch (e) {
      res.statusCode = 400;
      return res.end(errorBody(`Could not read the request body: ${e.message}`, "bad_request"));
    }

    const cost = estimateCost(body);

    /* Too big for any key's full window. More keys will never help — the
       request itself has to shrink. Say so plainly instead of returning a
       rate-limit error that invites the user to just wait and retry. */
    if (cost > pool.ceiling) {
      res.statusCode = 413;
      res.setHeader("x-qiq-cost", String(cost));
      return res.end(
        errorBody(
          `This single request needs about ${cost} tokens, more than one key's entire ` +
            `per-minute allowance of ${pool.ceiling}. Send fewer pages at once, or reduce the ` +
            `token budget for this step — adding more keys cannot help with this.`,
          "request_too_large"
        )
      );
    }

    const started = Date.now();
    let waited = 0;
    let lastStatus = 0;
    let lastText = "";
    let lastHeaders = null;
    let usedKey = 0;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      let key = pool.pick(cost);

      if (!key) {
        // Every key is spent. Wait for whichever refills first.
        const wait = pool.soonest(cost);
        if (!Number.isFinite(wait) || waited + wait > maxWaitMs) break;
        const nap = Math.min(wait + 150, maxWaitMs - waited);
        if (nap <= 0) break;
        pool.log(`all ${pool.size} keys spent; waiting ${Math.round(nap / 1000)}s for a refill`);
        await sleep(nap);
        waited += nap;
        key = pool.pick(cost);
        if (!key) continue;
      }

      pool.reserve(key, cost);
      let upstream;
      try {
        upstream = await fetch(GROQ_UPSTREAM, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${key.value}`,
          },
          body,
        });
      } catch (e) {
        pool.release(key, cost);
        // A network failure is not the key's fault; do not burn the pool on it.
        res.statusCode = 502;
        return res.end(
          errorBody(`Could not reach the Groq API: ${e.message}`, "upstream_unreachable")
        );
      }

      const text = await upstream.text();
      pool.release(key, cost);
      pool.observe(key, upstream.headers, upstream.status, cost);

      lastStatus = upstream.status;
      lastText = text;
      lastHeaders = upstream.headers;
      usedKey = key.index + 1;

      if (!REROUTABLE.has(upstream.status)) break;

      pool.log(
        `key ${usedKey} returned ${upstream.status} (cost ~${cost}); ` +
          `rerouting — pool: ${pool
            .stats()
            .map((s) => `k${s.key}:${s.remaining}`)
            .join(" ")}`
      );
    }

    res.statusCode = lastStatus || 503;
    // Index and cost only — never a key value.
    res.setHeader("x-qiq-key", `${usedKey}/${pool.size}`);
    res.setHeader("x-qiq-cost", String(cost));
    if (waited) res.setHeader("x-qiq-waited-ms", String(waited));

    if (lastHeaders) {
      for (const h of ["retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-remaining-tokens"]) {
        const v = lastHeaders.get(h);
        if (v) res.setHeader(h, v);
      }
    }

    if (!lastStatus) {
      res.statusCode = 429;
      return res.end(
        errorBody(
          `All ${pool.size} keys are at their per-minute token limit. Waiting for a refill timed out.`,
          "rate_limit_exceeded"
        )
      );
    }
    res.end(lastText);
  };

  /* A tiny read-only view of the pool, handy while a long paper is marking. */
  const statsHandler = (req, res, next) => {
    if ((req.url || "").split("?")[0] !== "/api/groq/stats") return next();
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ keys: pool.size, pool: pool.stats() }, null, 2));
  };

  return {
    name: "qiq-groq-proxy",
    configureServer(server) {
      server.middlewares.use(statsHandler);
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(statsHandler);
      server.middlewares.use(handler);
    },
  };
}

/**
 * Collect the key pool, in the order it should be spent.
 *
 * Accepts a comma-separated GROQ_API_KEYS, or numbered
 * GROQ_API_KEY / GROQ_API_KEY_2 / GROQ_API_KEY_3 … Duplicates are dropped so a
 * key pasted twice cannot make the pool look deeper than it is — which would
 * otherwise inflate the ceiling and cause avoidable 413s.
 */
export function collectKeys(env) {
  const out = [];
  const push = (v) => {
    for (const k of String(v || "").split(",")) {
      const key = k.trim();
      if (key && !out.includes(key)) out.push(key);
    }
  };

  push(env.GROQ_API_KEYS);
  push(env.GROQ_API_KEY);
  for (let i = 2; i <= 10; i++) push(env[`GROQ_API_KEY_${i}`]);

  return out;
}

export default defineConfig(({ mode }) => {
  // The empty prefix loads every variable, not just VITE_ ones. That is safe
  // here because the values are only ever read in this file and in the pool,
  // both of which run in Node — nothing reaches the client bundle.
  const env = { ...process.env, ...loadEnv(mode, process.cwd(), "") };
  const keys = collectKeys(env);

  const pool = new KeyPool(keys, { log: (m) => console.log(`[qiq] ${m}`) });

  console.log(
    keys.length
      ? `[qiq] Groq proxy ready — ${keys.length} key${keys.length === 1 ? "" : "s"}, ` +
          `~${keys.length * 8000} tokens/min combined`
      : "[qiq] WARNING: no GROQ_API_KEY found — set one in .env"
  );

  return {
    plugins: [react(), groqProxy(pool)],
    server: { port: 5173, open: true },
  };
});
