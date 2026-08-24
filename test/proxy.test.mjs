/**
 * The Groq proxy: key pooling and rate-limit failover.
 *
 * Drives the real middleware with a stubbed fetch, so the failover path is
 * exercised without needing to actually exhaust a key's token allowance.
 */
import { EventEmitter } from "node:events";
import { groqProxy, collectKeys } from "../vite.config.js";
import { KeyPool } from "../src/server/keypool.js";

/* The proxy now takes a scheduled pool rather than a bare array. */
/* maxWaitMs 0 keeps the fail-fast assertions instant; the waiting path is
   asserted separately with a short budget. */
const proxy = (keys, opts = { maxWaitMs: 0, maxAttempts: 4 }) => groqProxy(new KeyPool(keys), opts);

let failures = 0;
const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) failures++;
};
const section = (s) => console.log("\n" + s);

/* ------------------------------------------------------------ key pool -- */
section("key pool");

t("single key", collectKeys({ GROQ_API_KEY: "a" }).length === 1);
t("numbered keys, in order", (() => {
  const k = collectKeys({ GROQ_API_KEY: "a", GROQ_API_KEY_2: "b", GROQ_API_KEY_3: "c" });
  return k.join() === "a,b,c";
})());
t("comma-separated list", collectKeys({ GROQ_API_KEYS: "a, b ,c" }).join() === "a,b,c");
t("duplicates dropped", collectKeys({ GROQ_API_KEY: "a", GROQ_API_KEY_2: "a" }).length === 1);
t("blank values ignored", collectKeys({ GROQ_API_KEY: "", GROQ_API_KEY_2: "b" }).join() === "b");
t("no keys -> empty pool", collectKeys({}).length === 0);
t("gaps in numbering are skipped, not fatal",
  collectKeys({ GROQ_API_KEY: "a", GROQ_API_KEY_4: "d" }).join() === "a,d");

/* ------------------------------------------------------------- harness -- */

function makeReq(method = "POST", url = "/api/groq/chat/completions", body = '{"model":"x"}') {
  const req = new EventEmitter();
  req.method = method;
  req.url = url;
  process.nextTick(() => {
    req.emit("data", Buffer.from(body));
    req.emit("end");
  });
  return req;
}

function makeRes() {
  const headers = {};
  let resolve;
  const done = new Promise((r) => (resolve = r));
  return {
    statusCode: 0,
    headers,
    setHeader: (k, v) => (headers[k.toLowerCase()] = v),
    end: (text) => resolve({ text }),
    done,
  };
}

/** Stub upstream that returns a scripted status per call, recording auth used. */
function stubFetch(statuses) {
  const seen = [];
  global.fetch = async (_url, opts) => {
    const status = statuses[seen.length] ?? 200;
    seen.push(opts.headers.authorization.replace("Bearer ", ""));
    return {
      status,
      headers: { get: (h) => (h === "retry-after" && status === 429 ? "12" : null) },
      text: async () => JSON.stringify({ status }),
    };
  };
  return seen;
}

const run = async (plugin, req) => {
  const res = makeRes();
  let nexted = false;
  const handler = { middlewares: { use: (h) => (handler.fn = h) } };
  plugin.configureServer(handler);
  const p = handler.fn(req, res, () => {
    nexted = true;
    res.end("");
  });
  await Promise.all([p, res.done]);
  return { res, nexted };
};

/* ------------------------------------------------------------ failover -- */
section("rate-limit failover");

{
  const seen = stubFetch([200]);
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("healthy first key is used alone", seen.length === 1 && seen[0] === "k1");
  t("status passed through", res.statusCode === 200);
  t("reports which key served it", res.headers["x-qiq-key"] === "1/2");
}

{
  const seen = stubFetch([429, 200]);
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("429 on key 1 fails over to key 2", seen.join() === "k1,k2");
  t("client sees the success, not the 429", res.statusCode === 200);
  t("reports key 2 served it", res.headers["x-qiq-key"] === "2/2");
}

{
  const seen = stubFetch([413, 429, 200]);
  const { res } = await run(proxy(["k1", "k2", "k3"]), makeReq());
  t("413 also triggers failover", seen.join() === "k1,k2,k3");
  t("walks the whole pool until one works", res.statusCode === 200);
}

{
  const seen = stubFetch([401, 200]);
  const { res } = await run(proxy(["revoked", "good"]), makeReq());
  t("401 (revoked key) reroutes to a live key", seen.join() === "revoked,good");
  t("  and the client sees the success", res.statusCode === 200);
}

{
  /* A dead key must not be retried on every subsequent request. */
  const pool = new KeyPool(["revoked", "good"]);
  const plugin = groqProxy(pool, { maxWaitMs: 0, maxAttempts: 4 });
  stubFetch([401, 200]);
  await run(plugin, makeReq());
  const seen2 = stubFetch([200]);
  await run(plugin, makeReq());
  t("a revoked key is retired, not retried forever", seen2.join() === "good");
  t("  and is reported dead in the pool stats", pool.stats()[0].dead === true);
}

{
  const seen = stubFetch([429, 429, 429, 429]);
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("all keys exhausted -> 429 reaches the client", res.statusCode === 429);
  t("  (so groqChat's own back-off still engages)", res.headers["retry-after"] === "12");
  t("every key was tried before giving up", new Set(seen).size === 2);
  t("reports the estimated cost for diagnosis", Number(res.headers["x-qiq-cost"]) > 0);
}

{
  /* The point of the pool: rather than failing the moment every key is spent,
     hold the request until the soonest window refills. */
  let call = 0;
  global.fetch = async (_u, o) => {
    call++;
    const spent = call === 1;
    return {
      status: spent ? 429 : 200,
      headers: { get: (h) => (h === "retry-after" && spent ? "0.3" : null) },
      text: async () => JSON.stringify({ ok: !spent }),
    };
  };
  const started = Date.now();
  const { res } = await run(proxy(["solo"], { maxWaitMs: 5000, maxAttempts: 4 }), makeReq());
  const took = Date.now() - started;
  t("a single spent key is waited out, not failed", res.statusCode === 200);
  t("  (and it actually waited)", took >= 300 && Number(res.headers["x-qiq-waited-ms"]) >= 300);
}

{
  /* A request larger than any single key's whole window: more keys cannot help,
     so it must be refused up front rather than burning the pool. */
  const seen = stubFetch([200]);
  const big = JSON.stringify({ max_completion_tokens: 500000, messages: [{ content: "hi" }] });
  const { res } = await run(proxy(["k1", "k2", "k3", "k4"]), makeReq("POST", "/api/groq/chat/completions", big));
  t("oversized request is refused without any upstream call", seen.length === 0);
  t("  reported as 413 request_too_large", res.statusCode === 413 && res.text === undefined ? true : res.statusCode === 413);
}

/* ------------------------------------------------- non-key failures ----- */
section("failures that are not the key's fault");

{
  const seen = stubFetch([400, 200]);
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("400 does NOT burn the pool", seen.length === 1);
  t("400 passed straight through", res.statusCode === 400);
}

{
  const seen = stubFetch([500, 200]);
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("500 does NOT burn the pool", seen.length === 1 && res.statusCode === 500);
}

{
  global.fetch = async () => {
    throw new Error("ECONNREFUSED");
  };
  const { res } = await run(proxy(["k1", "k2"]), makeReq());
  t("network failure returns 502, does not burn the pool", res.statusCode === 502);
}

/* ---------------------------------------------------------- guard rails -- */
section("guard rails");

{
  stubFetch([200]);
  const { res } = await run(proxy([]), makeReq());
  t("no keys configured -> clear 500", res.statusCode === 500);
}

{
  stubFetch([200]);
  const { res } = await run(proxy(["k1"]), makeReq("GET"));
  t("GET is rejected 405", res.statusCode === 405);
}

{
  stubFetch([200]);
  const { nexted } = await run(proxy(["k1"]), makeReq("POST", "/some/other/path"));
  t("unrelated paths fall through to the next middleware", nexted === true);
}

{
  let sentAuth = null;
  global.fetch = async (_u, o) => {
    sentAuth = o.headers.authorization;
    return { status: 200, headers: { get: () => null }, text: async () => "{}" };
  };
  const { res } = await run(proxy(["secret-key-value"]), makeReq());
  t("key IS sent upstream", sentAuth === "Bearer secret-key-value");
  t("key is NOT echoed to the client in any header",
    !JSON.stringify(res.headers).includes("secret-key-value"));
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall proxy assertions passed");
process.exitCode = failures ? 1 : 0;
