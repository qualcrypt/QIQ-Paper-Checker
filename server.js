/**
 * QIQ Paper Checker — production server.
 *
 * One Node process serves both halves of the product:
 *
 *   1. the built React app (dist/, produced by `npm run build`), and
 *   2. the server-side AI layer (/api/groq/*), which schedules requests
 *      across the Groq key pool in src/server/keypool.js.
 *
 * The browser never sees a Groq key: it posts to same-origin /api/groq and
 * this process attaches the Authorization header upstream. Keys come from the
 * environment only (GROQ_API_KEY / GROQ_API_KEYS — see .env.example).
 *
 * This is deliberately the same proxy code the Vite dev server uses
 * (src/server/proxy.js), so behaviour in development and production matches.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";
import rateLimit from "express-rate-limit";

import { KeyPool } from "./src/server/keypool.js";
import { collectKeys, createGroqHandlers } from "./src/server/proxy.js";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.join(ROOT, "dist");

/* A .env file is a convenience for local runs; real deployments (Hostinger)
   set the environment directly, and existing process.env values always win.
   Dependency-free on purpose: KEY=VALUE lines, optional quotes, # comments. */
function loadEnvFile(file) {
  let text;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return;
  }
  for (const line of text.split("\n")) {
    if (line.trimStart().startsWith("#")) continue;
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*\S)\s*$/.exec(line);
    if (!m) continue;
    let value = m[2];
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[m[1]] === undefined) process.env[m[1]] = value;
  }
}

loadEnvFile(path.join(ROOT, ".env"));

const int = (name, fallback) => {
  const n = Number(process.env[name]);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
};

const PORT = int("PORT", 3000); // Hostinger injects PORT; 3000 for local use
/* Per-IP ceiling on the AI proxy. A full paper run costs roughly one call per
   page plus one per question over several minutes, so 120/min is far above
   legitimate single-user traffic and still stops quota-scraping. */
const PROXY_RATE_LIMIT = int("PROXY_RATE_LIMIT_PER_MIN", 120);
/* Page images travel as base64 JSON; a big upload batch stays well under this. */
const MAX_BODY_BYTES = int("MAX_BODY_MB", 25) * 1024 * 1024;

const keys = collectKeys(process.env);
const pool = new KeyPool(keys, { log: (m) => console.log(`[qiq] ${m}`) });

console.log(
  keys.length
    ? `[qiq] Groq proxy ready — ${keys.length} key${keys.length === 1 ? "" : "s"}`
    : "[qiq] WARNING: no GROQ_API_KEY set — AI features will return missing_api_key"
);

const app = express();
app.disable("x-powered-by");
app.set("trust proxy", 1); // Hostinger terminates TLS in front of this process

/* Health check for the hosting panel / uptime monitors. Reports whether keys
   exist, never their values. */
app.get("/api/health", (req, res) => {
  res.json({ ok: true, keys: pool.size });
});

/* The AI layer. Rate-limited per IP, body-size guarded, then handed to the
   same key-scheduling handlers the dev server uses. The handlers read the raw
   request stream themselves, so no body parser may run before them. */
const groqLimiter = rateLimit({
  windowMs: 60_000,
  limit: PROXY_RATE_LIMIT,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: { message: "Too many requests — slow down.", code: "rate_limit_exceeded" } },
});

app.use("/api/groq", groqLimiter, (req, res, next) => {
  const len = Number(req.headers["content-length"]);
  if (Number.isFinite(len) && len > MAX_BODY_BYTES) {
    res.statusCode = 413;
    res.setHeader("content-type", "application/json");
    return res.end(
      JSON.stringify({ error: { message: "Request body too large.", code: "payload_too_large" } })
    );
  }
  next();
});

const { statsHandler, handler } = createGroqHandlers(pool);
app.use(statsHandler);
app.use(handler);

/* The frontend. Static assets first; anything else is an SPA route and gets
   index.html. */
if (!fs.existsSync(path.join(DIST_DIR, "index.html"))) {
  console.warn("[qiq] WARNING: dist/index.html not found — run `npm run build` first.");
}
app.use(express.static(DIST_DIR, { maxAge: "1h", index: false }));
app.get("*", (req, res) => res.sendFile(path.join(DIST_DIR, "index.html")));

app.listen(PORT, () => {
  console.log(`[qiq] QIQ Paper Checker listening on port ${PORT}`);
});
