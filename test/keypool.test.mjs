/**
 * Budget-aware scheduling across a pool of Groq keys.
 *
 * The interesting cases are the ones plain failover gets wrong: a request that
 * is too big for a nearly-spent key must go to a fresh one, and a request too
 * big for ANY key must be rejected up front instead of burning the whole pool.
 */
import { KeyPool, estimateCost, parseReset } from "../src/server/keypool.js";

let failures = 0;
const t = (label, cond) => {
  console.log("  " + (cond ? "PASS" : "FAIL") + "  " + label);
  if (!cond) failures++;
};
const section = (s) => console.log("\n" + s);

/* --------------------------------------------------------- reset parsing */
section("reset header parsing");
t('"7.66s"', parseReset("7.66s") === 7660);
t('"500ms"', parseReset("500ms") === 500);
t('"1m2.5s"', parseReset("1m2.5s") === 62500);
t('"2m59.56s"', parseReset("2m59.56s") === 179560);
t('bare "12" is seconds', parseReset("12") === 12000);
t("empty -> null", parseReset("") === null);
t("garbage -> null", parseReset("soon") === null);

/* ------------------------------------------------------- cost estimation */
section("request costing");

const bodyOf = (o) => Buffer.from(JSON.stringify(o));

t("completion budget is charged in full", (() => {
  const c = estimateCost(bodyOf({ max_completion_tokens: 3000, messages: [{ role: "u", content: "" }] }));
  return c >= 3000 && c < 3400;
})());

t("prompt text adds ~chars/4", (() => {
  const small = estimateCost(bodyOf({ max_completion_tokens: 100, messages: [{ content: "a".repeat(40) }] }));
  const big = estimateCost(bodyOf({ max_completion_tokens: 100, messages: [{ content: "a".repeat(4040) }] }));
  return big - small === 1000;
})());

t("an image is charged a flat ~1900, not by base64 length", (() => {
  const withImg = estimateCost(bodyOf({
    max_completion_tokens: 3000,
    messages: [{ content: [{ type: "text", text: "hi" }, { type: "image_url", image_url: { url: "data:image/jpeg;base64," + "A".repeat(500000) } }] }],
  }));
  return withImg > 4700 && withImg < 5300;
})());

t("an OCR page request lands near the measured ~4900", (() => {
  const c = estimateCost(bodyOf({
    max_completion_tokens: 3000,
    messages: [{ content: "Extract the student's answer from this answer paper." },
               { content: [{ type: "image_url", image_url: { url: "data:..." } }] }],
  }));
  return c > 4900 && c < 5300;
})());

t("unparseable body gets a middling estimate", estimateCost(Buffer.from("not json")) === 2000);

/* -------------------------------------------------------------- routing */
section("routing by remaining budget");

const pool4 = () => new KeyPool(["k1", "k2", "k3", "k4"]);

t("fresh pool picks a key", pool4().pick(4000) !== null);

t("combined ceiling is ONE key's limit, not the sum", (() => {
  // A pool cannot serve a request bigger than a single key's window.
  return pool4().ceiling === 8000;
})());

t("a request too large for any key is refused, not retried", (() => {
  const p = pool4();
  return 20000 > p.ceiling && p.pick(20000) === null;
})());

t("nearly-spent key is skipped for an expensive request", (() => {
  const p = pool4();
  p.keys[0].remaining = 3000; // this is the 413 case from the screenshot
  p.keys[1].remaining = 8000;
  const picked = p.pick(4900);
  return picked && picked.index !== 0;
})());

t("  ...but that same key still serves a cheap request", (() => {
  const p = pool4();
  p.keys.forEach((k) => (k.remaining = 500));
  p.keys[2].remaining = 2500;
  const picked = p.pick(2000);
  return picked && picked.index === 2;
})());

t("picks the key with the most headroom", (() => {
  const p = pool4();
  p.keys[0].remaining = 5000;
  p.keys[1].remaining = 7900;
  p.keys[2].remaining = 6000;
  p.keys[3].remaining = 4000;
  return p.pick(1000).index === 1;
})());

t("spreads concurrent work rather than queueing on one key", (() => {
  const p = pool4();
  p.keys.forEach((k) => (k.remaining = 8000));
  const picked = [];
  for (let i = 0; i < 4; i++) {
    const k = p.pick(1000);
    p.reserve(k, 1000);
    picked.push(k.index);
  }
  return new Set(picked).size === 4; // all four used, not one four times
})());

t("reserved budget stops over-committing a key in flight", (() => {
  const p = new KeyPool(["only"]);
  p.keys[0].remaining = 8000;
  const k = p.pick(5000);
  p.reserve(k, 5000);
  return p.pick(5000) === null; // 3000 left, cannot afford a second
})());

t("releasing restores the headroom", (() => {
  const p = new KeyPool(["only"]);
  const k = p.pick(5000);
  p.reserve(k, 5000);
  p.release(k, 5000);
  return p.pick(5000) !== null;
})());

/* ------------------------------------------------------ learning limits */
section("learning real budgets from Groq's headers");

const hdr = (o) => ({ get: (h) => (h in o ? o[h] : null) });

t("adopts the reported remaining", (() => {
  const p = pool4();
  p.observe(p.keys[0], hdr({ "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "1200" }), 200, 500);
  return p.keys[0].remaining === 1200;
})());

t("adopts a larger limit on a paid tier", (() => {
  const p = pool4();
  p.observe(p.keys[0], hdr({ "x-ratelimit-limit-tokens": "300000", "x-ratelimit-remaining-tokens": "290000" }), 200, 500);
  return p.keys[0].limit === 300000 && p.ceiling === 300000;
})());

t("without headers, spend is deducted locally", (() => {
  const p = pool4();
  const before = p.keys[0].remaining;
  p.observe(p.keys[0], hdr({}), 200, 1500);
  return p.keys[0].remaining === before - 1500;
})());

t("429 zeroes the key and sets a cooldown", (() => {
  const p = pool4();
  p.observe(p.keys[0], hdr({ "retry-after": "20" }), 429, 4000);
  return p.keys[0].available(Date.now()) === 0 && p.keys[0].waitMs(Date.now()) > 15000;
})());

t("a cooled-down key is not picked", (() => {
  const p = new KeyPool(["a", "b"]);
  p.observe(p.keys[0], hdr({ "retry-after": "20" }), 429, 4000);
  const picked = p.pick(4000);
  return picked && picked.index === 1;
})());

t("budget refills once the window rolls over", (() => {
  const p = new KeyPool(["a"]);
  p.observe(p.keys[0], hdr({ "retry-after": "20" }), 429, 4000);
  const future = Date.now() + 21000;
  return p.keys[0].available(future) === 8000;
})());

t("soonest() reports the wait when everything is spent", (() => {
  const p = new KeyPool(["a", "b"]);
  p.observe(p.keys[0], hdr({ "retry-after": "30" }), 429, 4000);
  p.observe(p.keys[1], hdr({ "retry-after": "5" }), 429, 4000);
  const w = p.soonest(4000);
  return w > 0 && w <= 6000; // the 5s key, not the 30s one
})());

t("soonest() is Infinity when no key could ever afford it", (() => {
  return !Number.isFinite(new KeyPool(["a"]).soonest(999999));
})());

/* --------------------------------------------- the screenshot's scenario */
section("the 413 from the screenshot");

t("4 keys survive an OCR run that would 413 on one", (() => {
  const p = pool4();
  const cost = 4900; // one page + 3000 completion budget
  let served = 0;
  // Each key affords one such request per window (8000 limit).
  for (let i = 0; i < 4; i++) {
    const k = p.pick(cost);
    if (!k) break;
    p.reserve(k, cost);
    p.observe(k, hdr({ "x-ratelimit-remaining-tokens": String(8000 - cost) }), 200, cost);
    p.release(k, cost);
    served++;
  }
  return served === 4; // a single key would have managed one
})());

t("and the 5th correctly finds nothing until a refill", (() => {
  const p = pool4();
  p.keys.forEach((k) => { k.remaining = 3100; k.resetAt = Date.now() + 30000; });
  return p.pick(4900) === null && Number.isFinite(p.soonest(4900));
})());

console.log(failures ? `\n${failures} FAILURE(S)` : "\nall key-pool assertions passed");
process.exitCode = failures ? 1 : 0;
