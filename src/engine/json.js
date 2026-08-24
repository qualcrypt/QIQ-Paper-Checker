/**
 * Tolerant JSON extraction plus a tiny schema validator.
 *
 * Every LLM stage in this engine returns JSON, and a malformed reply must never
 * be allowed to corrupt an assessment. The order is always: parse strictly →
 * recover structurally → validate against a schema → only then use the value.
 */

/** Thrown when a model reply cannot be turned into the expected shape. */
export class SchemaError extends Error {
  constructor(message, details) {
    super(message);
    this.name = "SchemaError";
    this.details = details || [];
  }
}

/**
 * Pull a JSON value out of a reply that may be fenced, prefaced or truncated.
 * @param {string} raw
 * @returns {any}
 */
export function extractJson(raw) {
  let s = String(raw || "").trim();

  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) s = fence[1].trim();

  try {
    return JSON.parse(s);
  } catch {
    /* fall through */
  }

  const start = firstStructuralIndex(s);
  if (start === -1) throw new SchemaError("No JSON value found in the model reply.");

  const balanced = sliceBalanced(s, start);
  if (balanced !== null) {
    try {
      return JSON.parse(balanced);
    } catch {
      /* fall through to repair */
    }
  }

  const repaired = repairTruncated(s.slice(start));
  if (repaired !== null) {
    try {
      return JSON.parse(repaired);
    } catch {
      /* give up below */
    }
  }

  throw new SchemaError("The model reply contained malformed JSON.");
}

function firstStructuralIndex(s) {
  const obj = s.indexOf("{");
  const arr = s.indexOf("[");
  if (obj === -1) return arr;
  if (arr === -1) return obj;
  return Math.min(obj, arr);
}

/** Walk from `start` and return the balanced object/array substring, if closed. */
function sliceBalanced(s, start) {
  const open = s[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inStr = false;
  let esc = false;

  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) return s.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * Close a reply that ran out of tokens mid-structure. The trailing partial
 * element is dropped rather than guessed at, so nothing is invented.
 */
function repairTruncated(s) {
  const stack = [];
  let inStr = false;
  let esc = false;
  let lastSafe = -1;

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{" || ch === "[") stack.push(ch === "{" ? "}" : "]");
    else if (ch === "}" || ch === "]") stack.pop();
    else if (ch === "," && stack.length > 0) lastSafe = i;
  }

  if (stack.length === 0) return null;
  // Cut back to the last completed element, then close every open container.
  const cut = lastSafe > 0 ? s.slice(0, lastSafe) : s.replace(/[,\s]*$/, "");
  const closers = stack.reverse().join("");
  return cut + closers;
}

/* ---------------------------------------------------------- validation -- */

/**
 * Minimal declarative validator. Supported node shapes:
 *   {type:"object", required:[...], props:{...}, unknown:"strip"|"keep"}
 *   {type:"array", of: node, min, max}
 *   {type:"string", enum:[...], min, max, default}
 *   {type:"number", min, max, integer, default}
 *   {type:"boolean", default}
 *
 * Returns a *coerced copy*: numbers-as-strings become numbers, out-of-range
 * values clamp, and unknown keys are stripped. Coercion is what keeps a
 * cosmetically wrong reply usable instead of throwing the whole paper away.
 *
 * @param {any} value
 * @param {any} schema
 * @param {string} [path]
 * @returns {any}
 */
export function validate(value, schema, path = "$") {
  const errors = [];
  const out = walk(value, schema, path, errors);
  if (errors.length) throw new SchemaError(`Schema validation failed: ${errors[0]}`, errors);
  return out;
}

/** Same as `validate` but returns a result object instead of throwing. */
export function tryValidate(value, schema) {
  try {
    return { ok: true, value: validate(value, schema), errors: [] };
  } catch (e) {
    return { ok: false, value: null, errors: e instanceof SchemaError ? e.details : [String(e)] };
  }
}

function walk(value, schema, path, errors) {
  if (!schema || !schema.type) return value;

  if (value === undefined || value === null) {
    if ("default" in schema) return schema.default;
    if (schema.optional) return undefined;
    errors.push(`${path} is required`);
    return undefined;
  }

  switch (schema.type) {
    case "object": {
      if (typeof value !== "object" || Array.isArray(value)) {
        errors.push(`${path} must be an object`);
        return {};
      }
      const out = {};
      const props = schema.props || {};
      for (const key of Object.keys(props)) {
        const sub = props[key];
        const has = Object.prototype.hasOwnProperty.call(value, key);
        const required = (schema.required || []).includes(key);
        if (!has && !required && !("default" in sub)) continue;
        if (!has && required && !("default" in sub)) {
          errors.push(`${path}.${key} is required`);
          continue;
        }
        const v = walk(has ? value[key] : undefined, sub, `${path}.${key}`, errors);
        if (v !== undefined) out[key] = v;
      }
      if (schema.unknown === "keep") {
        for (const key of Object.keys(value)) if (!(key in out)) out[key] = value[key];
      }
      return out;
    }

    case "array": {
      if (!Array.isArray(value)) {
        errors.push(`${path} must be an array`);
        return [];
      }
      const out = [];
      for (let i = 0; i < value.length; i++) {
        const before = errors.length;
        const v = walk(value[i], schema.of, `${path}[${i}]`, errors);
        // Drop only the bad element; one broken row must not void the list.
        if (errors.length > before) {
          errors.length = before;
          continue;
        }
        if (v !== undefined) out.push(v);
      }
      if (schema.min != null && out.length < schema.min) {
        errors.push(`${path} needs at least ${schema.min} item(s), got ${out.length}`);
      }
      return schema.max != null ? out.slice(0, schema.max) : out;
    }

    case "string": {
      let v = typeof value === "string" ? value : String(value);
      v = v.trim();
      if (schema.enum && !schema.enum.includes(v)) {
        const lowered = schema.enum.find((e) => e.toLowerCase() === v.toLowerCase());
        if (lowered) v = lowered;
        else if ("default" in schema) return schema.default;
        else {
          errors.push(`${path} must be one of ${schema.enum.join("|")}, got "${v}"`);
          return undefined;
        }
      }
      if (schema.min != null && v.length < schema.min) {
        if ("default" in schema) return schema.default;
        errors.push(`${path} is shorter than ${schema.min}`);
        return undefined;
      }
      if (schema.max != null && v.length > schema.max) v = v.slice(0, schema.max);
      return v;
    }

    case "number": {
      const n = typeof value === "number" ? value : Number(String(value).replace(/[^\d.+-]/g, ""));
      if (!Number.isFinite(n)) {
        if ("default" in schema) return schema.default;
        errors.push(`${path} must be a number, got "${value}"`);
        return undefined;
      }
      let v = schema.integer ? Math.round(n) : n;
      if (schema.min != null) v = Math.max(schema.min, v);
      if (schema.max != null) v = Math.min(schema.max, v);
      return v;
    }

    case "boolean":
      if (typeof value === "boolean") return value;
      if (value === "true") return true;
      if (value === "false") return false;
      return "default" in schema ? schema.default : Boolean(value);

    default:
      return value;
  }
}
