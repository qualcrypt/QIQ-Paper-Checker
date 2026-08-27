import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

import { KeyPool } from "./src/server/keypool.js";
import { groqProxy, collectKeys } from "./src/server/proxy.js";

/* Re-exported: the proxy tests import these from the config, and the logic
   itself now lives in src/server/proxy.js so the production backend shares it. */
export { groqProxy, collectKeys };

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
