# QIQ Paper Checker

**The product of [QualCrypt](https://qualcrypt.com).** QIQ reads a scanned or
photographed exam answer sheet, works out which answer belongs to which
question, marks each answer against the teacher's own reference material, and
draws the marks back onto the student's handwriting — then hands the whole
thing to a human examiner who can override any of it.

- **Live app:** https://app.qiq.academy
- **Product page:** https://qualcrypt.com/ai-solutions.html
- **Pipeline internals:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Production deployment:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## How it is built

- **React 19 + Vite** frontend — no UI framework, no state library.
- **A dependency-free marking engine** (`src/engine/`): OCR structuring,
  question identity, "answer any 3 of 5" choice rules, answer↔question
  matching, BM25 retrieval over the teacher's material, per-question
  assessment, ink geometry and mark placement.
- **A server-side AI layer** (`server.js` + `src/server/`): all Groq calls go
  through a key-scheduling proxy. API keys live only in the server
  environment — never in the browser, never in this repo.
- **Models** (Groq): `qwen/qwen3.6-27b` vision/OCR, `openai/gpt-oss-120b`
  reasoning.

## Run it locally

```bash
cp .env.example .env      # add GROQ_API_KEY (or GROQ_API_KEYS=a,b,c)
npm install
npm run dev               # Vite dev server + the /api/groq proxy on :5173
```

## Run it in production

```bash
npm install               # postinstall builds dist/
npm start                 # server.js: dist/ + /api/groq on $PORT (default 3000)
```

One Node process serves the frontend **and** the AI proxy. Full Hostinger
VPS + Nginx + SSL + GitHub auto-deploy instructions are in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Test

```bash
npm test                  # seven suites, ~330 assertions, no network
```
