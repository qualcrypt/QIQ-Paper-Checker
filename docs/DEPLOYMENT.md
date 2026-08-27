# Deploying QIQ Paper Checker

Two supported paths. **Render Free** is the current production setup ($0);
the Hostinger VPS path is kept as the self-hosted alternative.

## Architecture (same on both)

```
Browser (app.qiq.academy / *.onrender.com)
   │  HTTPS
   ▼
Node 20 process (server.js)
   ├─ serves the built React app from dist/
   └─ /api/groq/*  ── HTTPS ──►  api.groq.com   (Authorization attached
                                                 server-side from the
                                                 GROQ_API_KEY env var)
```

- **Frontend**: static build (`npm run build` → `dist/`), served by the same
  Node process — no separate static hosting.
- **AI keys**: only in the host's environment variables. Never in the
  browser, never in the repo.
- **qualcrypt.com** stays on Hostinger (separate repo, untouched) and links
  here via the "Try QIQ" button.

---

## Path 1 — Render Free (current, $0)

### 1. Create the service from the Blueprint

1. Sign in at <https://render.com> with GitHub (grant access to
   `qualcrypt/QIQ-Paper-Checker`).
2. Dashboard → **New → Blueprint** → select the repo. Render reads
   `render.yaml` and pre-fills everything (free plan, build, start, health
   check, auto-deploy).
3. When prompted, paste your **GROQ_API_KEY** (from
   <https://console.groq.com/keys>). It is stored as a Render environment
   variable — server-side only. Add more keys later as `GROQ_API_KEYS=a,b,c`
   under the service's Environment tab for more per-minute headroom.
4. **Apply** — Render runs `npm install` (the postinstall script builds
   `dist/`), then `npm start`.

### 2. Verify the free URL

Render assigns `https://qiq-paper-checker.onrender.com`:

```bash
curl https://qiq-paper-checker.onrender.com/api/health     # {"ok":true,"keys":1}
curl https://qiq-paper-checker.onrender.com/api/groq/stats # pool view, no keys
```

Then open the URL and run a full paper through: upload question paper +
answer sheet → processing trace → evaluation → examiner review → report card.

**Free-plan caveat:** the service sleeps after ~15 minutes idle; the first
request after that waits ~30–60s for a cold start. Fine for a demo/early
product; upgrade to a paid instance later to remove it.

### 3. Custom domain — app.qiq.academy

Render free web services support custom domains with free managed TLS:

1. Render dashboard → your service → **Settings → Custom Domains** →
   **Add Custom Domain** → `app.qiq.academy`.
2. In the `qiq.academy` DNS zone (Hostinger hPanel → Domains → DNS, if
   managed there) add:

   | Type | Name | Value |
   |------|------|-------|
   | CNAME | `app` | `qiq-paper-checker.onrender.com` |

3. Back in Render, click **Verify** — the certificate issues automatically.

If the free plan ever refuses the custom domain, the app keeps working on
the `.onrender.com` URL; point the "Try QIQ" button there instead.

### 4. Automatic deployment

Already on: `autoDeploy: true` in `render.yaml`. Every push to `main`
rebuilds and redeploys — no GitHub Action needed for Render.

---

## Path 2 — Hostinger VPS (self-hosted alternative)

> Note: Hostinger **shared** hosting (Premium/Business) runs PHP only and
> cannot host this app. This path needs a VPS; it costs money, which is why
> Render is the current setup.

```bash
# On the VPS (Ubuntu):
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs nginx certbot python3-certbot-nginx git
npm install -g pm2

mkdir -p /var/www && cd /var/www
git clone https://github.com/qualcrypt/QIQ-Paper-Checker.git qiq
cd qiq
cp .env.example .env      # set GROQ_API_KEY
npm install               # postinstall builds dist/
PORT=3000 pm2 start server.js --name qiq
pm2 save && pm2 startup
```

Nginx (`/etc/nginx/sites-available/qiq`):

```nginx
server {
    server_name app.qiq.academy;
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 25m;   # page images travel as base64 JSON
    }
}
```

```bash
ln -s /etc/nginx/sites-available/qiq /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d app.qiq.academy
```

DNS: `A` record `app.qiq.academy` → VPS IP. For push-to-deploy on this path,
`.github/workflows/deploy.yml` can be run manually (Actions tab) with the
`VPS_HOST` / `VPS_USER` / `VPS_SSH_KEY` repo secrets set.

---

## Environment variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `GROQ_API_KEY` / `GROQ_API_KEYS` | Render env / VPS `.env` | Groq keys, server-side only |
| `PORT` | set by Render automatically | listen port (3000 default locally) |
| `PROXY_RATE_LIMIT_PER_MIN` | optional | per-IP ceiling on `/api/groq` (default 120) |
| `MAX_BODY_MB` | optional | max request body (default 25) |
| `VITE_API_URL` | build-time, optional | only for a split deployment (frontend and backend on different origins) |

## Split deployment (optional later)

Build with `VITE_API_URL=https://api.example.com npm run build`, host `dist/`
statically, run `server.js` on the API host. Same-origin (the default) is
what both paths above use.
