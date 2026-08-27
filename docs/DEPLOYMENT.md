# Deploying QIQ Paper Checker

This document is the exact path from this repository to a live production
deployment at **app.qiq.academy** on Hostinger, with GitHub auto-deploy on
every push to `main`.

## What runs where

```
Browser (app.qiq.academy)
   │  HTTPS
   ▼
Nginx on the Hostinger VPS  ── TLS termination (Let's Encrypt)
   │  proxy_pass http://127.0.0.1:3000
   ▼
Node 20 process (server.js, kept alive by PM2)
   ├─ serves the built React app from dist/
   └─ /api/groq/*  ── HTTPS ──►  api.groq.com   (Authorization attached
                                                 server-side from the
                                                 GROQ_API_KEY env var)
```

- **Frontend**: static build (`npm run build` → `dist/`), served by the same
  Node process. No separate static hosting needed.
- **AI keys**: live only in the server environment. The browser and the GitHub
  repo never see them.
- **qualcrypt.com** stays the marketing site (separate repo,
  `qualcrypt/qualcrypt-website`) and links here via the "Try QIQ" button.

## Prerequisites

1. A **Hostinger VPS plan** — Hostinger's shared/Premium/Business web hosting
   runs PHP only and **cannot** run this Node.js app. Any VPS tier works; the
   app is a single small Node process (256 MB RAM is plenty).
2. The `qiq.academy` domain's DNS zone (Hostinger hPanel → Domains →
   **DNS / Name Servers**, if the domain is managed at Hostinger).
3. One or more Groq API keys from <https://console.groq.com/keys>.

## 1. DNS — point app.qiq.academy at the VPS

In the DNS zone for `qiq.academy`:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | `app` | `<your VPS IP>` | 300 |

## 2. Server setup (once)

SSH into the VPS (`ssh root@<VPS IP>`), then:

```bash
# Node 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs nginx certbot python3-certbot-nginx git

# Process manager — restarts the app on crash/reboot
npm install -g pm2

# Clone the product
mkdir -p /var/www && cd /var/www
git clone https://github.com/qualcrypt/QIQ-Paper-Checker.git qiq
cd qiq

# Secrets — server-side only
cp .env.example .env
nano .env        # set GROQ_API_KEY (or GROQ_API_KEYS=a,b,c)

# Install + build (postinstall runs `vite build` automatically)
npm install

# Start on port 3000 and survive reboots
PORT=3000 pm2 start server.js --name qiq
pm2 save && pm2 startup
```

## 3. Nginx + SSL for app.qiq.academy

`/etc/nginx/sites-available/qiq`:

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
certbot --nginx -d app.qiq.academy     # free HTTPS certificate
```

The app is now live at **https://app.qiq.academy**.

## 4. Automatic GitHub → VPS deployment

This repo ships `.github/workflows/deploy.yml`. On every push to `main` it
SSHes into the VPS, pulls, rebuilds and restarts the app.

Set these once in **GitHub → repo → Settings → Secrets and variables →
Actions**:

| Secret | Value |
|--------|-------|
| `VPS_HOST` | your VPS IP |
| `VPS_USER` | `root` (or your deploy user) |
| `VPS_SSH_KEY` | private key allowed to SSH into the VPS |

Generate the key on the VPS if needed: `ssh-keygen -t ed25519`, append the
`.pub` to `~/.ssh/authorized_keys`, paste the **private** key into
`VPS_SSH_KEY`.

Every push to `main` then updates https://app.qiq.academy automatically.

## 5. Verifying production

```bash
curl https://app.qiq.academy/api/health          # {"ok":true,"keys":N}
curl https://app.qiq.academy/api/groq/stats      # pool view, no key values
```

Then open https://app.qiq.academy, upload a question paper + answer sheet,
and watch the live processing trace through to the report card.

## Environment variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `GROQ_API_KEY` / `GROQ_API_KEYS` | VPS `.env` | Groq keys, server-side only |
| `PORT` | systemd/PM2 env | listen port (3000 behind Nginx) |
| `PROXY_RATE_LIMIT_PER_MIN` | optional | per-IP ceiling on `/api/groq` (default 120) |
| `MAX_BODY_MB` | optional | max request body (default 25) |
| `VITE_API_URL` | build-time, optional | only needed for a split deployment where frontend and backend live on different origins |

## Split deployment (optional later)

The same codebase supports frontend-on-CDN + backend-elsewhere: build with
`VITE_API_URL=https://api.example.com npm run build`, host `dist/` statically,
and run `server.js` on the API host. Same-origin (the default above) is
simpler and is what app.qiq.academy uses.
