# Deploying

Two pieces: the FastAPI backend (Render or Railway, via Docker) and the
Next.js frontend (Vercel). ~10 minutes total.

## Demo-mode key policy (important)

The public deployment ships **without** a server-side OpenRouter key:

- Browsing all existing results, charts, exports → works for everyone, free.
- Running new evals → the visitor pastes **their own** key into the sidebar
  input. It is stored only in their browser's localStorage and sent
  per-request (`api_key` query param). Your credits cannot be spent by
  strangers.
- For a private deployment where live evals "just work", set
  `OPENROUTER_API_KEY` in the backend environment instead.

## Backend — Render (free tier)

1. Push this repo to GitHub (already done if you're reading this there).
2. [dashboard.render.com](https://dashboard.render.com) → New → Blueprint →
   pick this repo/branch. `render.yaml` configures everything (Docker build
   from `backend/`, health check on `/health`).
3. Note the service URL, e.g. `https://llm-eval-api.onrender.com`.

Free-tier notes: the instance sleeps after idle (first request takes ~30s),
and the **filesystem is ephemeral** — results created on the deployment are
lost on redeploy. The CSVs committed in `backend/results/` are the demo data
every deploy starts from. For persistence, attach a Render Disk mounted at
`/app/results` (paid) or just treat the demo as read-mostly.

## Backend — Railway (alternative)

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub repo.
2. Set the service **Root Directory** to `backend/` — Railway auto-detects the
   Dockerfile. Same ephemeral-filesystem caveat; Railway Volumes fix it.

## Frontend — Vercel

1. [vercel.com/new](https://vercel.com/new) → import the repo.
2. Set **Root Directory** to `frontend/`.
3. Add env var `NEXT_PUBLIC_API_BASE` = your backend URL
   (e.g. `https://llm-eval-api.onrender.com`).
4. Deploy. CORS already allows `*.vercel.app`; for a custom domain, set
   `ALLOWED_ORIGINS=https://yourdomain.com` on the backend.

## Local docker sanity check

```bash
cd backend
docker build -t llm-eval-api .
docker run -p 8000:8000 --env-file .env llm-eval-api
curl localhost:8000/health
```
