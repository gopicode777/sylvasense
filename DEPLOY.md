# Deploying SylvaSense to Vercel

This folder is already structured the way Vercel expects: FastAPI backend in `api/`, static frontend in `public/`, both deployed together as **one project on one domain** — no CORS setup needed, since the dashboard calls `/api/...` on the same origin it's served from.

```
sylvasense_vercel/
├── api/                  # FastAPI backend — Vercel auto-detects api/main.py
│   ├── main.py
│   ├── imagery.py
│   ├── segmentation.py
│   └── biomass.py
├── public/               # Static frontend — Vercel serves this via CDN automatically
│   ├── index.html
│   ├── app.js
│   └── vendor/           # Vendored Leaflet + Leaflet.draw (no external CDN)
├── requirements.txt
├── vercel.json
└── .python-version       # pinned to 3.12 (scipy/scikit-image need prebuilt wheels)
```

## Option A — Deploy via Vercel CLI (fastest)

```bash
npm i -g vercel        # one-time install
cd sylvasense_vercel
vercel                 # first run: creates the project, asks a few setup questions
```

Answer the prompts (link to your Vercel account, project name, etc. — defaults are fine). Vercel detects FastAPI automatically from `requirements.txt` + `api/main.py`, no build command needed.

When it finishes, you'll get a preview URL like `https://sylvasense-xyz.vercel.app`. Open it — the dashboard should load and work immediately (in synthetic-imagery mode, since no Sentinel Hub credentials are set yet).

To promote to a stable production URL:
```bash
vercel --prod
```

## Option B — Deploy via GitHub (good for ongoing updates)

1. Push this folder to a new GitHub repository.
2. Go to [vercel.com/new](https://vercel.com/new), import that repository.
3. Vercel auto-detects FastAPI + static frontend — leave build settings as default, click **Deploy**.
4. Every future `git push` automatically redeploys.

## Adding real Sentinel Hub credentials

Same environment variables as local dev, just set them in Vercel's dashboard instead of your terminal:

1. Go to your project on vercel.com → **Settings** → **Environment Variables**.
2. Add:
   - `SENTINEL_HUB_CLIENT_ID` = your client ID
   - `SENTINEL_HUB_CLIENT_SECRET` = your client secret
   - `USE_REAL_IMAGERY` = `1`
3. Redeploy (Vercel → Deployments → ⋯ → Redeploy) so the new env vars take effect.

## Things worth knowing

- **Function duration**: `vercel.json` sets `maxDuration: 30` (seconds) for the API. Local testing showed the pipeline finishing in ~7s for a 1024×1024 synthetic tile — 30s gives comfortable headroom for cold starts and real-imagery fetches. Vercel's Hobby plan supports up to 60s with Fluid Compute (which FastAPI deployments use by default), so this fits without needing a paid plan.
- **Cold starts**: the first request after a period of inactivity will be slower (loading scipy/scikit-image takes a moment). Subsequent requests are fast.
- **No GDAL/rasterio**: this project deliberately has zero native-binary dependencies (see main README), which also makes it much lighter and more reliable to deploy as a serverless function — no risk of missing system libraries on Vercel's build image.
- **Bundle size**: Python function bundles on Vercel have a 500MB uncompressed limit. This project's dependencies (scipy, scikit-image, shapely, etc.) are comfortably under that.
- **Same-origin API calls**: `public/app.js` auto-detects whether it's running on `localhost` (uses `http://localhost:8000`) or deployed anywhere else (uses relative `/api/...` paths, same-origin). No manual editing needed either way.
