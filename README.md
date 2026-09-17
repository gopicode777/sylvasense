# SylvaSense — Automated Tree Enumeration & AGB Estimation

> **Deploying this to Vercel?** This folder's directories are named `api/` and `public/` (Vercel's required layout) instead of `backend/` and `frontend/` used in the general instructions below — see **DEPLOY.md** in this same folder for Vercel-specific steps. Everywhere below that says `backend/` read it as `api/`, and `frontend/` as `public/`; the file contents and commands are otherwise identical.

Submission for ORION-PS-03. Full pipeline: multi-spectral optical + SAR fusion → canopy instance segmentation → GeoJSON export → biomass/carbon regression → interactive map dashboard → live inference.

## What's real vs. simulated in this build

**Real and fully functional, either way:**
- Canopy segmentation pipeline (NDVI thresholding, SAR cloud-fallback, watershed instance separation)
- GeoJSON vectorization with per-tree attributes (pure Python/scikit-image — **no GDAL/rasterio dependency**, so it installs cleanly on locked-down Windows machines where GDAL's native DLLs commonly get blocked by antivirus/Application Control policies)
- AGB/carbon/CO2e regression using a published, cited allometric equation (Jucker et al. 2017) + IPCC (2006) carbon fraction
- Confidence scoring, fully inspectable
- FastAPI backend, tested end-to-end
- Interactive dashboard with live layer toggles, polygon drawing, and GeoJSON export
- **Real Sentinel Hub Process API integration** for Sentinel-2 L2A + Sentinel-1 GRD, with OAuth token handling and GeoTIFF decoding — active the moment you set `USE_REAL_IMAGERY=1` and your credentials (see "Going live" below)

**Simulated by default (until you add credentials):**
- Without `SENTINEL_HUB_CLIENT_ID`/`SECRET` set, `backend/imagery.py` falls back to a procedurally realistic synthetic Sentinel-2 + Sentinel-1 raster (forest canopy pattern, cloud gaps, SAR speckle), so the whole pipeline is demoable with zero setup and zero API keys.

This means: **today, with no setup**, you can demo the entire pipeline live, end-to-end, on any coordinates using synthetic imagery. **The moment you add a free Sentinel Hub API key**, the exact same code runs on real satellite data.

## Project structure

```
sylvasense/
├── backend/
│   ├── main.py          # FastAPI app: /api/infer, /api/layer-preview
│   ├── imagery.py        # Imagery fetch (synthetic now, Sentinel Hub-ready)
│   ├── segmentation.py    # NDVI/SAR fusion, watershed segmentation, GeoJSON export
│   └── biomass.py         # Allometric AGB/carbon/CO2e + confidence scoring
├── frontend/
│   ├── index.html          # Dashboard UI (map, layer toggles, results panel)
│   └── app.js               # Map logic, API calls, rendering
├── validation/
│   ├── validate.py           # Error-metric harness (MAE/RMSE vs reference plots)
│   └── REPORT.md               # Full math writeup, citations, limitations
└── README.md
```

## Running it

### 1. Backend

```bash
cd backend
pip install -r ../requirements.txt
python -m uvicorn main:app --reload --port 8000
```

(Use `python -m uvicorn ...` rather than bare `uvicorn ...` — it works even if uvicorn's script isn't on your PATH, which is common on Windows.)

Confirm it's up: `curl http://localhost:8000/api/health` → `{"status":"ok"}`

**No GDAL/rasterio to worry about** — this project deliberately avoids that dependency (see note below) so `pip install` is fast and doesn't hit native-DLL issues.

### 2. Frontend

```bash
cd frontend
python serve.py 8080
```

(This uses the included `serve.py` instead of plain `python -m http.server` — it disables browser caching entirely, which avoids a confusing issue where your browser keeps showing an old cached copy of `app.js`/`index.html` even after you've updated the files on disk. If you'd rather use the plain server, that works too, but do a hard refresh — Ctrl+Shift+R — every time you change frontend files.)

Open `http://localhost:8080` in a browser. Click **Load Sample** for a known-good demo AOI, or **Draw AOI** to draw your own polygon (keep it to roughly 500m–1km across for realistic per-tree resolution — see the resolution note in `validation/REPORT.md` §7.4). Then click **Run Live Inference**.

**No CDN dependency**: Leaflet and Leaflet.draw are vendored directly into `frontend/vendor/` — the map and drawing tools work fully offline / behind restrictive firewalls, with zero external JS dependencies (only Google Fonts loads externally, and the app works fine even if that's blocked too — text just falls back to system fonts).

If your frontend is served from a different host/port than the backend, set `window.SYLVASENSE_API_BASE` in a `<script>` tag before `app.js` loads in `index.html`.

### 3. Validation

```bash
cd validation
python3 validate.py
```

Prints per-plot predicted vs. reference tree counts and AGB, plus aggregate MAE/RMSE. **Replace the placeholder `REFERENCE_PLOTS` values with real field/LiDAR ground truth before citing these numbers as real accuracy** — see the disclosure in `REPORT.md` §6.

## Going live with real satellite data

1. Get free credentials at [Sentinel Hub](https://www.sentinel-hub.com/) or [Copernicus Dataspace Ecosystem](https://dataspace.copernicus.eu/) (Dashboard → User Settings → OAuth clients → create a new client → copy the Client ID and Client Secret).

2. In the **same terminal** you'll run uvicorn from, set three environment variables before starting the server:

   **Windows PowerShell:**
   ```powershell
   $env:SENTINEL_HUB_CLIENT_ID="your_client_id"
   $env:SENTINEL_HUB_CLIENT_SECRET="your_client_secret"
   $env:USE_REAL_IMAGERY="1"
   python -m uvicorn main:app --reload --port 8000
   ```

   **macOS/Linux:**
   ```bash
   export SENTINEL_HUB_CLIENT_ID="your_client_id"
   export SENTINEL_HUB_CLIENT_SECRET="your_client_secret"
   export USE_REAL_IMAGERY=1
   python -m uvicorn main:app --reload --port 8000
   ```

3. That's it — `backend/imagery.py` now calls the real Sentinel Hub Process API for Sentinel-2 L2A (optical) and Sentinel-1 GRD (SAR), decodes the returned GeoTIFF with `tifffile` (no GDAL needed), and feeds it into the exact same segmentation/biomass pipeline. Nothing else in the project changes.

**Notes on going live:**
- If your drawn AOI has no cloud-free Sentinel-2 pass in the last 60 days, the optical fetch may return mostly cloud — the pipeline will automatically lean on the SAR fallback, and you'll see the "estimated via SAR" badges in the dashboard. You can widen the search window by editing `days_back` in `imagery.py`'s `_default_time_range()`.
- The first request after starting the server will be slightly slower (OAuth token fetch); subsequent requests reuse the cached token until it expires.
- If you see a `Sentinel Hub Process API error`, the error message includes the actual response body — it's almost always either an auth issue (double-check the client ID/secret) or an AOI with no available imagery.
- Real Sentinel-2/Sentinel-1 native resolution is 10m/pixel — see the resolution note in `backend/imagery.py` and `validation/REPORT.md` §7.4 for what that means for per-tree crown accuracy.

## Round 1 submission checklist mapping

| Requirement | Where it lives |
|---|---|
| Canopy segmentation pipeline + GeoJSON export | `backend/segmentation.py`, exported live via `/api/infer` and the dashboard's "Export GeoJSON" button |
| Interactive map dashboard with layer toggles | `frontend/index.html` + `app.js` — RGB / NDVI / SAR / canopy mask / detected crowns |
| Biomass estimation formulation & validation report | `validation/REPORT.md` |
| Live raster inference demo on target coordinates | `POST /api/infer` — draw any polygon in the dashboard and run it live |

## Known limitations (also documented in REPORT.md)

- Imagery is synthetic pending real Sentinel Hub/GEE credentials (see above)
- Watershed segmentation undercounts in very densely closed canopy (>85% cover)
- Allometric coefficients are pantropical-generalized, not biome-specific
- No real LiDAR height data ingested yet — height is proxied via crown diameter only

## Why no GDAL/rasterio

Early versions of this project used `rasterio` for coordinate transforms and polygon vectorization. It was removed because:
1. `rasterio` bundles native GDAL binaries that are a common source of install pain on Windows (missing DLLs, antivirus/Application Control policies blocking them outright, architecture mismatches).
2. This pipeline only ever needs axis-aligned bbox↔pixel math and raster-to-polygon tracing — both are fully expressible in pure Python/NumPy + scikit-image (`skimage.measure.find_contours`), with no loss of correctness for this use case.

If you later ingest real GeoTIFFs from Sentinel Hub and need arbitrary CRS reprojection, `rasterio` (or `pyproj` alone, which is lighter-weight) is the right tool to reach for at that point — it just isn't needed for the bbox-based flow this project uses today.
