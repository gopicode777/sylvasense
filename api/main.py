"""
main.py — FastAPI inference backend for SylvaSense (Vercel-safe single-file build)

Endpoints:
  POST /api/infer        -> full pipeline: fetch imagery -> segment -> AGB -> GeoJSON + stats
  POST /api/layer-preview -> base64 PNG previews of RGB / NDVI / SAR / canopy-mask layers
  GET  /api/health       -> liveness check

NOTE ON FILE LAYOUT: this file intentionally inlines the logic that used to
live in imagery.py / segmentation.py / biomass.py. Vercel's Python function
runtime treats each file under api/ as its own isolated function and does
not reliably bundle sibling module imports (e.g. `from imagery import ...`)
even with `includeFiles` in vercel.json — that caused
`ModuleNotFoundError: No module named 'imagery'` in production. Keeping
everything the deployed endpoint needs in this one file avoids that failure
mode entirely. The original imagery.py / segmentation.py / biomass.py files
are left in this folder for readability/reference, but main.py no longer
depends on them at runtime.

Run locally:
  uvicorn main:app --reload --port 8000
"""

import io
import os
import time
import math
import base64
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, opening, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon, mapping
import geojson


# ============================================================================
# IMAGERY — satellite imagery acquisition layer (was imagery.py)
# ============================================================================

USE_REAL_IMAGERY = os.environ.get("USE_REAL_IMAGERY", "0") == "1"


@dataclass
class RasterStack:
    red: np.ndarray
    nir: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    sar_vv: np.ndarray
    sar_vh: np.ndarray
    cloud_mask: np.ndarray
    crs: str
    bounds: tuple
    width: int
    height: int


def _perlin_like(shape, scale=8.0, seed=0, octaves=3):
    rng = np.random.default_rng(seed)
    h, w = shape
    total = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    freq = scale
    for o in range(octaves):
        gh, gw = max(2, int(h / freq)), max(2, int(w / freq))
        grid = rng.random((gh, gw)).astype(np.float32)
        up = np.kron(grid, np.ones((int(np.ceil(h / gh)), int(np.ceil(w / gw)))))
        up = up[:h, :w]
        total += amplitude * up
        amplitude *= 0.5
        freq /= 2.0
    total -= total.min()
    total /= (total.max() + 1e-9)
    return total


def _bbox_to_shape(bbox, target_px=512):
    return (target_px, target_px)


def _synth_stack(bbox, seed, target_px=512, density_scale=1.0) -> RasterStack:
    minx, miny, maxx, maxy = bbox
    h, w = _bbox_to_shape(bbox, target_px)

    canopy_density = _perlin_like((h, w), scale=w / 14, seed=seed, octaves=4)
    canopy_density = np.clip(canopy_density * 1.15 * density_scale, 0, 1)

    rng = np.random.default_rng(seed + 1)
    crown_field = np.zeros((h, w), dtype=np.float32)
    n_crowns = int((h * w) / 900 * (0.6 + 0.8 * canopy_density.mean()) * density_scale)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_crowns):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        if canopy_density[cy, cx] < 0.35:
            continue
        r = rng.integers(4, 11)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        crown_field[mask] = np.maximum(crown_field[mask], rng.uniform(0.75, 1.0))

    veg = np.clip(0.35 * canopy_density + 0.65 * crown_field, 0, 1)

    nir = 0.15 + 0.65 * veg + rng.normal(0, 0.015, (h, w))
    red = 0.35 - 0.22 * veg + rng.normal(0, 0.01, (h, w))
    green = 0.12 + 0.10 * veg + rng.normal(0, 0.01, (h, w))
    blue = 0.08 + 0.04 * veg + rng.normal(0, 0.008, (h, w))
    for band in (nir, red, green, blue):
        np.clip(band, 0, 1, out=band)

    sar_vv = -12 + 9 * np.tanh(2.2 * veg) + rng.normal(0, 0.6, (h, w))
    sar_vh = -18 + 7 * np.tanh(2.0 * veg) + rng.normal(0, 0.6, (h, w))

    cloud_mask = np.zeros((h, w), dtype=np.uint8)
    if rng.random() < 0.5:
        ccy, ccx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
        cr = rng.integers(int(h * 0.12), int(h * 0.22))
        cmask = (yy - ccy) ** 2 + (xx - ccx) ** 2 <= cr * cr
        cloud_mask[cmask] = 1
        for band in (nir, red, green, blue):
            band[cmask] = np.clip(band[cmask] + rng.uniform(0.35, 0.55), 0, 1)

    return RasterStack(
        red=red.astype(np.float32), nir=nir.astype(np.float32),
        green=green.astype(np.float32), blue=blue.astype(np.float32),
        sar_vv=sar_vv.astype(np.float32), sar_vh=sar_vh.astype(np.float32),
        cloud_mask=cloud_mask, crs="EPSG:4326",
        bounds=bbox, width=w, height=h,
    )


def _bbox_seed(bbox):
    return abs(hash(tuple(round(v, 4) for v in bbox))) % (2 ** 31)


SH_TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
SH_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

_token_cache = {"access_token": None, "expires_at": 0.0}

_S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "SCL"], units: "REFLECTANCE" }],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.SCL];
}
"""

_S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""


def _get_access_token():
    client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "USE_REAL_IMAGERY=1 but SENTINEL_HUB_CLIENT_ID / SENTINEL_HUB_CLIENT_SECRET "
            "are not set. Get free credentials at https://www.sentinel-hub.com/ or "
            "https://dataspace.copernicus.eu/ (Dashboard -> User Settings -> OAuth clients), "
            "then set them as environment variables."
        )

    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    import requests

    resp = requests.post(
        SH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Sentinel Hub OAuth token request failed ({resp.status_code}): {resp.text[:300]}\n"
            "Double-check SENTINEL_HUB_CLIENT_ID/SECRET are correct and active."
        )
    payload = resp.json()
    _token_cache["access_token"] = payload["access_token"]
    _token_cache["expires_at"] = now + payload.get("expires_in", 3600)
    return _token_cache["access_token"]


def _default_time_range(days_back=60):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return {"from": start.strftime(fmt), "to": now.strftime(fmt)}


def _process_api_request(evalscript, bbox, width, height, data_type,
                          token, extra_data_filter=None, processing=None):
    import requests

    data_filter = {"timeRange": _default_time_range(), "mosaickingOrder": "leastCC"}
    if extra_data_filter:
        data_filter.update(extra_data_filter)

    data_entry = {"type": data_type, "dataFilter": data_filter}
    if processing:
        data_entry["processing"] = processing

    body = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [data_entry],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    resp = requests.post(
        SH_PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Sentinel Hub Process API error {resp.status_code} for {data_type}: {resp.text[:500]}"
        )
    return resp.content


def _decode_multiband_tiff(tiff_bytes, expected_bands):
    import tifffile
    arr = tifffile.imread(io.BytesIO(tiff_bytes))
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[-1] == expected_bands:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.ndim == 3 and arr.shape[0] == expected_bands:
        pass
    else:
        raise ValueError(
            f"Unexpected TIFF shape {arr.shape}, expected {expected_bands} bands. "
            "Sentinel Hub may have returned an error image instead of data — "
            "check that your AOI has recent cloud-free coverage."
        )
    return arr


def _fetch_real_stack(bbox, width, height) -> RasterStack:
    token = _get_access_token()

    s2_bytes = _process_api_request(
        _S2_EVALSCRIPT, bbox, width, height, "sentinel-2-l2a", token,
    )
    s2 = _decode_multiband_tiff(s2_bytes, expected_bands=5)
    blue, green, red, nir, scl = s2[0], s2[1], s2[2], s2[3], s2[4]

    cloud_mask = np.isin(np.round(scl).astype(int), [3, 8, 9, 10]).astype(np.uint8)

    s1_bytes = _process_api_request(
        _S1_EVALSCRIPT, bbox, width, height, "sentinel-1-grd", token,
        extra_data_filter={"resolution": "HIGH"},
        processing={"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True},
    )
    s1 = _decode_multiband_tiff(s1_bytes, expected_bands=2)
    vv_linear, vh_linear = s1[0], s1[1]
    sar_vv = 10 * np.log10(np.clip(vv_linear, 1e-6, None))
    sar_vh = 10 * np.log10(np.clip(vh_linear, 1e-6, None))

    return RasterStack(
        red=red.astype(np.float32), nir=nir.astype(np.float32),
        green=green.astype(np.float32), blue=blue.astype(np.float32),
        sar_vv=sar_vv.astype(np.float32), sar_vh=sar_vh.astype(np.float32),
        cloud_mask=cloud_mask, crs="EPSG:4326",
        bounds=bbox, width=width, height=height,
    )


def fetch_optical(bbox, target_px=1024, density_scale=1.0) -> RasterStack:
    if USE_REAL_IMAGERY:
        h, w = _bbox_to_shape(bbox, target_px)
        return _fetch_real_stack(bbox, w, h)
    return _synth_stack(bbox, seed=_bbox_seed(bbox), target_px=target_px, density_scale=density_scale)


def fetch_sar(bbox, target_px=1024, density_scale=1.0) -> RasterStack:
    if USE_REAL_IMAGERY:
        h, w = _bbox_to_shape(bbox, target_px)
        return _fetch_real_stack(bbox, w, h)
    return _synth_stack(bbox, seed=_bbox_seed(bbox), target_px=target_px, density_scale=density_scale)


# ============================================================================
# SEGMENTATION — canopy detection, instance separation, vectorization
# (was segmentation.py)
# ============================================================================

def compute_ndvi(stack: RasterStack) -> np.ndarray:
    red, nir = stack.red, stack.nir
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def compute_sar_veg_index(stack: RasterStack) -> np.ndarray:
    vv = stack.sar_vv
    lo, hi = np.percentile(vv, 2), np.percentile(vv, 98)
    idx = (vv - lo) / max(hi - lo, 1e-6)
    return np.clip(idx, 0, 1)


def build_canopy_mask(stack: RasterStack, ndvi_threshold: float = 0.35):
    ndvi = compute_ndvi(stack)
    sar_idx = compute_sar_veg_index(stack)
    cloud = stack.cloud_mask.astype(bool)

    canopy_optical = ndvi > ndvi_threshold
    canopy_sar = sar_idx > 0.55

    canopy = np.where(cloud, canopy_sar, canopy_optical)
    source_map = np.where(cloud, "sar", "optical")

    canopy_clean = opening(canopy, disk(1))
    canopy_clean = remove_small_objects(canopy_clean, min_size=6)

    return canopy_clean, source_map, ndvi, sar_idx, cloud


def instance_segment(canopy_mask: np.ndarray, min_distance: int = 4):
    distance = ndi.distance_transform_edt(canopy_mask)
    coords = peak_local_max(
        distance, min_distance=min_distance, labels=canopy_mask,
        exclude_border=False,
    )
    peak_mask = np.zeros_like(distance, dtype=bool)
    if len(coords):
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    labels = watershed(-distance, markers, mask=canopy_mask)
    return labels, distance


def _pixel_to_lonlat(row: float, col: float, bbox, width: int, height: int):
    minx, miny, maxx, maxy = bbox
    lon = minx + (col / width) * (maxx - minx)
    lat = maxy - (row / height) * (maxy - miny)
    return lon, lat


def vectorize_instances(labels: np.ndarray, bbox, width: int, height: int,
                         ndvi: np.ndarray, source_map: np.ndarray,
                         pixel_size_m: float):
    features = []
    px_area_m2 = pixel_size_m ** 2
    max_label = int(labels.max()) if labels.size else 0

    for val in range(1, max_label + 1):
        mask = labels == val
        n_px = int(mask.sum())
        if n_px < 3:
            continue

        padded = np.pad(mask.astype(float), 1, mode="constant")
        contours = find_contours(padded, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        contour = contour - 1

        coords_ll = [_pixel_to_lonlat(r, c, bbox, width, height) for r, c in contour]
        if len(coords_ll) < 4:
            continue

        poly = Polygon(coords_ll)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area == 0:
            continue

        ys, xs = np.where(mask)
        mean_ndvi = float(np.mean(ndvi[ys, xs]))
        area_m2 = n_px * px_area_m2
        crown_diameter_m = 2 * math.sqrt(area_m2 / math.pi)

        sources_here = source_map[ys, xs]
        is_sar = bool(np.mean(sources_here == "sar") > 0.5)

        features.append(geojson.Feature(
            geometry=mapping(poly),
            properties={
                "tree_id": val,
                "area_m2": round(area_m2, 2),
                "crown_diameter_m": round(float(crown_diameter_m), 2),
                "mean_ndvi": round(mean_ndvi, 3),
                "estimation_source": "sar" if is_sar else "optical",
            },
        ))

    return geojson.FeatureCollection(features)


# ============================================================================
# BIOMASS — Aboveground Biomass (AGB) & Carbon estimation (was biomass.py)
# ============================================================================

ALLOMETRIC_A = 0.567
ALLOMETRIC_B = 2.393

CARBON_FRACTION = 0.47
CO2_CONVERSION = 3.667

SAR_SATURATION_WARNING_KG = 450.0


def agb_from_crown_diameter(diameter_m: float) -> float:
    if diameter_m <= 0:
        return 0.0
    return ALLOMETRIC_A * (diameter_m ** ALLOMETRIC_B)


def carbon_metrics(agb_kg: float) -> dict:
    carbon_kg = agb_kg * CARBON_FRACTION
    co2e_kg = carbon_kg * CO2_CONVERSION
    return {
        "agb_kg": round(agb_kg, 2),
        "carbon_kg": round(carbon_kg, 2),
        "co2e_kg": round(co2e_kg, 2),
    }


def confidence_score(mean_ndvi: float, is_sar: bool, agb_kg: float, cloud_frac: float) -> float:
    score = 0.5 + 0.4 * max(0.0, min(mean_ndvi, 1.0))
    if is_sar:
        score -= 0.15
        if agb_kg > SAR_SATURATION_WARNING_KG:
            score -= 0.15
    score -= 0.2 * cloud_frac
    return round(max(0.05, min(score, 0.98)), 2)


def summarize_polygon(features: list, cloud_frac: float) -> dict:
    total_agb = 0.0
    total_trees = len(features)
    confidences = []

    for f in features:
        props = f["properties"]
        d = props["crown_diameter_m"]
        agb = agb_from_crown_diameter(d)
        metrics = carbon_metrics(agb)
        is_sar = props["estimation_source"] == "sar"
        conf = confidence_score(props["mean_ndvi"], is_sar, agb, cloud_frac)

        props.update(metrics)
        props["confidence"] = conf
        confidences.append(conf)
        total_agb += agb

    total_metrics = carbon_metrics(total_agb)
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "tree_count": total_trees,
        "total_agb_kg": total_metrics["agb_kg"],
        "total_carbon_kg": total_metrics["carbon_kg"],
        "total_co2e_kg": total_metrics["co2e_kg"],
        "avg_confidence": avg_confidence,
        "cloud_fraction": round(cloud_frac, 3),
    }


# ============================================================================
# FASTAPI APP (was main.py)
# ============================================================================

app = FastAPI(title="SylvaSense Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolygonRequest(BaseModel):
    coordinates: List[Tuple[float, float]]

    @field_validator("coordinates")
    @classmethod
    def must_have_enough_points(cls, v):
        if len(v) < 3:
            raise ValueError("Polygon needs at least 3 points")
        return v


def _bbox_from_coords(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _approx_pixel_size_m(bbox, width_px):
    minx, miny, maxx, maxy = bbox
    mean_lat = (miny + maxy) / 2
    deg_lon_m = 111320 * np.cos(np.radians(mean_lat))
    width_deg = maxx - minx
    width_m = max(width_deg * deg_lon_m, 1.0)
    return width_m / width_px


PREVIEW_MAX_PX = 480


def _array_to_png_b64(arr: np.ndarray) -> str:
    arr = np.nan_to_num(arr)
    arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-9)).astype(np.uint8)
    img = Image.fromarray(arr)
    img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rgb_to_png_b64(r, g, b) -> str:
    def norm(x):
        x = np.nan_to_num(x)
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        return (x * 255).astype(np.uint8)
    rgb = np.dstack([norm(r), norm(g), norm(b)])
    img = Image.fromarray(rgb, mode="RGB")
    img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/infer")
def infer(req: PolygonRequest):
    t0 = time.time()
    bbox = _bbox_from_coords(req.coordinates)

    try:
        stack = fetch_optical(bbox)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Imagery fetch failed: {e}")

    pixel_size_m = _approx_pixel_size_m(bbox, stack.width)

    canopy_mask, source_map, ndvi, sar_idx, cloud = build_canopy_mask(stack)
    cloud_frac = float(cloud.mean())

    labels, distance = instance_segment(canopy_mask)
    fc = vectorize_instances(labels, stack.bounds, stack.width, stack.height, ndvi, source_map, pixel_size_m)

    summary = summarize_polygon(fc["features"], cloud_frac)

    elapsed = round(time.time() - t0, 3)

    return {
        "bbox": bbox,
        "geojson": fc,
        "summary": summary,
        "processing_seconds": elapsed,
        "pixel_size_m": round(pixel_size_m, 2),
        "raster_shape": [stack.height, stack.width],
    }


@app.post("/api/layer-preview")
def layer_preview(req: PolygonRequest):
    bbox = _bbox_from_coords(req.coordinates)
    stack = fetch_optical(bbox)
    canopy_mask, source_map, ndvi, sar_idx, cloud = build_canopy_mask(stack)

    return {
        "rgb": _rgb_to_png_b64(stack.red, stack.green, stack.blue),
        "ndvi": _array_to_png_b64(ndvi),
        "sar_vv": _array_to_png_b64(stack.sar_vv),
        "canopy_mask": _array_to_png_b64(canopy_mask.astype(float)),
        "cloud_fraction": float(cloud.mean()),
        "bbox": bbox,
    }
