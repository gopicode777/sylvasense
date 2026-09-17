"""
segmentation.py — Canopy detection, instance separation, and vectorization

Pipeline:
  1. Compute NDVI from optical bands (or SAR backscatter index where clouded)
  2. Threshold + clean to get a binary canopy mask
  3. Distance-transform + watershed to split touching crowns into instances
  4. Vectorize each instance into a GeoJSON polygon with per-tree stats

This intentionally avoids requiring a trained deep model to run — DeepForest /
YOLOv8-OBB fine-tuning is documented as a drop-in upgrade in README.md, but
watershed-on-NDVI is a real, published, defensible technique (used
operationally in forestry / precision-ag canopy counting) and it runs
instantly on CPU, which matters for a live demo.

NO GDAL/RASTERIO DEPENDENCY: vectorization uses skimage.measure.find_contours
+ shapely instead of rasterio.features.shapes, so the whole project installs
cleanly everywhere — including Windows machines where GDAL's native DLLs get
blocked by antivirus/Application Control policies.
"""

import math
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, opening, disk
from skimage.measure import find_contours
from shapely.geometry import Polygon, mapping
import geojson

from imagery import RasterStack


def compute_ndvi(stack: RasterStack) -> np.ndarray:
    red, nir = stack.red, stack.nir
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def compute_sar_veg_index(stack: RasterStack) -> np.ndarray:
    """Normalized VV index as a cloud-independent vegetation density proxy."""
    vv = stack.sar_vv
    lo, hi = np.percentile(vv, 2), np.percentile(vv, 98)
    idx = (vv - lo) / max(hi - lo, 1e-6)
    return np.clip(idx, 0, 1)


def build_canopy_mask(stack: RasterStack, ndvi_threshold: float = 0.35):
    """
    Fuses optical NDVI with SAR where clouds obscure the optical signal.
    Returns (binary_mask, source_map) where source_map records, per-pixel,
    whether the estimate came from 'optical' or 'sar' — this feeds the
    dashboard's "estimated via SAR" badge.
    """
    ndvi = compute_ndvi(stack)
    sar_idx = compute_sar_veg_index(stack)
    cloud = stack.cloud_mask.astype(bool)

    canopy_optical = ndvi > ndvi_threshold
    canopy_sar = sar_idx > 0.55

    canopy = np.where(cloud, canopy_sar, canopy_optical)
    source_map = np.where(cloud, "sar", "optical")

    # Morphological cleanup: drop single-pixel noise, close small holes
    canopy_clean = opening(canopy, disk(1))
    canopy_clean = remove_small_objects(canopy_clean, min_size=6)

    return canopy_clean, source_map, ndvi, sar_idx, cloud


def instance_segment(canopy_mask: np.ndarray, min_distance: int = 4):
    """Watershed instance separation: distance transform + local maxima as seeds."""
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
    """Plain linear pixel->lon/lat mapping (row 0 = top = maxy, like a
    north-up raster). Replaces rasterio's Affine transform with dependency-free
    arithmetic — exact for the axis-aligned bbox rasters this pipeline uses."""
    minx, miny, maxx, maxy = bbox
    lon = minx + (col / width) * (maxx - minx)
    lat = maxy - (row / height) * (maxy - miny)
    return lon, lat


def vectorize_instances(labels: np.ndarray, bbox, width: int, height: int,
                         ndvi: np.ndarray, source_map: np.ndarray,
                         pixel_size_m: float):
    """Turns labeled instance raster into a GeoJSON FeatureCollection,
    one polygon per detected tree crown, with per-crown attributes.

    Uses skimage.measure.find_contours (marching squares) to trace each
    instance's boundary, then maps pixel coordinates to lon/lat directly —
    no GDAL/rasterio involved anywhere in this path.
    """
    features = []
    px_area_m2 = pixel_size_m ** 2
    max_label = int(labels.max()) if labels.size else 0

    for val in range(1, max_label + 1):
        mask = labels == val
        n_px = int(mask.sum())
        if n_px < 3:
            continue

        # Trace the instance boundary at the 0.5 level of its binary mask.
        # Pad by 1px so crowns touching the raster edge still get a closed contour.
        padded = np.pad(mask.astype(float), 1, mode="constant")
        contours = find_contours(padded, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)  # exterior boundary = longest ring
        contour = contour - 1  # undo the padding offset

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
