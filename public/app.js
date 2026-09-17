// SylvaSense dashboard logic — vanilla JS + Leaflet, no build step required.

// Auto-detects deployment: on Vercel (or any deployment where frontend and
// backend share one domain), API calls go to the same origin with no CORS
// needed. For local dev (frontend on :8080, backend on :8000), it points at
// localhost:8000 instead. Override anytime by setting
// window.SYLVASENSE_API_BASE in a <script> tag before this file loads.
const API_BASE = window.SYLVASENSE_API_BASE ?? (
  (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : ''  // same-origin — works when deployed on Vercel with api/ + public/
);

// ---------- Map setup ----------
const map = L.map('map', { zoomControl: false, attributionControl: true }).setView([11.3136, 76.9636], 14);
L.control.zoom({ position: 'bottomright' }).addTo(map);

// Plain OpenStreetMap tiles (no API key needed, ever) with a CSS dark-mode
// filter applied via .leaflet-tile-pane in the stylesheet — avoids relying
// on hosted "dark" basemap styles (e.g. CARTO) that have started requiring
// API keys / accounts, which would otherwise silently break the map for
// anyone without a key.
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  subdomains: 'abc',
  maxZoom: 19,
}).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

// Leaflet.draw is loaded from a CDN and can occasionally fail (network
// filtering, ad-blockers, offline CDN). Guard its setup so the rest of the
// app — sample AOI, layer toggles, inference, export — still works even if
// free-drawing a polygon isn't available.
let drawingAvailable = typeof L.Control.Draw === 'function';
let drawControl = null;

if (drawingAvailable) {
  try {
    drawControl = new L.Control.Draw({
      draw: {
        polygon: {
          shapeOptions: { color: '#3fb87a', weight: 2, fillOpacity: 0.08 },
          allowIntersection: false,
          showArea: true,
        },
        polyline: false, rectangle: false, circle: false, marker: false, circlemarker: false,
      },
      edit: { featureGroup: drawnItems, remove: true },
    });
  } catch (e) {
    console.error('Leaflet.draw failed to initialize — freehand AOI drawing disabled. Sample AOI still works.', e);
    drawingAvailable = false;
  }
} else {
  console.warn('Leaflet.draw did not load (CDN blocked?) — freehand AOI drawing disabled. Sample AOI still works.');
}

let currentPolygonLatLngs = null;
let currentResult = null;
let overlayLayers = { rgb: null, ndvi: null, sar: null, mask: null, crowns: null };

// ---------- UI wiring ----------
const els = {
  btnDraw: document.getElementById('btn-draw'),
  btnSample: document.getElementById('btn-sample'),
  btnInfer: document.getElementById('btn-infer'),
  btnExport: document.getElementById('btn-export'),
  status: document.getElementById('status'),
  statusText: document.getElementById('status-text'),
  loadingOverlay: document.getElementById('loading-overlay'),
  loadingText: document.getElementById('loading-text'),
  emptyState: document.getElementById('empty-state'),
  resultsContent: document.getElementById('results-content'),
  toggles: {
    rgb: document.getElementById('toggle-rgb'),
    ndvi: document.getElementById('toggle-ndvi'),
    sar: document.getElementById('toggle-sar'),
    mask: document.getElementById('toggle-mask'),
    crowns: document.getElementById('toggle-crowns'),
  },
};

function setStatus(mode, text) {
  els.status.className = mode;
  els.statusText.textContent = text;
}

function showLoading(text) {
  els.loadingText.textContent = text;
  els.loadingOverlay.classList.add('show');
}
function hideLoading() {
  els.loadingOverlay.classList.remove('show');
}

els.btnDraw.addEventListener('click', () => {
  if (!drawingAvailable) {
    setStatus('error', 'drawing tool unavailable — use Load Sample instead');
    return;
  }
  drawnItems.clearLayers();
  currentPolygonLatLngs = null;
  els.btnInfer.disabled = true;
  new L.Draw.Polygon(map, drawControl.options.draw.polygon).enable();
});

if (drawingAvailable && typeof L.Draw !== 'undefined') {
  map.on(L.Draw.Event.CREATED, (e) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    currentPolygonLatLngs = e.layer.getLatLngs()[0];
    els.btnInfer.disabled = false;
    setStatus('idle', 'aoi ready');
  });
}

// Sample AOI near Coimbatore / Nilgiri foothills — known-good demo coordinates
const SAMPLE_AOI = [
  [11.310, 76.960], [11.310, 76.967], [11.317, 76.967], [11.317, 76.960],
];

if (!drawingAvailable) {
  els.btnDraw.disabled = true;
  els.btnDraw.title = 'Drawing tool failed to load (CDN blocked?) — use Load Sample instead';
}

els.btnSample.addEventListener('click', () => {
  drawnItems.clearLayers();
  const latlngs = SAMPLE_AOI.map(([lat, lng]) => L.latLng(lat, lng));
  const poly = L.polygon(latlngs, { color: '#3fb87a', weight: 2, fillOpacity: 0.08 });
  drawnItems.addLayer(poly);
  currentPolygonLatLngs = latlngs;
  map.fitBounds(poly.getBounds(), { padding: [40, 40] });
  els.btnInfer.disabled = false;
  setStatus('idle', 'sample aoi loaded');
});

function polygonToCoords(latlngs) {
  // API expects [lon, lat] pairs, closed ring
  const coords = latlngs.map(ll => [ll.lng, ll.lat]);
  coords.push(coords[0]);
  return coords;
}

function boundsFromCoords(coords) {
  const lons = coords.map(c => c[0]);
  const lats = coords.map(c => c[1]);
  return [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]];
}

async function runInference() {
  if (!currentPolygonLatLngs) return;
  const coords = polygonToCoords(currentPolygonLatLngs);

  els.btnInfer.disabled = true;
  setStatus('busy', 'fetching imagery');
  showLoading('FETCHING SENTINEL IMAGERY…');

  try {
    const [previewRes, inferRes] = await Promise.all([
      fetch(`${API_BASE}/api/layer-preview`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates: coords }),
      }),
      (async () => {
        showLoading('RUNNING CANOPY SEGMENTATION…');
        return fetch(`${API_BASE}/api/infer`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ coordinates: coords }),
        });
      })(),
    ]);

    if (!previewRes.ok || !inferRes.ok) throw new Error('Backend returned an error status');

    const preview = await previewRes.json();
    const result = await inferRes.json();
    currentResult = result;

    renderLayers(preview, coords);
    renderCrowns(result.geojson);
    renderResults(result);

    setStatus('live', `done in ${result.processing_seconds}s`);
  } catch (err) {
    console.error(err);
    setStatus('error', 'backend unreachable — is uvicorn running on :8000?');
  } finally {
    hideLoading();
    els.btnInfer.disabled = false;
  }
}

els.btnInfer.addEventListener('click', runInference);

function clearOverlay(key) {
  if (overlayLayers[key]) {
    map.removeLayer(overlayLayers[key]);
    overlayLayers[key] = null;
  }
}

function renderLayers(preview, coords) {
  const bounds = boundsFromCoords(coords);

  clearOverlay('rgb'); clearOverlay('ndvi'); clearOverlay('sar'); clearOverlay('mask');

  overlayLayers.rgb = L.imageOverlay(`data:image/png;base64,${preview.rgb}`, bounds, { opacity: 1 });
  overlayLayers.ndvi = L.imageOverlay(`data:image/png;base64,${preview.ndvi}`, bounds, { opacity: 0.85 });
  overlayLayers.sar = L.imageOverlay(`data:image/png;base64,${preview.sar_vv}`, bounds, { opacity: 0.85 });
  overlayLayers.mask = L.imageOverlay(`data:image/png;base64,${preview.canopy_mask}`, bounds, { opacity: 0.6 });

  applyToggleState();
  map.fitBounds(bounds, { padding: [40, 40] });
}

function renderCrowns(geojson) {
  clearOverlay('crowns');
  overlayLayers.crowns = L.geoJSON(geojson, {
    style: (feature) => ({
      color: feature.properties.estimation_source === 'sar' ? '#4aa3c9' : '#d9614f',
      weight: 1,
      fillOpacity: 0.15,
    }),
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      layer.bindPopup(
        `<b>Tree #${p.tree_id}</b><br/>` +
        `Crown diameter: ${p.crown_diameter_m} m<br/>` +
        `NDVI: ${p.mean_ndvi}<br/>` +
        `AGB: ${p.agb_kg ?? '—'} kg<br/>` +
        `Source: ${p.estimation_source}<br/>` +
        `Confidence: ${p.confidence ?? '—'}`
      );
    },
  });
  if (els.toggles.crowns.checked) overlayLayers.crowns.addTo(map);
}

function applyToggleState() {
  const map_ = { rgb: 'rgb', ndvi: 'ndvi', sar: 'sar', mask: 'mask' };
  for (const key of Object.keys(map_)) {
    const on = els.toggles[key].checked;
    const layer = overlayLayers[key];
    if (!layer) continue;
    if (on && !map.hasLayer(layer)) layer.addTo(map);
    if (!on && map.hasLayer(layer)) map.removeLayer(layer);
  }
}

Object.entries(els.toggles).forEach(([key, input]) => {
  input.addEventListener('change', () => {
    if (key === 'crowns') {
      if (!overlayLayers.crowns) return;
      if (input.checked) overlayLayers.crowns.addTo(map);
      else map.removeLayer(overlayLayers.crowns);
      return;
    }
    applyToggleState();
  });
});

function fmt(n, unit) {
  if (n === undefined || n === null) return '—';
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M${unit}`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)} k${unit}`;
  return `${n.toFixed(1)} ${unit}`;
}

function renderResults(result) {
  const s = result.summary;
  els.emptyState.style.display = 'none';
  els.resultsContent.style.display = 'block';

  document.getElementById('r-count').textContent = s.tree_count.toLocaleString();
  document.getElementById('r-agb').textContent = fmt(s.total_agb_kg, 'g');
  document.getElementById('r-carbon').textContent = fmt(s.total_carbon_kg, 'gC');
  document.getElementById('r-co2').textContent = fmt(s.total_co2e_kg, 'g CO2e');
  document.getElementById('r-cloud').textContent = `${(s.cloud_fraction * 100).toFixed(0)}%`;
  document.getElementById('r-time').textContent = `${result.processing_seconds}s`;

  const sarCount = result.geojson.features.filter(f => f.properties.estimation_source === 'sar').length;
  const total = result.geojson.features.length || 1;
  const sarPct = Math.round((sarCount / total) * 100);
  document.getElementById('r-source').innerHTML =
    sarPct > 0
      ? `Optical <span class="badge optical">${100 - sarPct}%</span><span class="badge sar">SAR ${sarPct}%</span>`
      : `<span class="badge optical">Optical</span>`;

  document.getElementById('r-conf').textContent = `${Math.round(s.avg_confidence * 100)}%`;
  document.getElementById('confidence-fill').style.width = `${s.avg_confidence * 100}%`;
}

els.btnExport.addEventListener('click', () => {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult.geojson, null, 2)], { type: 'application/geo+json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sylvasense_canopy_boundaries.geojson';
  a.click();
  URL.revokeObjectURL(url);
});
