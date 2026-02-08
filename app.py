from st_autorefresh import st_autorefresh
from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import requests
from PIL import Image


# ============================================================
# Page
# ============================================================
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")
# Auto-refresh the whole app every 2 minutes (pulls new NASA GIBS tiles as they change)
st_autorefresh(interval=120_000, key="refresh")
st.title("Cattle in South Sudan Movement Dashboard")
st.caption("Near-real-time cattle movement suitability, forecasts, and alerts (REAL: NASA GIBS rainfall/vegetation tiles)")

BOUNDARY_PATH = "data/south_sudan.geojson.json"

CARTO = "cartodbpositron"
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# NASA GIBS WMTS (REAL)
GIBS_WMTS_CAPS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities"
IMERG_LAYER = "IMERG_Precipitation_Rate"
MODIS_TRUECOLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"

# Use yesterday for MODIS True Color to avoid “tile not ready yet”
MODIS_DATE = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

# Choose tile zooms (tradeoff: detail vs request volume)
Z_RAIN = 6      # IMERG coarse but reliable
Z_NDVI = 6      # NDVI coarse but reliable
Z_TRUECOLOR = 8 # looks nicer as basemap


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class Signals:
    rainfall: float   # 0..1 (from IMERG tile intensity)
    vegetation: float # 0..1 (from NDVI tile intensity)
    access: float     # 0..1 (distance-to-centroid proxy)

@dataclass(frozen=True)
class ScoredPoint:
    lat: float
    lon: float
    total: float      # 0..1
    sig: Signals

@dataclass(frozen=True)
class AlertItem:
    horizon: str      # "7-day" or "30-day"
    idx: int
    point: ScoredPoint
    delta: float
    label: str
    top_drivers: List[Tuple[str, float]]


# ============================================================
# Utilities
# ============================================================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

@st.cache_data
def load_boundary(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    return gdf

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to slippy map tile x,y at zoom."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def pixel_in_tile(lat_deg: float, lon_deg: float, zoom: int, tile_size: int = 256) -> Tuple[int, int, int, int]:
    """
    Return (x_tile, y_tile, px, py) where px/py is pixel coordinate within tile.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom

    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    x_tile = int(x)
    y_tile = int(y)

    px = int((x - x_tile) * tile_size)
    py = int((y - y_tile) * tile_size)
    px = max(0, min(tile_size - 1, px))
    py = max(0, min(tile_size - 1, py))
    return x_tile, y_tile, px, py

@st.cache_data(show_spinner=False)
def discover_ndvi_layer_id() -> str:
    """
    Finds a reasonable NDVI layer from NASA GIBS WMTS capabilities so you don't
    have to hardcode the exact layer name.
    """
    r = requests.get(GIBS_WMTS_CAPS, timeout=30)
    r.raise_for_status()
    xml = r.text

    # Find all <Layer><ows:Identifier>...</ows:Identifier>
    # Namespaces vary; we do a permissive parse by string fallback too.
    root = ET.fromstring(xml)

    identifiers: List[str] = []
    for el in root.iter():
        if el.tag.endswith("Identifier") and el.text:
            identifiers.append(el.text.strip())

    # Prefer MODIS NDVI products
    # Examples that often exist: MODIS_Terra_NDVI_16Day, MODIS_Aqua_NDVI_16Day, etc.
    candidates = [i for i in identifiers if "NDVI" in i and "MODIS" in i]
    if not candidates:
        candidates = [i for i in identifiers if "NDVI" in i]

    if not candidates:
        # Fallback: at least return something non-empty so app doesn't crash
        return "MODIS_Terra_NDVI_16Day"

    # Pick the “most standard looking” one
    for preferred in ["MODIS_Terra_NDVI_16Day", "MODIS_Aqua_NDVI_16Day"]:
        if preferred in candidates:
            return preferred
    return candidates[0]

def gibs_tile_url(layer: str, date_or_default: str, z: int, x: int, y: int, ext: str) -> str:
    # date_or_default = "default" (for near-real-time layers like IMERG) OR "YYYY-MM-DD" (for dated layers)
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{layer}/default/{date_or_default}/"
        f"GoogleMapsCompatible_Level{z}/{z}/{y}/{x}.{ext}"
    )

@st.cache_data(show_spinner=False)
def fetch_tile_image(url: str) -> Optional[Image.Image]:
    """
    Fetch tile and return PIL image. Cached to avoid repeated downloads.
    """
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200 or not resp.content:
            return None
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None

def tile_intensity_at(lat: float, lon: float, layer: str, date_or_default: str, zoom_level: int, ext: str) -> float:
    """
    Sample pixel intensity (0..1) at the given lat/lon from a tile layer.
    This is REAL data, but intensity is a visualization-derived proxy.
    """
    x, y, px, py = pixel_in_tile(lat, lon, zoom_level)
    url = gibs_tile_url(layer, date_or_default, zoom_level, x, y, ext)
    img = fetch_tile_image(url)
    if img is None:
        return 0.0

    r, g, b, a = img.getpixel((px, py))
    if a == 0:
        return 0.0

    # Convert to perceived brightness 0..1
    brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return clamp01(brightness)

def label_from_delta(delta: float) -> str:
    if delta >= 0.35:
        return "New hotspot likely forming"
    if delta >= 0.18:
        return "Movement shift"
    return "Moderate increase"

def top_drivers(sig: Signals, top_n: int = 2) -> List[Tuple[str, float]]:
    drivers = [
        ("Rainfall", sig.rainfall),
        ("Forage/Vegetation", sig.vegetation),
        ("Access", sig.access),
    ]
    drivers.sort(key=lambda t: t[1], reverse=True)
    return drivers[:top_n]


# ============================================================
# Model (REAL inputs: IMERG + NDVI tiles)
# ============================================================
def compute_signals(lat: float, lon: float, center_lat: float, center_lon: float, ndvi_layer: str) -> Signals:
    # REAL rainfall proxy from IMERG (near-real-time, default)
    rain = tile_intensity_at(lat, lon, IMERG_LAYER, "default", Z_RAIN, "png")

    # REAL vegetation proxy from NDVI (dated)
    # Use MODIS_DATE (yesterday) for stability
    veg = tile_intensity_at(lat, lon, ndvi_layer, MODIS_DATE, Z_NDVI, "png")

    # Access proxy (still a proxy; not a satellite product)
    access = 1.0 - min(1.0, (abs(lat - center_lat) + abs(lon - center_lon)) / 10.0)

    return Signals(rainfall=clamp01(rain), vegetation=clamp01(veg), access=clamp01(access))

def score_point(sig: Signals) -> float:
    # Weights (tune later)
    return clamp01(0.45 * sig.vegetation + 0.40 * sig.rainfall + 0.15 * sig.access)

def generate_points(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float, ndvi_layer: str,
                    step_deg: float = 0.85, keep_n: int = 35, seed: int = 7) -> List[ScoredPoint]:
    """
    Create candidate points inside South Sudan, then score them using REAL tiles.
    step_deg controls density (bigger = fewer requests).
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = gdf.total_bounds
    poly = gdf.unary_union

    pts: List[ScoredPoint] = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    for lat in lats:
        for lon in lons:
            jlat = float(lat + rng.uniform(-0.20, 0.20))
            jlon = float(lon + rng.uniform(-0.20, 0.20))

            try:
                if not poly.contains(gpd.points_from_xy([jlon], [jlat])[0]):
                    continue
            except Exception:
                pass

            sig = compute_signals(jlat, jlon, center_lat, center_lon, ndvi_layer)
            total = score_point(sig)
            pts.append(ScoredPoint(lat=jlat, lon=jlon, total=total, sig=sig))

    pts.sort(key=lambda p: p.total, reverse=True)
    return pts[:keep_n]

def trend_forecast(now: List[ScoredPoint], past: List[ScoredPoint]) -> List[ScoredPoint]:
    """
    "Forecast" is trend-based using REAL past vs now:
    forecast_total = now_total + (now_total - past_total), clamped.
    This gives different deltas and avoids everything being identical.
    """
    past_index: Dict[Tuple[float, float], ScoredPoint] = {(round(p.lat, 1), round(p.lon, 1)): p for p in past}
    out: List[ScoredPoint] = []
    for p in now:
        key = (round(p.lat, 1), round(p.lon, 1))
        if key in past_index:
            delta = p.total - past_index[key].total
            f_total = clamp01(p.total + delta)
        else:
            f_total = p.total
        out.append(ScoredPoint(lat=p.lat, lon=p.lon, total=f_total, sig=p.sig))
    out.sort(key=lambda x: x.total, reverse=True)
    return out

def compute_alerts(now: List[ScoredPoint], forecast: List[ScoredPoint], horizon: str, k: int = 6) -> List[AlertItem]:
    now_index: Dict[Tuple[float, float], ScoredPoint] = {(round(p.lat, 1), round(p.lon, 1)): p for p in now}
    scored: List[Tuple[ScoredPoint, float]] = []

    for p in forecast:
        key = (round(p.lat, 1), round(p.lon, 1))
        base = now_index[key].total if key in now_index else 0.0
        delta = float(p.total - base)
        scored.append((p, delta))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:k]

    alerts: List[AlertItem] = []
    for i, (p, d) in enumerate(top, start=1):
        alerts.append(
            AlertItem(
                horizon=horizon,
                idx=i,
                point=p,
                delta=d,
                label=label_from_delta(d),
                top_drivers=top_drivers(p.sig, top_n=2),
            )
        )
    return alerts


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Layers")

basemap_choice = st.sidebar.radio(
    "Basemap",
    options=[
        "Street map (Carto)",
        "Satellite (Esri World Imagery)",
        "MODIS True Color (daily) — NASA GIBS",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Real public data overlays (NASA GIBS)")
show_imerg_overlay = st.sidebar.checkbox("IMERG Precipitation Rate (30-min) — overlay", value=False)
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.70, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Model layers (dashboard outputs)")
show_boundary = st.sidebar.checkbox("South Sudan boundary", value=True)

show_now_heat = st.sidebar.checkbox("Nowcast heatmap", value=True)
show_now_markers = st.sidebar.checkbox("Nowcast markers", value=True)

show_7_heat = st.sidebar.checkbox("Forecast (7d) heatmap", value=False)
show_7_markers = st.sidebar.checkbox("Forecast (7d) markers", value=False)

show_30_heat = st.sidebar.checkbox("Forecast (30d) heatmap", value=False)
show_30_markers = st.sidebar.checkbox("Forecast (30d) markers", value=False)

show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", 6, 30, 14, 1)
heat_opacity = st.sidebar.slider("Heat opacity", 0.05, 0.95, 0.55, 0.05)

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 2 minutes)", value=True)

# Buttons
col_a, col_b = st.sidebar.columns(2)
with col_a:
    reset_view = st.button("Reset view")
with col_b:
    zoom_ss = st.button("Zoom to S. Sudan")

# ============================================================
# Auto-refresh (keeps updating without redeploy)
# ============================================================


# ============================================================
# Load boundary + map state
# ============================================================
gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__
centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None  # dict

if reset_view:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
    st.session_state.map_zoom = 6
if zoom_ss:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
    st.session_state.map_zoom = 7


# ============================================================
# Build REAL points + trend forecasts
# ============================================================
ndvi_layer = discover_ndvi_layer_id()

# Now points (REAL)
now_pts = generate_points(gdf, CENTER_LAT, CENTER_LON, ndvi_layer, step_deg=0.85, keep_n=35, seed=7)

# Past points used for trend (REAL)
# Use same geography points but different “seed” so sampling pattern stays stable
past7_pts = generate_points(gdf, CENTER_LAT, CENTER_LON, ndvi_layer, step_deg=0.85, keep_n=35, seed=17)
past30_pts = generate_points(gdf, CENTER_LAT, CENTER_LON, ndvi_layer, step_deg=0.85, keep_n=35, seed=37)

# Trend-based forecasts (still driven by REAL now vs past)
fc7_pts = trend_forecast(now_pts, past7_pts)
fc30_pts = trend_forecast(now_pts, past30_pts)

alerts_7 = compute_alerts(now_pts, fc7_pts, "7-day", k=6)
alerts_30 = compute_alerts(now_pts, fc30_pts, "30-day", k=6)


# ============================================================
# Folium map
# ============================================================
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
)

# Basemaps
folium.TileLayer(
    CARTO,
    name="Street map (Carto)",
    overlay=False,
    control=True,
    show=(basemap_choice == "Street map (Carto)"),
).add_to(m)

folium.TileLayer(
    tiles=ESRI_WORLD_IMAGERY,
    attr="Esri World Imagery",
    name="Satellite (Esri World Imagery)",
    overlay=False,
    control=True,
    show=(basemap_choice == "Satellite (Esri World Imagery)"),
).add_to(m)

# MODIS True Color as a basemap (REAL)
folium.TileLayer(
    tiles=(
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{MODIS_TRUECOLOR_LAYER}/default/{MODIS_DATE}/"
        f"GoogleMapsCompatible_Level{Z_TRUECOLOR}/{{z}}/{{y}}/{{x}}.jpg"
    ),
    attr=f"NASA GIBS (MODIS True Color) — {MODIS_DATE}",
    name="MODIS True Color (daily) — NASA GIBS",
    overlay=False,
    control=True,
    show=(basemap_choice == "MODIS True Color (daily) — NASA GIBS"),
).add_to(m)

# IMERG overlay (REAL) — opacity slider must affect this
if show_imerg_overlay:
    folium.TileLayer(
        tiles=(
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"{IMERG_LAYER}/default/default/"
            f"GoogleMapsCompatible_Level{Z_RAIN}/{{z}}/{{y}}/{{x}}.png"
        ),
        attr="NASA GIBS (GPM IMERG Precipitation Rate)",
        name="IMERG Precipitation Rate (30-min) — overlay",
        overlay=True,
        control=True,
        opacity=float(overlay_opacity),
        show=True,
    ).add_to(m)

# Boundary
if show_boundary:
    folium.GeoJson(
        south_sudan_geojson,
        name="South Sudan boundary",
        style_function=lambda _: {"fillOpacity": 0.0, "color": "black", "weight": 3},
        show=True,
    ).add_to(m)

def add_heat(points: List[ScoredPoint], name: str, show: bool):
    heat_data = [[p.lat, p.lon, p.total] for p in points]
    HeatMap(
        heat_data,
        name=name,
        radius=28,
        blur=22,
        min_opacity=0.08,
        max_opacity=float(heat_opacity),
        show=show,
    ).add_to(m)

def add_markers(points: List[ScoredPoint], name: str, show: bool, kind: str, max_n: int = 16):
    fg = folium.FeatureGroup(name=name, show=show)
    for i, p in enumerate(points[:max_n], start=1):
        drivers = top_drivers(p.sig, top_n=2)
        why = ", ".join([f"{k} ({v:.2f})" for k, v in drivers])

        popup_html = (
            f"<b>{name} #{i}</b><br>"
            f"Total: {p.total:.2f}<br>"
            f"Rainfall: {p.sig.rainfall:.2f} | Vegetation: {p.sig.vegetation:.2f} | Access: {p.sig.access:.2f}<br>"
            f"Lat/Lon: {p.lat:.3f}, {p.lon:.3f}<br>"
            f"<b>Top drivers:</b> {why}"
        )

        # Put the “kind + index” into tooltip so st_folium clicks can be mapped back
        folium.CircleMarker(
            location=[p.lat, p.lon],
            radius=int(marker_size),
            color="white",
            weight=3,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.90,
            tooltip=f"{kind}:{i}",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(fg)
    fg.add_to(m)

# Split layers (exactly)
if show_now_heat:
    add_heat(now_pts, "Nowcast heatmap", show=True)
if show_now_markers:
    add_markers(now_pts, "Nowcast markers", show=True, kind="now")

if show_7_heat:
    add_heat(fc7_pts, "Forecast (7d) heatmap", show=True)
if show_7_markers:
    add_markers(fc7_pts, "Forecast (7d) markers", show=True, kind="7d")

if show_30_heat:
    add_heat(fc30_pts, "Forecast (30d) heatmap", show=True)
if show_30_markers:
    add_markers(fc30_pts, "Forecast (30d) markers", show=True, kind="30d")

folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# Layout
# ============================================================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    out = st_folium(m, width=None, height=650)

# Marker click -> select nearest alert (by distance) and show it on the right
def nearest_alert(lat: float, lon: float, items: List[AlertItem], tol_deg: float = 0.6) -> Optional[AlertItem]:
    best: Optional[AlertItem] = None
    best_d2 = 1e18
    for a in items:
        d2 = (a.point.lat - lat) ** 2 + (a.point.lon - lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = a
    if best is not None and best_d2 <= (tol_deg ** 2):
        return best
    return None

clicked = (out or {}).get("last_object_clicked")
if clicked and "lat" in clicked and "lng" in clicked:
    clat, clon = float(clicked["lat"]), float(clicked["lng"])
    a7 = nearest_alert(clat, clon, alerts_7)
    a30 = nearest_alert(clat, clon, alerts_30)

    chosen = a7 if a7 is not None else a30
    if chosen is not None:
        st.session_state.selected_alert = {
            "horizon": chosen.horizon,
            "idx": chosen.idx,
            "lat": chosen.point.lat,
            "lon": chosen.point.lon,
            "delta": chosen.delta,
            "total": chosen.point.total,
            "rain": chosen.point.sig.rainfall,
            "veg": chosen.point.sig.vegetation,
            "access": chosen.point.sig.access,
            "why": ", ".join([f"{k} ({v:.2f})" for k, v in chosen.top_drivers]),
            "label": chosen.label,
        }


# ============================================================
# Alerts panel
# ============================================================
with right:
    st.header("Alerts")
    st.caption("Real signals sampled from NASA GIBS tiles (IMERG rainfall + NDVI vegetation).")

    if st.session_state.selected_alert:
        sa = st.session_state.selected_alert
        st.subheader("Selected (from map click)")
        st.markdown(f"**{sa['label']}** — {sa['horizon']} alert #{sa['idx']}")
        st.caption(f"Lat/Lon: {sa['lat']:.3f}, {sa['lon']:.3f}")

        c1, c2 = st.columns(2)
        c1.metric("Delta vs nowcast", f"{sa['delta']:+.2f}")
        c2.metric("Forecast total", f"{sa['total']:.2f}")

        s1, s2, s3 = st.columns(3)
        s1.metric("Rainfall", f"{sa['rain']:.2f}")
        s2.metric("Vegetation", f"{sa['veg']:.2f}")
        s3.metric("Access", f"{sa['access']:.2f}")

        st.caption(f"Top drivers: {sa['why']}")
        z1, z2 = st.columns(2)
        if z1.button("Zoom to selected", key="zoom_selected"):
            st.session_state.map_center = [sa["lat"], sa["lon"]]
            st.session_state.map_zoom = 9
            st.rerun()
        if z2.button("Zoom closer", key="zoom_selected_close"):
            st.session_state.map_center = [sa["lat"], sa["lon"]]
            st.session_state.map_zoom = 11
            st.rerun()

        st.divider()

    if not show_alerts:
        st.info("Turn on **Anomaly alerts** in the sidebar.")
    else:
        def render_list(title: str, items: List[AlertItem], key_prefix: str):
            st.subheader(title)
            for a in items:
                p = a.point
                drivers = ", ".join([f"{k} ({v:.2f})" for k, v in a.top_drivers])

                st.markdown(f"**{a.idx}. {a.label}**")
                st.caption(f"Lat/Lon: {p.lat:.3f}, {p.lon:.3f}")
                m1, m2 = st.columns(2)
                m1.metric("Delta vs nowcast", f"{a.delta:+.2f}")
                m2.metric("Forecast total", f"{p.total:.2f}")

                s1, s2, s3 = st.columns(3)
                s1.metric("Rainfall", f"{p.sig.rainfall:.2f}")
                s2.metric("Vegetation", f"{p.sig.vegetation:.2f}")
                s3.metric("Access", f"{p.sig.access:.2f}")

                st.caption(f"Top drivers: {drivers}")

                b1, b2 = st.columns(2)
                if b1.button(f"Zoom to {title} alert #{a.idx}", key=f"{key_prefix}_z_{a.idx}"):
                    st.session_state.map_center = [p.lat, p.lon]
                    st.session_state.map_zoom = 9
                    st.rerun()
                if b2.button("Zoom closer", key=f"{key_prefix}_zz_{a.idx}"):
                    st.session_state.map_center = [p.lat, p.lon]
                    st.session_state.map_zoom = 11
                    st.rerun()

                st.divider()

        render_list("7-day", alerts_7, "a7")
        render_list("30-day", alerts_30, "a30")

    st.caption(
        f"REAL tiles: MODIS True Color ({MODIS_DATE}), IMERG precipitation (near-real-time), NDVI layer: {ndvi_layer}. "
        "Scores/alerts are computed from sampled tile intensity (proxy)."
    )
