import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Cattle in South Sudan Movement Dashboard",
    layout="wide",
)

TITLE = "Cattle in South Sudan Movement Dashboard"
CAPTION = "Near-real-time cattle movement suitability, forecasts, and alerts (REAL: NASA GIBS tiles for rainfall/imagery)"
BOUNDARY_PATH = "data/south_sudan.geojson.json"

# Real basemaps / tiles
CARTO = "cartodbpositron"
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# NASA GIBS layers (REAL)
GIBS_MODIS_TRUECOLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
GIBS_IMERG_LAYER = "IMERG_Precipitation_Rate"


# ============================================================
# Helpers
# ============================================================
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


def yyyy_mm_dd_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def modis_date_string() -> str:
    # Using "yesterday" avoids "today tiles not published yet" problems.
    return yyyy_mm_dd_utc(datetime.now(timezone.utc) - timedelta(days=1))


def make_gibs_modis_truecolor_url(date_str: str) -> str:
    # WMTS GoogleMapsCompatible levels; Level9 is fairly detailed.
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_MODIS_TRUECOLOR_LAYER}/default/{date_str}/"
        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )


def make_gibs_imerg_url() -> str:
    # Near-real-time tiles (no explicit date)
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_IMERG_LAYER}/default/default/"
        "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    )


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ============================================================
# Simple demo engine (NOT real cattle telemetry)
# - Real NASA tiles are in basemap/overlay
# - These scores are a placeholder "suitability" heuristic
#   until you connect a real model/data pipeline.
# ============================================================
@dataclass(frozen=True)
class ScoredPoint:
    lat: float
    lon: float
    total: float
    rainfall: float
    forage: float
    access: float


@dataclass(frozen=True)
class AlertItem:
    point: ScoredPoint
    delta: float
    label: str
    drivers: List[Tuple[str, float]]  # sorted high->low


def make_points(
    seed: int,
    gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    step_deg: float = 0.6,
    keep_n: int = 35,
) -> List[ScoredPoint]:
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = gdf.total_bounds
    poly = gdf.unary_union

    points: List[ScoredPoint] = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    for lat in lats:
        for lon in lons:
            jlat = float(lat + rng.uniform(-0.12, 0.12))
            jlon = float(lon + rng.uniform(-0.12, 0.12))

            # keep only inside boundary
            try:
                pt = gpd.points_from_xy([jlon], [jlat])[0]
                if not poly.contains(pt):
                    continue
            except Exception:
                pass

            # --- proxy signals (0..1) ---
            # rainfall proxy: stable randomness per seed
            rainfall = float(rng.random())

            # forage proxy: encourages band near centroid latitude
            forage = float(1.0 - min(1.0, abs(jlat - center_lat) / 6.0))

            # access proxy: closeness to centroid
            access = float(1.0 - min(1.0, (abs(jlat - center_lat) + abs(jlon - center_lon)) / 10.0))

            total = 0.45 * forage + 0.35 * rainfall + 0.20 * access

            points.append(
                ScoredPoint(
                    lat=jlat,
                    lon=jlon,
                    total=clamp01(total),
                    rainfall=clamp01(rainfall),
                    forage=clamp01(forage),
                    access=clamp01(access),
                )
            )

    points.sort(key=lambda p: p.total, reverse=True)
    return points[:keep_n]


def top_drivers(p: ScoredPoint, n: int = 2) -> List[Tuple[str, float]]:
    drivers = [
        ("Rainfall", p.rainfall),
        ("Forage/Vegetation", p.forage),
        ("Access", p.access),
    ]
    drivers.sort(key=lambda t: t[1], reverse=True)
    return drivers[:n]


def label_from_delta(delta: float) -> str:
    # Vary labels so they don't all say the same thing
    if delta >= 0.70:
        return "New hotspot likely forming"
    if delta >= 0.45:
        return "Movement shift likely"
    if delta >= 0.25:
        return "Moderate increase"
    return "Small increase"


def compute_alerts(now_pts: List[ScoredPoint], fc_pts: List[ScoredPoint], k: int = 6) -> List[AlertItem]:
    # coarse matching by rounded coordinates
    now_index: Dict[Tuple[float, float], ScoredPoint] = {(round(p.lat, 1), round(p.lon, 1)): p for p in now_pts}

    diffs: List[Tuple[ScoredPoint, float]] = []
    for p in fc_pts:
        key = (round(p.lat, 1), round(p.lon, 1))
        if key in now_index:
            delta = p.total - now_index[key].total
        else:
            delta = p.total  # "new" hotspot relative to nowcast
        diffs.append((p, float(delta)))

    diffs.sort(key=lambda t: t[1], reverse=True)
    diffs = diffs[:k]

    alerts: List[AlertItem] = []
    for p, delta in diffs:
        alerts.append(
            AlertItem(
                point=p,
                delta=delta,
                label=label_from_delta(delta),
                drivers=top_drivers(p, n=2),
            )
        )
    return alerts


def nearest_point(lat: float, lon: float, pts: List[ScoredPoint]) -> Tuple[Optional[int], Optional[ScoredPoint], float]:
    best_idx = None
    best_p = None
    best_d2 = 1e18
    for i, p in enumerate(pts):
        d2 = (p.lat - lat) ** 2 + (p.lon - lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
            best_p = p
    return best_idx, best_p, float(best_d2)


# ============================================================
# UI: Header
# ============================================================
st.title(TITLE)
st.caption(CAPTION)

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__

centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)


# ============================================================
# Session state (fix “weird zoom” / blinking)
# ============================================================
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
if "selected_info" not in st.session_state:
    st.session_state.selected_info = None  # dict shown on right panel


# ============================================================
# Sidebar controls
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
st.sidebar.subheader("Real public data overlay (NASA GIBS)")
show_imerg = st.sidebar.checkbox("IMERG Precipitation Rate (30-min) — NASA GIBS", value=False)
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.70, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Model layers (dashboard outputs)")
show_boundary = st.sidebar.checkbox("South Sudan boundary", value=True)

show_now_heat = st.sidebar.checkbox("Nowcast heatmap", value=True)
show_now_markers = st.sidebar.checkbox("Nowcast markers", value=True)

show_7_heat = st.sidebar.checkbox("Forecast (7 days) heatmap", value=False)
show_7_markers = st.sidebar.checkbox("Forecast (7 days) markers", value=False)

show_30_heat = st.sidebar.checkbox("Forecast (30 days) heatmap", value=False)
show_30_markers = st.sidebar.checkbox("Forecast (30 days) markers", value=False)

show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", 6, 40, 18, 1)
heat_opacity = st.sidebar.slider("Heat opacity", 0.10, 0.95, 0.55, 0.05)

st.sidebar.divider()
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Reset view"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 6
        st.rerun()
with c2:
    if st.button("Zoom to S. Sudan"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 7
        st.rerun()

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh (reload page)", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 30, 600, 120, 30)

if auto_refresh:
    # No extra packages required.
    st.components.v1.html(
        f"<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_seconds)*1000});</script>",
        height=0,
    )


# ============================================================
# Generate points (weekly changes)
# ============================================================
week_seed = int(datetime.now(timezone.utc).strftime("%Y%U"))
now_pts = make_points(week_seed, gdf, CENTER_LAT, CENTER_LON)
fc7_pts = make_points(week_seed + 7, gdf, CENTER_LAT, CENTER_LON)
fc30_pts = make_points(week_seed + 30, gdf, CENTER_LAT, CENTER_LON)

alerts_7 = compute_alerts(now_pts, fc7_pts, k=6)
alerts_30 = compute_alerts(now_pts, fc30_pts, k=6)


# ============================================================
# Build Folium map
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

modis_date = modis_date_string()
folium.TileLayer(
    tiles=make_gibs_modis_truecolor_url(modis_date),
    attr=f"NASA GIBS (MODIS True Color) — {modis_date}",
    name="MODIS True Color (daily) — NASA GIBS",
    overlay=False,
    control=True,
    show=(basemap_choice == "MODIS True Color (daily) — NASA GIBS"),
).add_to(m)

# IMERG overlay (REAL) + opacity works
if show_imerg:
    folium.TileLayer(
        tiles=make_gibs_imerg_url(),
        attr="NASA GIBS (GPM IMERG Precipitation Rate)",
        name="IMERG Precipitation Rate (30-min) — NASA GIBS",
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
        style_function=lambda x: {"fillOpacity": 0.0, "color": "black", "weight": 3},
        show=True,
    ).add_to(m)


def add_heat(points: List[ScoredPoint], name: str, show: bool) -> None:
    heat_data = [[p.lat, p.lon, p.total] for p in points]
    HeatMap(
        heat_data,
        name=name,
        radius=28,
        blur=22,
        min_opacity=0.10,
        max_opacity=float(heat_opacity),
        show=show,
    ).add_to(m)


def add_markers(points: List[ScoredPoint], name: str, show: bool, max_n: int = 14) -> None:
    fg = folium.FeatureGroup(name=name, show=show)
    for i, p in enumerate(points[:max_n], start=1):
        drivers = top_drivers(p, n=2)
        why = ", ".join([f"{k} ({v:.2f})" for k, v in drivers])

        popup_html = (
            f"<b>{name} #{i}</b><br>"
            f"Total: {p.total:.2f}<br>"
            f"Rainfall: {p.rainfall:.2f} | Forage: {p.forage:.2f} | Access: {p.access:.2f}<br>"
            f"Lat/Lon: {p.lat:.3f}, {p.lon:.3f}<br>"
            f"<b>Top drivers:</b> {why}"
        )

        folium.CircleMarker(
            location=[p.lat, p.lon],
            radius=int(marker_size),
            weight=2,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.85,
            tooltip=f"{name} #{i} • total {p.total:.2f}",
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(fg)

    fg.add_to(m)


# Model layers: split heatmap vs markers exactly
if show_now_heat:
    add_heat(now_pts, "Nowcast heatmap", show=True)
if show_now_markers:
    add_markers(now_pts, "Nowcast markers", show=True)

if show_7_heat:
    add_heat(fc7_pts, "Forecast (7 days) heatmap", show=True)
if show_7_markers:
    add_markers(fc7_pts, "Forecast (7 days) markers", show=True)

if show_30_heat:
    add_heat(fc30_pts, "Forecast (30 days) heatmap", show=True)
if show_30_markers:
    add_markers(fc30_pts, "Forecast (30 days) markers", show=True)

folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# Layout
# ============================================================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    map_out = st_folium(m, width=None, height=650)

# Persist user pan/zoom (this is what stops “weird zoom”)
if isinstance(map_out, dict):
    # Keep view updated from user interactions
    if map_out.get("center") and map_out.get("zoom") is not None:
        st.session_state.map_center = [map_out["center"]["lat"], map_out["center"]["lng"]]
        st.session_state.map_zoom = int(map_out["zoom"])

# Marker click -> show details on right
clicked = (map_out or {}).get("last_object_clicked")
if clicked and "lat" in clicked and "lng" in clicked:
    clat, clon = float(clicked["lat"]), float(clicked["lng"])

    # find which layer it’s closest to (now / 7 / 30)
    i_now, p_now, d_now = nearest_point(clat, clon, now_pts)
    i_7, p_7, d_7 = nearest_point(clat, clon, fc7_pts)
    i_30, p_30, d_30 = nearest_point(clat, clon, fc30_pts)

    best = min(
        [("Nowcast", i_now, p_now, d_now), ("Forecast 7d", i_7, p_7, d_7), ("Forecast 30d", i_30, p_30, d_30)],
        key=lambda t: t[3],
    )

    name, idx, p, d2 = best
    # Only accept if click is reasonably close (degrees² threshold)
    if p is not None and d2 <= (0.35 ** 2):
        drivers = top_drivers(p, n=2)
        st.session_state.selected_info = {
            "layer": name,
            "idx": int(idx) + 1 if idx is not None else None,
            "lat": p.lat,
            "lon": p.lon,
            "total": p.total,
            "rainfall": p.rainfall,
            "forage": p.forage,
            "access": p.access,
            "drivers": drivers,
        }


# ============================================================
# Alerts panel
# ============================================================
def alert_card(title: str, idx: int, a: AlertItem, key_prefix: str):
    p = a.point
    drivers_txt = ", ".join([f"{k} ({v:.2f})" for k, v in a.drivers])

    st.markdown(f"**{idx}. {a.label}**")
    st.caption(f"Lat/Lon: {p.lat:.3f}, {p.lon:.3f}")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Delta vs nowcast", f"{a.delta:+.2f}")
    with m2:
        st.metric("Forecast total", f"{p.total:.2f}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Rainfall", f"{p.rainfall:.2f}")
    s2.metric("Forage", f"{p.forage:.2f}")
    s3.metric("Access", f"{p.access:.2f}")

    st.caption(f"Top drivers: {drivers_txt}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button(f"Zoom to alert #{idx}", key=f"{key_prefix}_z_{idx}"):
            st.session_state.map_center = [p.lat, p.lon]
            st.session_state.map_zoom = 9
            st.rerun()
    with b2:
        if st.button("Zoom closer", key=f"{key_prefix}_zz_{idx}"):
            st.session_state.map_center = [p.lat, p.lon]
            st.session_state.map_zoom = 11
            st.rerun()

    st.divider()


with right:
    st.header("Alerts")
    st.caption("Alerts highlight places where forecast suitability increases vs nowcast (delta score).")

    # Selected marker info
    if st.session_state.selected_info is not None:
        si = st.session_state.selected_info
        st.subheader("Selected marker")
        st.caption(f"{si['layer']} • point #{si['idx']} • Lat/Lon: {si['lat']:.3f}, {si['lon']:.3f}")

        mm1, mm2 = st.columns(2)
        mm1.metric("Total score", f"{si['total']:.2f}")
        mm2.metric("Top drivers", ", ".join([f"{k} ({v:.2f})" for k, v in si["drivers"]]))

        ss1, ss2, ss3 = st.columns(3)
        ss1.metric("Rainfall", f"{si['rainfall']:.2f}")
        ss2.metric("Forage", f"{si['forage']:.2f}")
        ss3.metric("Access", f"{si['access']:.2f}")

        st.divider()

    st.info(
        f"**Real layers:** MODIS True Color (NASA GIBS, {modis_date}) and IMERG precipitation (NASA GIBS). "
        f"**Your ‘cow movement’ layers** are currently a placeholder scoring engine until you connect a real model/data feed."
    )

    if not show_alerts:
        st.warning("Turn on **Anomaly alerts** in the sidebar to view alerts.")
    else:
        st.subheader("7-day")
        for i, a in enumerate(alerts_7, start=1):
            alert_card("7-day", i, a, "a7")

        st.subheader("30-day")
        for i, a in enumerate(alerts_30, start=1):
            alert_card("30-day", i, a, "a30")
