from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium


# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")

TITLE = "Cattle in South Sudan Movement Dashboard"
CAPTION = "Near-real-time (weekly) cattle movement suitability, forecasts, and alerts"

BOUNDARY_PATH = "data/south_sudan.geojson.json"

# NASA GIBS tiles (REAL data)
# MODIS True Color (daily) – needs a date in the URL
# Docs pattern: .../layer/default/{YYYY-MM-DD}/GoogleMapsCompatible_Level{N}/{z}/{y}/{x}.jpg
GIBS_MODIS_TRUECOLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
GIBS_IMERG_LAYER = "IMERG_Precipitation_Rate"

# Esri World Imagery (REAL data)
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# Carto basemap
CARTO = "cartodbpositron"


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class ScoredPoint:
    lat: float
    lon: float
    total: float
    forage: float
    rainfall: float
    access: float


@dataclass(frozen=True)
class AlertItem:
    point: ScoredPoint
    delta: float
    label: str
    top_drivers: List[Tuple[str, float]]


# ============================================================
# Helpers
# ============================================================
@st.cache_data
def load_boundary(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # Force EPSG:4326
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        # keep whatever it is; don't crash app
        pass
    return gdf


def today_utc_yyyy_mm_dd() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def make_gibs_modis_truecolor_url(date_str: str) -> str:
    # Level9 gives nicer detail; you can change to Level8 if needed
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_MODIS_TRUECOLOR_LAYER}/default/{date_str}/"
        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )


def make_gibs_imerg_url() -> str:
    # IMERG precip rate – PNG tiles, no date in URL (near-real-time service)
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_IMERG_LAYER}/default/default/"
        "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    )


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def explain_drivers(p: ScoredPoint, top_n: int = 2) -> List[Tuple[str, float]]:
    drivers = [
        ("Rainfall", p.rainfall),
        ("Forage/Vegetation", p.forage),
        ("Access", p.access),
    ]
    drivers.sort(key=lambda t: t[1], reverse=True)
    return drivers[:top_n]


def label_from_delta(delta: float) -> str:
    # You can tune thresholds later
    if delta >= 0.75:
        return "New hotspot likely forming"
    if delta >= 0.45:
        return "Route deviation likely"
    return "Moderate increase"


# ============================================================
# Simple heuristic engine (v1)
# ============================================================
def make_points(
    seed: int,
    gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    step_deg: float = 0.6,
    keep_n: int = 30,
) -> List[ScoredPoint]:
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = gdf.total_bounds
    poly = gdf.unary_union

    points: List[ScoredPoint] = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    for lat in lats:
        for lon in lons:
            # jitter so it doesn't look like a perfect grid
            jlat = float(lat + rng.uniform(-0.12, 0.12))
            jlon = float(lon + rng.uniform(-0.12, 0.12))

            # keep only if inside boundary
            try:
                pt = gpd.points_from_xy([jlon], [jlat])[0]
                if not poly.contains(pt):
                    continue
            except Exception:
                # don't crash if geometry check fails
                pass

            # --- Model signals (0..1) ---
            # "Forage" proxy: favor mid-lat band relative to centroid
            forage = 1.0 - min(1.0, abs(jlat - center_lat) / 6.0)

            # "Rainfall" proxy: random (placeholder) but stable per seed
            rainfall = rng.random()

            # "Access" proxy: favor closer to centroid
            access = 1.0 - min(1.0, (abs(jlat - center_lat) + abs(jlon - center_lon)) / 10.0)

            # Weighted total
            total = 0.45 * forage + 0.35 * rainfall + 0.20 * access

            points.append(
                ScoredPoint(
                    lat=jlat,
                    lon=jlon,
                    total=clamp01(total),
                    forage=clamp01(forage),
                    rainfall=clamp01(rainfall),
                    access=clamp01(access),
                )
            )

    points.sort(key=lambda p: p.total, reverse=True)
    return points[:keep_n]


def compute_alerts(now_pts: List[ScoredPoint], forecast_pts: List[ScoredPoint], k: int = 6) -> List[AlertItem]:
    # Index now by rounded lat/lon (coarse matching)
    now_index: Dict[Tuple[float, float], ScoredPoint] = {(round(p.lat, 1), round(p.lon, 1)): p for p in now_pts}

    scored: List[Tuple[ScoredPoint, float]] = []
    for p in forecast_pts:
        key = (round(p.lat, 1), round(p.lon, 1))
        delta = p.total - now_index[key].total if key in now_index else p.total
        scored.append((p, float(delta)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:k]

    alerts: List[AlertItem] = []
    for p, delta in top:
        label = label_from_delta(delta)
        alerts.append(
            AlertItem(
                point=p,
                delta=delta,
                label=label,
                top_drivers=explain_drivers(p, top_n=2),
            )
        )
    return alerts


# ============================================================
# UI
# ============================================================
st.title(TITLE)
st.caption(CAPTION)

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__

centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

# --- Session state for stable map view ---
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
# selection state (marker click -> selects an alert)
if "selected_kind" not in st.session_state:
    st.session_state.selected_kind = None   # "7d" or "30d"
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None    # 1-based alert number
if "force_view" not in st.session_state:
    st.session_state.force_view = False  # only True when user clicks zoom buttons

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Layers")

basemap_choice = st.sidebar.radio(
    "Basemap",
    options=["Street map (Carto)", "Satellite (Esri World Imagery)", "MODIS True Color (daily) — NASA GIBS"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Real public data overlays (NASA GIBS)")
show_imerg = st.sidebar.checkbox("Weather: IMERG Precipitation Rate (30-min) — NASA GIBS", value=False)

overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.70, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Model layers (dashboard outputs)")
show_now_heat = st.sidebar.checkbox("Nowcast heatmap", value=True)
show_now_markers = st.sidebar.checkbox("Nowcast markers", value=True)

show_7_heat = st.sidebar.checkbox("Forecast (7d) heatmap", value=False)
show_7_markers = st.sidebar.checkbox("Forecast (7d) markers", value=False)

show_30_heat = st.sidebar.checkbox("Forecast (30d) heatmap", value=False)
show_30_markers = st.sidebar.checkbox("Forecast (30d) markers", value=False)

show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)
show_boundary = st.sidebar.checkbox("South Sudan boundary", value=True)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", 6, 40, 18, 1)
heat_max_opacity = st.sidebar.slider("Heat opacity (max)", 0.10, 0.95, 0.55, 0.05)

st.sidebar.divider()
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Reset view"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 6
        st.session_state.force_view = True
        st.rerun()

with col_b:
    if st.button("Zoom to S. Sudan"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 7
        st.session_state.force_view = True
        st.rerun()


# ============================================================
# Generate model points (weekly)
# ============================================================
week_seed = int(datetime.now(timezone.utc).strftime("%Y%U"))

now_pts = make_points(seed=week_seed, gdf=gdf, center_lat=CENTER_LAT, center_lon=CENTER_LON)
fc7_pts = make_points(seed=week_seed + 7, gdf=gdf, center_lat=CENTER_LAT, center_lon=CENTER_LON)
fc30_pts = make_points(seed=week_seed + 30, gdf=gdf, center_lat=CENTER_LAT, center_lon=CENTER_LON)

alerts_7 = compute_alerts(now_pts, fc7_pts, k=6)
alerts_30 = compute_alerts(now_pts, fc30_pts, k=6)


# ============================================================
# Build folium map
# ============================================================
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,  # helps marker performance
)

# --- Basemaps ---
folium.TileLayer(CARTO, name="Street map (Carto)", overlay=False, control=True, show=(basemap_choice == "Street map (Carto)")).add_to(m)

folium.TileLayer(
    tiles=ESRI_WORLD_IMAGERY,
    attr="Esri World Imagery",
    name="Satellite (Esri World Imagery)",
    overlay=False,
    control=True,
    show=(basemap_choice == "Satellite (Esri World Imagery)"),
).add_to(m)

# MODIS True Color as a BASEMAP option (REAL)
modis_date = today_utc_yyyy_mm_dd()
folium.TileLayer(
    tiles=make_gibs_modis_truecolor_url(modis_date),
    attr=f"NASA GIBS (MODIS True Color) — {modis_date}",
    name="MODIS True Color (daily) — NASA GIBS",
    overlay=False,
    control=True,
    show=(basemap_choice == "MODIS True Color (daily) — NASA GIBS"),
).add_to(m)

# --- Optional overlay: IMERG precipitation rate (REAL) ---
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

# --- Boundary ---
if show_boundary:
    folium.GeoJson(
        south_sudan_geojson,
        name="South Sudan boundary",
        style_function=lambda x: {"fillOpacity": 0.0, "color": "black", "weight": 3},
        show=True,
    ).add_to(m)


def add_heat(points: List[ScoredPoint], name: str, show: bool) -> None:
    heat_data = [[p.lat, p.lon, p.total] for p in points]
    # HeatMap opacity behavior: use max_opacity to control intensity transparency
    HeatMap(
        heat_data,
        name=name,
        radius=28,
        blur=22,
        min_opacity=0.10,
        max_opacity=float(heat_max_opacity),
        show=show,
    ).add_to(m)


def add_markers(points: List[ScoredPoint], name: str, show: bool, max_n: int = 14) -> None:
    fg = folium.FeatureGroup(name=name, show=show)
    for i, p in enumerate(points[:max_n], start=1):
        top = explain_drivers(p, top_n=2)
        why = ", ".join([f"{k} ({v:.2f})" for k, v in top])

        popup_html = (
            f"<b>{name} #{i}</b><br>"
            f"Total score: {p.total:.2f}<br>"
            f"Rainfall: {p.rainfall:.2f} &nbsp;|&nbsp; Forage: {p.forage:.2f} &nbsp;|&nbsp; Access: {p.access:.2f}<br>"
            f"Lat/Lon: {p.lat:.3f}, {p.lon:.3f}<br>"
            f"<b>Top drivers:</b> {why}"
        )

        folium.CircleMarker(
            location=[p.lat, p.lon],
            radius=int(marker_size),
            color="#1f77b4",      # visible blue outline
            weight=2,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.85,
            tooltip=f"{name} #{i} • total {p.total:.2f}",
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(fg)

    fg.add_to(m)


# --- Model layers split exactly as requested ---
if show_now_heat:
    add_heat(now_pts, "Nowcast heatmap", show=True)
if show_now_markers:
    add_markers(now_pts, "Nowcast markers", show=True)

if show_7_heat:
    add_heat(fc7_pts, "Forecast (7d) heatmap", show=True)
if show_7_markers:
    add_markers(fc7_pts, "Forecast (7d) markers", show=True)

if show_30_heat:
    add_heat(fc30_pts, "Forecast (30d) heatmap", show=True)
if show_30_markers:
    add_markers(fc30_pts, "Forecast (30d) markers", show=True)

folium.LayerControl(collapsed=False).add_to(m)

def nearest_alert(click_lat, click_lon, alerts, tol=0.35):
    """
    Returns (idx, p, delta) for the nearest alert within tol degrees, else (None, None, None).
    alerts = [(p, delta), ...] where p=(lat, lon, score, veg, rain, access)
    """
    best = (None, None, None, 1e9)  # idx, p, delta, dist2
    for idx, (p, delta) in enumerate(alerts, start=1):
        d2 = (p[0] - click_lat) ** 2 + (p[1] - click_lon) ** 2
        if d2 < best[3]:
            best = (idx, p, delta, d2)
    # tol is in degrees; compare squared distance
    if best[0] is not None and best[3] <= (tol ** 2):
        return best[0], best[1], best[2]
    return None, None, None

# ============================================================
# Layout: map + alerts panel
# ============================================================
# ============================================================
# layout: map + alerts panel (clean + working)
# ============================================================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    folium_out = st_folium(
        m,
        width=None,
        height=650,
        key="main_map",
        returned_objects=["center", "zoom", "last_object_clicked"],
    )

# keep map view stable (fixes weird zoom resets)
if isinstance(folium_out, dict):
    if not st.session_state.force_view:
        c = folium_out.get("center")
        z = folium_out.get("zoom")
        if c and "lat" in c and "lng" in c:
            st.session_state.map_center = [float(c["lat"]), float(c["lng"])]
        if z is not None:
            st.session_state.map_zoom = int(z)
    else:
        st.session_state.force_view = False

# click -> select nearest alert (works with AlertItem dataclass)
clicked = (folium_out or {}).get("last_object_clicked")
if clicked and "lat" in clicked and "lng" in clicked:
    clat, clon = float(clicked["lat"]), float(clicked["lng"])

    def _nearest(items):
        best_idx, best_item, best_d2 = None, None, 1e18
        for i, a in enumerate(items, start=1):
            d2 = (a.point.lat - clat) ** 2 + (a.point.lon - clon) ** 2
            if d2 < best_d2:
                best_idx, best_item, best_d2 = i, a, d2
        return best_idx, best_item, best_d2

    i7, a7, d7 = _nearest(alerts_7)
    i30, a30, d30 = _nearest(alerts_30)

    # choose whichever is closer
    if a7 is not None and (a30 is None or d7 <= d30):
        chosen_kind, chosen_idx, chosen = "7-day", i7, a7
    else:
        chosen_kind, chosen_idx, chosen = "30-day", i30, a30

    # store selection for the right panel
    st.session_state.selected_alert = {
        "horizon": chosen_kind,
        "idx": int(chosen_idx),
        "lat": float(chosen.point.lat),
        "lon": float(chosen.point.lon),
        "delta": float(chosen.delta),
        "total": float(chosen.point.total),
        "rain": float(chosen.point.rainfall),
        "forage": float(chosen.point.forage),
        "access": float(chosen.point.access),
        "label": str(chosen.label),
        "drivers": ", ".join([f"{k} ({v:.2f})" for k, v in chosen.top_drivers]),
    }

    # zoom to it
    st.session_state.map_center = [float(chosen.point.lat), float(chosen.point.lon)]
    st.session_state.map_zoom = 9
    st.session_state.force_view = True
    st.rerun()


with right:
    st.header("alerts")
    st.caption("alerts highlight places where forecast suitability increases vs nowcast (delta score).")

    # selected-from-map card (shows after you click a marker)
    if "selected_alert" in st.session_state:
        sa = st.session_state.selected_alert
        st.markdown("### selected from map click")
        st.markdown(f"**{sa['horizon']} alert #{sa['idx']} — {sa['label']}**")
        st.caption(f"lat/lon: {sa['lat']:.3f}, {sa['lon']:.3f}")

        c1, c2 = st.columns(2)
        c1.metric("delta vs nowcast", f"{sa['delta']:+.2f}")
        c2.metric("forecast total", f"{sa['total']:.2f}")

        v1, v2, v3 = st.columns(3)
        v1.metric("rainfall", f"{sa['rain']:.2f}")
        v2.metric("forage", f"{sa['forage']:.2f}")
        v3.metric("access", f"{sa['access']:.2f}")

        st.caption(f"top drivers: {sa['drivers']}")
        st.divider()

    if not show_alerts:
        st.info("turn on 'anomaly alerts' in the sidebar to view alerts.")
    else:
        def render_list(title: str, items: List[AlertItem], key_prefix: str):
            st.subheader(title)
            for idx, a in enumerate(items, start=1):
                st.markdown(f"**{idx}. {a.label}**")
                st.caption(f"lat/lon: {a.point.lat:.3f}, {a.point.lon:.3f}")

                c1, c2 = st.columns(2)
                c1.metric("delta", f"{a.delta:+.2f}")
                c2.metric("total", f"{a.point.total:.2f}")

                v1, v2, v3 = st.columns(3)
                v1.metric("rain", f"{a.point.rainfall:.2f}")
                v2.metric("forage", f"{a.point.forage:.2f}")
                v3.metric("access", f"{a.point.access:.2f}")

                st.caption("drivers: " + ", ".join([f"{k} ({v:.2f})" for k, v in a.top_drivers]))

                b1, b2 = st.columns(2)
                with b1:
                    if st.button(f"zoom to {title} alert #{idx}", key=f"{key_prefix}_z_{idx}"):
                        st.session_state.map_center = [a.point.lat, a.point.lon]
                        st.session_state.map_zoom = 9
                        st.session_state.force_view = True
                        st.rerun()
                with b2:
                    if st.button("zoom closer", key=f"{key_prefix}_zz_{idx}"):
                        st.session_state.map_center = [a.point.lat, a.point.lon]
                        st.session_state.map_zoom = 11
                        st.session_state.force_view = True
                        st.rerun()

                st.divider()

        render_list("7-day", alerts_7, "a7")
        render_list("30-day", alerts_30, "a30")
