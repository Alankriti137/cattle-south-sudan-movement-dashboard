import math
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Dict, Any

import json
import urllib.request
import urllib.parse

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")

TITLE = "Cattle in South Sudan Movement Dashboard"
CAPTION = "REAL drivers only: Open-Meteo (rain/temp) + NASA GIBS (imagery/IMERG tiles)"
BOUNDARY_PATH = "data/south_sudan.geojson.json"

# Basemaps / tiles (REAL)
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


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def safe_sum(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs)) if xs else 0.0


def make_gibs_modis_truecolor_url(date_str: str) -> str:
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_MODIS_TRUECOLOR_LAYER}/default/{date_str}/"
        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    )


def make_gibs_imerg_url() -> str:
    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{GIBS_IMERG_LAYER}/default/default/"
        "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    )


# ============================================================
# REAL weather fetch (Open-Meteo) — cached
# ============================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_open_meteo_daily(lat: float, lon: float, past_days: int = 30, forecast_days: int = 7) -> dict:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": "precipitation_sum,temperature_2m_mean",
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
        "timezone": "UTC",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def score_from_real_weather(
    obs_rain_7: float,
    obs_tmean_7: float,
    fc_rain_7: float,
    fc_tmean_7: float,
    obs_rain_30: float,
    obs_tmean_30: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    # rainfall scoring (mm)
    water_7 = normalize(obs_rain_7, 0.0, 70.0)
    water_fc7 = normalize(fc_rain_7, 0.0, 70.0)
    water_30 = normalize(obs_rain_30, 0.0, 200.0)

    # temperature comfort (C) — hotter => worse comfort
    comfort_7 = 1.0 - normalize(obs_tmean_7, 28.0, 42.0)
    comfort_fc7 = 1.0 - normalize(fc_tmean_7, 28.0, 42.0)
    comfort_30 = 1.0 - normalize(obs_tmean_30, 28.0, 42.0)

    now_total = clamp01(0.60 * water_7 + 0.40 * comfort_7)
    fc7_total = clamp01(0.60 * water_fc7 + 0.40 * comfort_fc7)
    trend30_total = clamp01(0.60 * water_30 + 0.40 * comfort_30)

    features = {
        "obs_rain_7": float(obs_rain_7),
        "obs_tmean_7": float(obs_tmean_7),
        "fc_rain_7": float(fc_rain_7),
        "fc_tmean_7": float(fc_tmean_7),
        "obs_rain_30": float(obs_rain_30),
        "obs_tmean_30": float(obs_tmean_30),
        "water_7": float(water_7),
        "comfort_7": float(comfort_7),
        "water_fc7": float(water_fc7),
        "comfort_fc7": float(comfort_fc7),
        "water_30": float(water_30),
        "comfort_30": float(comfort_30),
    }
    return now_total, fc7_total, trend30_total, features


def top_reasons(features: Dict[str, float], mode: str) -> List[Tuple[str, float]]:
    if mode == "now":
        drivers = [("Rainfall (past 7d)", features["water_7"]), ("Heat comfort (past 7d)", features["comfort_7"])]
    elif mode == "fc7":
        drivers = [("Forecast rainfall (next 7d)", features["water_fc7"]), ("Forecast heat comfort (next 7d)", features["comfort_fc7"])]
    else:
        drivers = [("Rainfall (past 30d)", features["water_30"]), ("Heat comfort (past 30d)", features["comfort_30"])]

    drivers.sort(key=lambda t: t[1], reverse=True)
    return drivers[:2]


def label_from_delta(delta: float) -> str:
    if delta >= 0.30:
        return "Strong improvement expected"
    if delta >= 0.15:
        return "Moderate improvement expected"
    if delta >= 0.05:
        return "Slight improvement expected"
    if delta <= -0.15:
        return "Conditions worsening"
    return "Small change"


def build_points_real(gdf: gpd.GeoDataFrame, step_deg: float, keep_n: int, max_api_points: int) -> List[Dict[str, Any]]:
    minx, miny, maxx, maxy = gdf.total_bounds
    poly = gdf.unary_union

    pts: List[Dict[str, Any]] = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    for lat in lats:
        for lon in lons:
            if len(pts) >= max_api_points:
                break

            lat = float(lat)
            lon = float(lon)

            try:
                if not poly.contains(gpd.points_from_xy([lon], [lat])[0]):
                    continue
            except Exception:
                pass

            wx = fetch_open_meteo_daily(lat, lon, past_days=30, forecast_days=7)
            daily = wx.get("daily", {}) or {}
            precip = daily.get("precipitation_sum", []) or []
            tmean = daily.get("temperature_2m_mean", []) or []

            past_len = min(len(precip), 30)
            obs_precip = precip[:past_len]
            obs_tmean = tmean[:past_len]
            fc_precip = precip[past_len:past_len + 7]
            fc_tmean = tmean[past_len:past_len + 7]

            obs_rain_7 = safe_sum(obs_precip[-7:])
            obs_tmean_7 = safe_mean(obs_tmean[-7:])
            fc_rain_7 = safe_sum(fc_precip[:7])
            fc_tmean_7 = safe_mean(fc_tmean[:7])

            obs_rain_30 = safe_sum(obs_precip[-30:])
            obs_tmean_30 = safe_mean(obs_tmean[-30:])

            now_total, fc7_total, trend30_total, features = score_from_real_weather(
                obs_rain_7, obs_tmean_7,
                fc_rain_7, fc_tmean_7,
                obs_rain_30, obs_tmean_30,
            )

            pts.append({
                "lat": lat,
                "lon": lon,
                "now_total": now_total,
                "fc7_total": fc7_total,
                "trend30_total": trend30_total,
                "features": features,
            })

    pts.sort(key=lambda p: p["now_total"], reverse=True)
    return pts[:keep_n]


def compute_alerts_7day(points: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    scored = []
    for p in points:
        delta = float(p["fc7_total"] - p["now_total"])
        scored.append((p, delta))
    scored.sort(key=lambda t: t[1], reverse=True)

    out = []
    for p, delta in scored[:k]:
        f = p["features"]
        out.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "delta": delta,
            "now_total": p["now_total"],
            "fc7_total": p["fc7_total"],
            "trend30_total": p["trend30_total"],
            "features": f,
            "label": label_from_delta(delta),
            "reasons": top_reasons(f, "fc7"),
        })
    return out


def nearest_alert_idx(lat: float, lon: float, alerts: List[Dict[str, Any]], tol_deg: float = 0.7) -> Optional[int]:
    best_idx = None
    best_d2 = 1e18
    for idx, a in enumerate(alerts, start=1):
        d2 = (a["lat"] - lat) ** 2 + (a["lon"] - lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = idx
    if best_idx is not None and best_d2 <= (tol_deg ** 2):
        return best_idx
    return None


# ============================================================
# UI header + boundary
# ============================================================
st.title(TITLE)
st.caption(CAPTION)

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__
centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

# Session state
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
if "selected_alert_idx" not in st.session_state:
    st.session_state.selected_alert_idx = None
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Layers")

basemap_choice = st.sidebar.radio(
    "Basemap",
    ["Street map (Carto)", "Satellite (Esri World Imagery)", "MODIS True Color (daily) — NASA GIBS"],
    index=0,
)

modis_day_choice = st.sidebar.radio(
    "MODIS day",
    ["Today", "Yesterday", "2 days ago"],
    index=1,
    horizontal=True,
)

days_back = {"Today": 0, "Yesterday": 1, "2 days ago": 2}[modis_day_choice]
modis_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

st.sidebar.divider()
st.sidebar.subheader("Real public data overlay (NASA GIBS)")
show_imerg = st.sidebar.checkbox("IMERG Precipitation Rate (30-min) — NASA GIBS", value=False)
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.70, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Real model layers (from Open-Meteo)")
show_boundary = st.sidebar.checkbox("South Sudan boundary", value=True)

show_now_heat = st.sidebar.checkbox("Nowcast (past 7d) heatmap", value=True)
show_now_markers = st.sidebar.checkbox("Nowcast (past 7d) markers", value=True)

show_fc7_heat = st.sidebar.checkbox("Forecast (next 7d) heatmap", value=False)
show_fc7_markers = st.sidebar.checkbox("Forecast (next 7d) markers", value=False)

show_trend30_heat = st.sidebar.checkbox("30-day trend heatmap", value=False)
show_trend30_markers = st.sidebar.checkbox("30-day trend markers", value=False)

show_alerts = st.sidebar.checkbox("Alerts (7-day change)", value=True)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", 6, 40, 18, 1)
heat_opacity = st.sidebar.slider("Heat opacity", 0.10, 0.95, 0.55, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Performance (Open-Meteo calls)")
step_deg = st.sidebar.slider("Grid spacing (degrees)", 0.6, 1.6, 0.9, 0.1)
max_api_points = st.sidebar.slider("Max API points per run", 10, 60, 25, 5)
keep_n = st.sidebar.slider("Keep top points", 10, 50, 30, 5)

st.sidebar.divider()
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Reset view"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 6
        st.session_state.selected_alert_idx = None
        st.session_state.selected_alert = None
        st.rerun()
with c2:
    if st.button("Zoom to S. Sudan"):
        st.session_state.map_center = [CENTER_LAT, CENTER_LON]
        st.session_state.map_zoom = 7
        st.rerun()


# ============================================================
# Build REAL points + alerts (Open-Meteo)
# ============================================================
with st.spinner("Fetching real weather (Open-Meteo) for sample points..."):
    points = build_points_real(gdf, step_deg=step_deg, keep_n=keep_n, max_api_points=max_api_points)

alerts_7 = compute_alerts_7day(points, k=6)


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
folium.TileLayer(CARTO, name="Street map (Carto)", overlay=False, control=True, show=(basemap_choice == "Street map (Carto)")).add_to(m)
folium.TileLayer(tiles=ESRI_WORLD_IMAGERY, attr="Esri World Imagery", name="Satellite (Esri World Imagery)",
                 overlay=False, control=True, show=(basemap_choice == "Satellite (Esri World Imagery)")).add_to(m)
folium.TileLayer(tiles=make_gibs_modis_truecolor_url(modis_date), attr=f"NASA GIBS (MODIS True Color) — {modis_date}",
                 name="MODIS True Color (daily) — NASA GIBS", overlay=False, control=True,
                 show=(basemap_choice == "MODIS True Color (daily) — NASA GIBS")).add_to(m)

# IMERG overlay
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


def add_heat(points_dicts: List[Dict[str, Any]], value_key: str, name: str, show: bool):
    heat_data = [[p["lat"], p["lon"], float(p[value_key])] for p in points_dicts]
    HeatMap(
        heat_data,
        name=name,
        radius=28,
        blur=22,
        min_opacity=0.10,
        max_opacity=float(heat_opacity),
        show=show,
    ).add_to(m)


def add_markers(points_dicts: List[Dict[str, Any]], value_key: str, name: str, reason_mode: str, show: bool, max_n: int = 16):
    fg = folium.FeatureGroup(name=name, show=show)
    for i, p in enumerate(points_dicts[:max_n], start=1):
        v = float(p[value_key])
        f = p["features"]
        reasons = top_reasons(f, reason_mode)
        why = ", ".join([f"{k} ({val:.2f})" for k, val in reasons])

        popup_html = (
            f"<b>{name} #{i}</b><br>"
            f"Score: {v:.2f}<br>"
            f"Lat/Lon: {p['lat']:.3f}, {p['lon']:.3f}<br>"
            f"<b>Reasons:</b> {why}<br><br>"
            f"<b>Raw (REAL):</b><br>"
            f"Past 7d rain: {f['obs_rain_7']:.1f}mm<br>"
            f"Past 7d mean temp: {f['obs_tmean_7']:.1f}°C<br>"
            f"Next 7d rain: {f['fc_rain_7']:.1f}mm<br>"
            f"Next 7d mean temp: {f['fc_tmean_7']:.1f}°C<br>"
            f"Past 30d rain: {f['obs_rain_30']:.1f}mm<br>"
            f"Past 30d mean temp: {f['obs_tmean_30']:.1f}°C"
        )

        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=int(marker_size),
            weight=2,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.85,
            tooltip=f"{name} #{i} • {v:.2f}",
            popup=folium.Popup(popup_html, max_width=420),
        ).add_to(fg)

    fg.add_to(m)


# Layers
if show_now_heat:
    add_heat(points, "now_total", "Nowcast (past 7d) heatmap", show=True)
if show_now_markers:
    add_markers(points, "now_total", "Nowcast (past 7d) markers", "now", show=True)

if show_fc7_heat:
    add_heat(points, "fc7_total", "Forecast (next 7d) heatmap", show=True)
if show_fc7_markers:
    add_markers(points, "fc7_total", "Forecast (next 7d) markers", "fc7", show=True)

if show_trend30_heat:
    add_heat(points, "trend30_total", "30-day trend heatmap", show=True)
if show_trend30_markers:
    add_markers(points, "trend30_total", "30-day trend markers", "trend30", show=True)

folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# Layout (NO map persistence = NO blinking)
# ============================================================
left, right = st.columns([2.2, 1.0], gap="large")

# ---- Build a fast lookup so clicks select the right alert ----
# key = rounded lat/lon -> alert index (1..k)
alert_lookup = {(round(a["lat"], 4), round(a["lon"], 4)): i for i, a in enumerate(alerts_7, start=1)}

def pick_alert_from_click(clat: float, clon: float):
    key = (round(clat, 4), round(clon, 4))
    if key in alert_lookup:
        idx = alert_lookup[key]
        return idx, alerts_7[idx - 1]
    # fallback: nearest
    idx, a = nearest_alert(clat, clon, alerts_7, tol_deg=0.7)
    return idx, a

# ---- Add ALERT markers (so clicking is reliable) ----
if show_alerts:
    fg_alerts = folium.FeatureGroup(name="Alerts (7-day change)", show=True)
    for i, a in enumerate(alerts_7, start=1):
        folium.CircleMarker(
            location=[a["lat"], a["lon"]],
            radius=10,
            weight=3,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.9,
            tooltip=f"Alert #{i}: {a['label']} (Δ {a['delta']:+.2f})",
        ).add_to(fg_alerts)
    fg_alerts.add_to(m)

with left:
    map_out = st_folium(
        m,
        width=None,
        height=650,
        key="main_map",
        returned_objects=["last_object_clicked"],  # keep it minimal (less flicker)
    )

# ---- Click handling (NO st.rerun here!) ----
clicked = (map_out or {}).get("last_object_clicked")
if clicked and "lat" in clicked and "lng" in clicked:
    clat, clon = float(clicked["lat"]), float(clicked["lng"])
    idx, a = pick_alert_from_click(clat, clon)
    if idx is not None:
        st.session_state.selected_alert_idx = idx
        st.session_state.selected_alert = {"idx": idx, **a}
        # OPTIONAL: if you want zoom-on-click, do it ONLY by setting state (no rerun call)
        st.session_state.map_center = [a["lat"], a["lon"]]
        st.session_state.map_zoom = 9

# ============================================================
# Right panel
# ============================================================
with right:
    st.header("Alerts")
    st.caption("REAL inputs: Open-Meteo (past 30d + next 7d). Alerts = biggest change now → next 7 days.")

    sel = st.session_state.get("selected_alert")
    if sel:
        st.subheader("Selected alert")
        st.markdown(f"**✅ Alert #{sel['idx']}: {sel['label']}**")
        st.caption(f"Lat/Lon: {sel['lat']:.3f}, {sel['lon']:.3f}")

        c1, c2 = st.columns(2)
        c1.metric("Delta (fc7 - now)", f"{sel['delta']:+.2f}")
        c2.metric("Forecast score", f"{sel['fc7_total']:.2f}")

        f = sel["features"]

        # show weighted contributions so “reasons” differ more
        rain_contrib = 0.60 * f["water_fc7"]
        comfort_contrib = 0.40 * f["comfort_fc7"]
        reasons = [
            ("Forecast rainfall contribution", rain_contrib),
            ("Forecast heat comfort contribution", comfort_contrib),
        ]
        reasons.sort(key=lambda t: t[1], reverse=True)

        st.write("**Top reasons (weighted):** " + ", ".join([f"{k} ({v:.2f})" for k, v in reasons]))
        st.write(
            f"**Raw REAL:** next7 rain {f['fc_rain_7']:.1f}mm, next7 temp {f['fc_tmean_7']:.1f}°C • "
            f"past7 rain {f['obs_rain_7']:.1f}mm, past7 temp {f['obs_tmean_7']:.1f}°C"
        )
        st.divider()

    st.subheader("7-day alerts (change)")
    if not show_alerts:
        st.info("Turn on **Alerts (7-day change)** in the sidebar.")
    else:
        for i, a in enumerate(alerts_7, start=1):
            is_selected = (st.session_state.get("selected_alert_idx") == i)
            st.markdown(f"{'✅ ' if is_selected else ''}**{i}. {a['label']}**")
            st.caption(f"Lat/Lon: {a['lat']:.3f}, {a['lon']:.3f}")

            m1, m2 = st.columns(2)
            m1.metric("Delta", f"{a['delta']:+.2f}")
            m2.metric("Forecast score", f"{a['fc7_total']:.2f}")

            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"Select #{i}", key=f"sel_{i}"):
                    st.session_state.selected_alert_idx = i
                    st.session_state.selected_alert = {"idx": i, **a}
            with b2:
                if st.button(f"Zoom #{i}", key=f"zoom_{i}"):
                    st.session_state.selected_alert_idx = i
                    st.session_state.selected_alert = {"idx": i, **a}
                    st.session_state.map_center = [a["lat"], a["lon"]]
                    st.session_state.map_zoom = 9
            st.divider()

    st.info(
        f"**Real data used:** Open-Meteo daily precip/temp + NASA GIBS MODIS True Color (date: {modis_date}) "
        f"+ optional IMERG overlay tiles."
    )
