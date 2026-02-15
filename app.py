import math
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Dict, Any

import json
import urllib.request
import urllib.parse

import numpy as np
import geopandas as gpd
import streamlit as st
import pandas as pd
import pydeck as pdk

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
    """
    All inputs are REAL (Open-Meteo).
    Converts to 0..1 suitability scores:
      - water = rainfall (more is better up to a cap)
      - comfort = temperature comfort (cooler is better past a threshold)
    """
    # rainfall scoring (mm)
    water_7 = normalize(obs_rain_7, 0.0, 70.0)
    water_fc7 = normalize(fc_rain_7, 0.0, 70.0)
    water_30 = normalize(obs_rain_30, 0.0, 200.0)

    # temperature comfort (C) — higher => worse comfort
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
    """
    mode: "now" | "fc7" | "trend30"
    Returns 2 drivers (score contributions) so reasons differ per point.
    """
    if mode == "now":
        drivers = [("Rainfall (past 7d)", features["water_7"]), ("Heat comfort (past 7d)", features["comfort_7"])]
    elif mode == "fc7":
        drivers = [
            ("Forecast rainfall (next 7d)", features["water_fc7"]),
            ("Forecast heat comfort (next 7d)", features["comfort_fc7"]),
        ]
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
    """
    REAL-ONLY points from Open-Meteo.
    - now_total = past 7 days observed
    - fc7_total = next 7 days forecast
    - trend30_total = past 30 days observed
    """
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

            # inside boundary
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

            pts.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "now_total": now_total,
                    "fc7_total": fc7_total,
                    "trend30_total": trend30_total,
                    "features": features,
                }
            )

    # rank by NOWCAST by default (what you see first)
    pts.sort(key=lambda p: p["now_total"], reverse=True)
    return pts[:keep_n]


def compute_alerts_7day(points: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Alerts = biggest change from now -> next 7 days (REAL).
    """
    scored = []
    for p in points:
        delta = float(p["fc7_total"] - p["now_total"])
        scored.append((p, delta))
    scored.sort(key=lambda t: t[1], reverse=True)

    out = []
    for p, delta in scored[:k]:
        f = p["features"]
        out.append(
            {
                "lat": p["lat"],
                "lon": p["lon"],
                "delta": delta,
                "now_total": p["now_total"],
                "fc7_total": p["fc7_total"],
                "trend30_total": p["trend30_total"],
                "features": f,
                "label": label_from_delta(delta),
                "reasons": top_reasons(f, "fc7"),
            }
        )
    return out


def nearest_alert(lat: float, lon: float, alerts: List[Dict[str, Any]], tol_deg: float = 0.7):
    best = None
    best_d2 = 1e18
    for idx, a in enumerate(alerts, start=1):
        d2 = (a["lat"] - lat) ** 2 + (a["lon"] - lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (idx, a)
    if best and best_d2 <= (tol_deg ** 2):
        return best[0], best[1]
    return None, None


# ============================================================
# UI header + boundary
# ============================================================
st.title(TITLE)
st.caption(CAPTION)

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__
centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

# Session state (map view + selections)
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None
if "selected_alert_idx" not in st.session_state:
    st.session_state.selected_alert_idx = None  # 1..6

if "pending_zoom_to_alert" not in st.session_state:
    st.session_state.pending_zoom_to_alert = False

if "last_click_key" not in st.session_state:
    st.session_state.last_click_key = None

if "map_booted" not in st.session_state:
    st.session_state.map_booted = False

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
        st.session_state.view_lat = float(a["lat"])
        st.session_state.view_lon = float(a["lon"])
        st.session_state.view_zoom = 9  # or 11 for closer
        st.rerun()
with c2:
    if st.button("Zoom to S. Sudan"):
        st.session_state.view_lat = float(a["lat"])
        st.session_state.view_lon = float(a["lon"])
        st.session_state.view_zoom = 9  # or 11 for closer
        st.rerun()


# ============================================================
# Build REAL points + alerts (Open-Meteo)
# ============================================================
with st.spinner("Fetching real weather (Open-Meteo) for sample points..."):
    points = build_points_real(gdf, step_deg=step_deg, keep_n=keep_n, max_api_points=max_api_points)

alerts_7 = compute_alerts_7day(points, k=6)


# ============================================================
# PyDeck map (NO blinking like Folium)
# ============================================================

# --- Build DataFrame for layers ---
df = pd.DataFrame(points)
df["id"] = np.arange(len(df))  # stable id for selection
df["delta_fc7_now"] = df["fc7_total"] - df["now_total"]

# Build alerts df (for highlighting)
alerts_df = pd.DataFrame(alerts_7)
if not alerts_df.empty:
    alerts_df["alert_idx"] = np.arange(1, len(alerts_df) + 1)

# --- Basemap styles (Mapbox styles; no Esri here) ---
MAP_STYLE_STREET = "mapbox://styles/mapbox/light-v10"
MAP_STYLE_SAT = "mapbox://styles/mapbox/satellite-v9"

# NASA GIBS tiles
MODIS_TILE = make_gibs_modis_truecolor_url(modis_date)
IMERG_TILE = make_gibs_imerg_url()

# --- Session state for view ---
if "view_lat" not in st.session_state:
    st.session_state.view_lat = CENTER_LAT
if "view_lon" not in st.session_state:
    st.session_state.view_lon = CENTER_LON
if "view_zoom" not in st.session_state:
    st.session_state.view_zoom = 6

# If user clicked an alert button earlier, we already set these:
view_state = pdk.ViewState(
    latitude=float(st.session_state.view_lat),
    longitude=float(st.session_state.view_lon),
    zoom=float(st.session_state.view_zoom),
    pitch=0,
    bearing=0,
)

# --- Layers ---
layers = []

# Boundary
if show_boundary:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=south_sudan_geojson,
            stroked=True,
            filled=False,
            get_line_color=[0, 0, 0, 200],
            line_width_min_pixels=2,
            pickable=False,
        )
    )

# MODIS as a basemap option (tile layer)
# If MODIS is selected as "basemap", we lay MODIS tiles on top of a neutral street style.
if basemap_choice == "MODIS True Color (daily) — NASA GIBS":
    layers.append(
        pdk.Layer(
            "TileLayer",
            data=MODIS_TILE,
            min_zoom=0,
            max_zoom=9,
            tile_size=256,
            opacity=1.0,
        )
    )

# IMERG overlay
if show_imerg:
    layers.append(
        pdk.Layer(
            "TileLayer",
            data=IMERG_TILE,
            min_zoom=0,
            max_zoom=8,
            tile_size=256,
            opacity=float(overlay_opacity),
        )
    )

# Heatmaps (weights are real scores)
def add_heat(df_in: pd.DataFrame, value_col: str, enabled: bool, name: str):
    if not enabled or df_in.empty:
        return
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=df_in,
            get_position=["lon", "lat"],
            get_weight=value_col,
            radius_pixels=60,   # feel free to tweak
            intensity=1.0,
            threshold=0.05,
        )
    )

# Markers
def add_markers(df_in: pd.DataFrame, value_col: str, enabled: bool, name: str):
    if not enabled or df_in.empty:
        return
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_in,
            get_position=["lon", "lat"],
            get_radius=int(marker_size) * 800,  # meters-ish
            radius_min_pixels=4,
            radius_max_pixels=30,
            get_fill_color=[31, 119, 180, 180],
            get_line_color=[31, 119, 180, 255],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    )

# Split layers exactly like you wanted
add_heat(df, "now_total", show_now_heat, "Nowcast heat")
add_markers(df, "now_total", show_now_markers, "Nowcast markers")

add_heat(df, "fc7_total", show_fc7_heat, "Forecast heat")
add_markers(df, "fc7_total", show_fc7_markers, "Forecast markers")

add_heat(df, "trend30_total", show_trend30_heat, "30-day trend heat")
add_markers(df, "trend30_total", show_trend30_markers, "30-day trend markers")

# Highlight the selected alert (draw a bigger red dot)
sel_idx = st.session_state.get("selected_alert_idx")
if sel_idx and not alerts_df.empty:
    sel_row = alerts_df[alerts_df["alert_idx"] == int(sel_idx)]
    if not sel_row.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=sel_row,
                get_position=["lon", "lat"],
                get_radius=25000,
                radius_min_pixels=10,
                get_fill_color=[220, 0, 0, 180],
                get_line_color=[220, 0, 0, 255],
                line_width_min_pixels=2,
                pickable=False,
            )
        )

# Tooltip (REAL reasons + raw values)
tooltip = {
    "html": """
    <b>Score:</b> {now_total}<br/>
    <b>Forecast:</b> {fc7_total}<br/>
    <b>30-day trend:</b> {trend30_total}<br/>
    <b>Δ (fc7-now):</b> {delta_fc7_now}<br/>
    <hr/>
    <b>Raw REAL (Open-Meteo):</b><br/>
    Past 7d rain: {features.obs_rain_7} mm<br/>
    Past 7d temp: {features.obs_tmean_7} °C<br/>
    Next 7d rain: {features.fc_rain_7} mm<br/>
    Next 7d temp: {features.fc_tmean_7} °C<br/>
    Past 30d rain: {features.obs_rain_30} mm<br/>
    Past 30d temp: {features.obs_tmean_30} °C
    """,
    "style": {"backgroundColor": "white", "color": "black"},
}

# Map style choice
if basemap_choice == "Satellite (Esri World Imagery)":
    # Closest non-Esri option: Mapbox satellite (still real imagery, different provider).
    map_style = MAP_STYLE_SAT
elif basemap_choice == "MODIS True Color (daily) — NASA GIBS":
    map_style = MAP_STYLE_STREET
else:
    map_style = MAP_STYLE_STREET

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=map_style,
    tooltip=tooltip,
)

# ============================================================
# Layout
# ============================================================
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    # If your Streamlit supports selection events, this enables click->select.
    # If it errors, comment out on_select/selection_mode and you still get tooltips + alert buttons.
    try:
        event = st.pydeck_chart(
            deck,
            use_container_width=True,
            height=650,
            on_select="rerun",
            selection_mode="single-object",
        )
    except TypeError:
        event = None
        st.pydeck_chart(deck, use_container_width=True, height=650)

# Marker click -> select nearest alert + highlight it on the right
# (We avoid changing view unless it's an alert button; that keeps it smooth.)
if isinstance(event, dict):
    sel = (event.get("selection") or {}).get("objects") or []
    if sel:
        obj = sel[0]
        # obj should include picked row fields
        lat = float(obj.get("lat", 0))
        lon = float(obj.get("lon", 0))

        idx, a = nearest_alert(lat, lon, alerts_7, tol_deg=0.7)
        if idx is not None:
            st.session_state.selected_alert_idx = idx
            st.session_state.selected_alert = {"idx": idx, **a}

# ============================================================
# Right panel
# ============================================================
with right:
    st.header("Alerts")
    st.caption("REAL: Alerts = biggest change from nowcast → next 7 days (Open-Meteo).")

    sel = st.session_state.selected_alert
    if sel:
        st.subheader("Selected (from map click)")
        st.caption(f"Alert #{sel['idx']} • Lat/Lon: {sel['lat']:.3f}, {sel['lon']:.3f}")

        c1, c2 = st.columns(2)
        c1.metric("Delta (fc7 - now)", f"{sel['delta']:+.2f}")
        c2.metric("Forecast score", f"{sel['fc7_total']:.2f}")

        f = sel["features"]
        st.write("**Reasons (scores 0..1):** " + ", ".join([f"{k} ({v:.2f})" for k, v in sel["reasons"]]))
        st.write(
            f"**Raw REAL:** past7 rain {f['obs_rain_7']:.1f}mm, past7 temp {f['obs_tmean_7']:.1f}°C • "
            f"next7 rain {f['fc_rain_7']:.1f}mm, next7 temp {f['fc_tmean_7']:.1f}°C"
        )
        st.divider()

    st.subheader("7-day alerts (change)")
    if not show_alerts:
        st.info("Turn on **Alerts (7-day change)** in the sidebar.")
    else:
        for i, a in enumerate(alerts_7, start=1):
            is_selected = (st.session_state.get("selected_alert_idx") == i)

            reasons_txt = ", ".join([f"{k} ({v:.2f})" for k, v in a["reasons"]])

            st.markdown(f"{'✅ ' if is_selected else ''}**{i}. {a['label']}**")
            st.caption(f"Lat/Lon: {a['lat']:.3f}, {a['lon']:.3f}")

            m1, m2 = st.columns(2)
            m1.metric("Delta", f"{a['delta']:+.2f}")
            m2.metric("Forecast score", f"{a['fc7_total']:.2f}")

            st.caption("Reasons: " + reasons_txt)

            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"Zoom to alert #{i}", key=f"z_{i}"):
                    st.session_state.selected_alert_idx = i
                    st.session_state.selected_alert = {"idx": i, **a}
                    st.session_state.view_lat = float(a["lat"])
                    st.session_state.view_lon = float(a["lon"])
                    st.session_state.view_zoom = 9  # or 11 for closer
                    st.session_state.pending_zoom_to_alert = True
                    st.rerun()
                    
            with b2:
                if st.button("Zoom closer", key=f"zz_{i}"):
                    st.session_state.selected_alert_idx = i
                    st.session_state.selected_alert = {"idx": i, **a}
                    st.session_state.view_lat = float(a["lat"])
                    st.session_state.view_lon = float(a["lon"])
                    st.session_state.view_zoom = 11  # or 11 for closer
                    st.session_state.pending_zoom_to_alert = True
                    st.rerun()

            st.divider()

    st.info(
        f"**Real data used:** Open-Meteo daily precip/temp (past 30d + next 7d), "
        f"NASA GIBS MODIS True Color (date: {modis_date}) and optional IMERG overlay tiles."
    )
