import math
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")

st.title("Cattle in South Sudan Movement Dashboard")
st.caption("Near-real-time (weekly) cattle movement suitability, forecasts, and alerts")

# =========================================================
# LOAD SOUTH SUDAN BOUNDARY (YOUR FILE)
# =========================================================
BOUNDARY_PATH = "data/south_sudan.geojson.json"

@st.cache_data
def load_boundary(path: str):
    gdf = gpd.read_file(path)
    # force to EPSG:4326
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    return gdf

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__

centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Layers")

# Basemap choice (fixes “language” surprises by keeping ONE basemap)
basemap = st.sidebar.radio(
    "Basemap",
    options=["Street map (Carto)", "Satellite (Esri World Imagery)"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Model layers (your dashboard outputs)")
show_nowcast = st.sidebar.checkbox("Nowcast (current)", value=True)
show_fc7 = st.sidebar.checkbox("Forecast (7 days)", value=False)
show_fc30 = st.sidebar.checkbox("Forecast (30 days)", value=False)
show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)

st.sidebar.divider()
st.sidebar.subheader("Real public data overlays (auto-updating tiles)")
show_modis = st.sidebar.checkbox("Satellite: MODIS True Color (daily) — overlay", value=False)
show_imerg = st.sidebar.checkbox("Weather: IMERG Precipitation (30-min) — overlay", value=False)

overlay_opacity = st.sidebar.slider("Overlay opacity", 0.10, 0.95, 0.65, 0.05)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", min_value=8, max_value=30, value=16, step=1)
heat_opacity = st.sidebar.slider("Heat opacity", min_value=0.15, max_value=0.90, value=0.55, step=0.05)

st.sidebar.divider()
st.sidebar.caption(
    "Note: MODIS/IMERG are real NASA tile overlays (visual layers). "
    "The hotspot scoring below is a v1 heuristic (prototype) that you can replace with a real model later."
)

# =========================================================
# SIMPLE “ENGINE” (V1 HEURISTIC) — GENERATES HOTSPOTS + FORECASTS
# =========================================================
minx, miny, maxx, maxy = gdf.total_bounds

def make_points(seed: int, step_deg: float = 0.6):
    rng = np.random.default_rng(seed)
    points = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    # build a fast polygon object once
    poly = gdf.unary_union

    for lat in lats:
        for lon in lons:
            jlat = lat + rng.uniform(-0.12, 0.12)
            jlon = lon + rng.uniform(-0.12, 0.12)

            try:
                pt = gpd.points_from_xy([jlon], [jlat])[0]
                if not poly.contains(pt):
                    continue
            except Exception:
                pass

            # v1 drivers (prototype signals)
            veg = 1.0 - min(1.0, abs(jlat - CENTER_LAT) / 6.0)   # central lat band proxy
            rain = rng.random()                                  # placeholder proxy
            access = 1.0 - min(1.0, (abs(jlat - CENTER_LAT) + abs(jlon - CENTER_LON)) / 10.0)

            score = 0.45 * veg + 0.35 * rain + 0.20 * access
            points.append((jlat, jlon, float(score), float(veg), float(rain), float(access)))

    points.sort(key=lambda x: x[2], reverse=True)
    return points[:40]  # keep a few more; we’ll show fewer markers

# Seed changes weekly (so your “nowcast” changes week-to-week)
week_seed = int(datetime.now(timezone.utc).strftime("%Y%U"))

nowcast_pts = make_points(seed=week_seed)
fc7_pts = make_points(seed=week_seed + 7)
fc30_pts = make_points(seed=week_seed + 30)

def top_alerts(now_pts, forecast_pts, k=8):
    # alert if a location gets hotter in forecast vs now
    alerts = []
    now_index = {(round(p[0], 1), round(p[1], 1)): p for p in now_pts}
    for p in forecast_pts:
        key = (round(p[0], 1), round(p[1], 1))
        delta = p[2] - now_index[key][2] if key in now_index else p[2]
        alerts.append((p, float(delta)))

    alerts.sort(key=lambda x: x[1], reverse=True)
    return alerts[:k]

alerts_7 = top_alerts(nowcast_pts, fc7_pts, k=8)
alerts_30 = top_alerts(nowcast_pts, fc30_pts, k=8)

def explain_reason(p, delta):
    """
    Makes alert explanations DIFFERENT per alert:
    - Uses delta category + strongest drivers (veg/rain/access)
    - Mentions real overlays when turned on (MODIS/IMERG) so UI makes sense
    """
    veg, rain, access = p[3], p[4], p[5]

    # strongest drivers
    drivers = sorted(
        [("forage/vegetation", veg), ("rainfall", rain), ("access/centrality", access)],
        key=lambda x: x[1],
        reverse=True
    )
    top1, top2 = drivers[0], drivers[1]

    # delta label
    if delta > 0.75:
        shift = "new hotspot likely forming"
    elif delta > 0.45:
        shift = "route deviation / redistribution"
    else:
        shift = "moderate increase"

    overlay_notes = []
    if show_imerg:
        overlay_notes.append("IMERG precipitation overlay is ON (real)")
    if show_modis:
        overlay_notes.append("MODIS true-color overlay is ON (real)")
    overlay_text = (" | " + ", ".join(overlay_notes)) if overlay_notes else ""

    return (
        f"{shift}: strongest signals = {top1[0]} ({top1[1]:.2f}), {top2[0]} ({top2[1]:.2f})"
        f"{overlay_text}"
    )

# =========================================================
# MAP VIEW STATE (FIXES WEIRD ZOOM / BLINKING)
# =========================================================
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6
if "pending_zoom" not in st.session_state:
    st.session_state.pending_zoom = False

# =========================================================
# BUILD FOLIUM MAP
# =========================================================
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
    max_zoom=18,
)

# basemap (only ONE active basemap so labels don’t “change language”)
if basemap == "Street map (Carto)":
    folium.TileLayer("cartodbpositron", name="Street map (Carto)", control=False).add_to(m)
else:
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri World Imagery)",
        control=False,
    ).add_to(m)

# boundary outline
folium.GeoJson(
    south_sudan_geojson,
    name="South Sudan",
    style_function=lambda x: {"fillOpacity": 0.0, "color": "black", "weight": 3},
).add_to(m)

# =========================================================
# REAL PUBLIC DATA OVERLAYS (NASA GIBS)
# =========================================================
# MODIS True Color daily (JPEG tiles). Use today’s UTC date.
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

if show_modis:
    folium.TileLayer(
        tiles=(
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"MODIS_Terra_CorrectedReflectance_TrueColor/default/{today}/"
            "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
        ),
        attr="NASA GIBS (MODIS Terra True Color)",
        name="MODIS True Color (daily) — NASA GIBS",
        overlay=True,
        control=True,
        opacity=overlay_opacity,
    ).add_to(m)

if show_imerg:
    folium.TileLayer(
        tiles=(
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "IMERG_Precipitation_Rate/default/default/"
            "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
        ),
        attr="NASA GIBS (GPM IMERG Precipitation Rate)",
        name="IMERG Precipitation Rate (30-min) — NASA GIBS",
        overlay=True,
        control=True,
        opacity=overlay_opacity,
    ).add_to(m)

# =========================================================
# MODEL LAYERS: HEATMAPS + MARKERS (CHECKBOXES CONTROL THESE)
# =========================================================
def add_heat_and_markers(points, layer_name, show_layer=True):
    # heatmap layer
    heat_data = [[p[0], p[1], p[2]] for p in points]
    heat_fg = folium.FeatureGroup(name=f"{layer_name} heatmap", show=show_layer)
    HeatMap(
        heat_data,
        radius=30,
        blur=24,
        min_opacity=heat_opacity,
        max_zoom=18,
    ).add_to(heat_fg)
    heat_fg.add_to(m)

    # markers layer (bigger + numbered)
    mark_fg = folium.FeatureGroup(name=f"{layer_name} markers", show=show_layer)
    for i, p in enumerate(points[:14], start=1):
        lat, lon, score = p[0], p[1], p[2]
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            weight=2,
            fill=True,
            fill_opacity=0.90,
            popup=folium.Popup(
                html=(
                    f"<b>{layer_name} hotspot #{i}</b><br>"
                    f"score: {score:.2f}<br>"
                    f"lat/lon: {lat:.3f}, {lon:.3f}<br>"
                    f"why: {explain_reason(p, delta=0.0)}"
                ),
                max_width=320,
            ),
        ).add_to(mark_fg)
    mark_fg.add_to(m)

if show_nowcast:
    add_heat_and_markers(nowcast_pts, "Nowcast", show_layer=True)
if show_fc7:
    add_heat_and_markers(fc7_pts, "Forecast (7d)", show_layer=True)
if show_fc30:
    add_heat_and_markers(fc30_pts, "Forecast (30d)", show_layer=True)

folium.LayerControl(collapsed=False).add_to(m)

# =========================================================
# LAYOUT: MAP + ALERTS
# =========================================================
left, right = st.columns([2.2, 1])

with left:
    # Render map and capture user pan/zoom so it doesn’t “jump” weirdly
    map_state = st_folium(
        m,
        width=None,
        height=650,
        key="map",
        returned_objects=["center", "zoom"],
    )

    # Update session state ONLY when user moves map (not when we pressed a zoom button)
    if map_state and not st.session_state.pending_zoom:
        try:
            if map_state.get("center") and map_state.get("zoom") is not None:
                st.session_state.map_center = [map_state["center"]["lat"], map_state["center"]["lng"]]
                st.session_state.map_zoom = int(map_state["zoom"])
        except Exception:
            pass

    # reset flag after render
    st.session_state.pending_zoom = False

with right:
    st.subheader("Alerts")

    if not show_alerts:
        st.info("Turn on **Anomaly alerts** in the sidebar to view alerts.")
    else:
        st.caption("Alerts highlight places where forecast suitability increases vs nowcast (delta score).")

        st.markdown("### 7-day")
        for idx, (p, delta) in enumerate(alerts_7, start=1):
            reason = explain_reason(p, delta)
            st.markdown(f"**{idx}.** {reason}")
            st.caption(f"delta: {delta:+.2f} | lat/lon: {p[0]:.3f}, {p[1]:.3f}")

            if st.button(f"Zoom to 7-day alert #{idx}", key=f"z7_{idx}"):
                st.session_state.pending_zoom = True
                st.session_state.map_center = [p[0], p[1]]
                st.session_state.map_zoom = 9
                st.rerun()

            st.divider()

        st.markdown("### 30-day")
        for idx, (p, delta) in enumerate(alerts_30, start=1):
            reason = explain_reason(p, delta)
            st.markdown(f"**{idx}.** {reason}")
            st.caption(f"delta: {delta:+.2f} | lat/lon: {p[0]:.3f}, {p[1]:.3f}")

            if st.button(f"Zoom to 30-day alert #{idx}", key=f"z30_{idx}"):
                st.session_state.pending_zoom = True
                st.session_state.map_center = [p[0], p[1]]
                st.session_state.map_zoom = 9
                st.rerun()

            st.divider()

st.caption(
    "Real data layers: MODIS True Color + IMERG precipitation are live NASA GIBS overlays. "
    "Hotspot scoring is a v1 heuristic placeholder (replaceable with a true model)."
)
