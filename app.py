import math
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium


# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")

st.title("Cattle in South Sudan Movement Dashboard")
st.caption("Near-real-time (weekly) cattle movement suitability, forecasts, and alerts")


# ============================================================
# Load boundary (your file)
# ============================================================
BOUNDARY_PATH = "data/south_sudan.geojson.json"

@st.cache_data
def load_boundary(path: str) -> gpd.GeoDataFrame:
    gdf_ = gpd.read_file(path)
    # Force to EPSG:4326 if missing
    if gdf_.crs is None:
        gdf_ = gdf_.set_crs("EPSG:4326")
    else:
        gdf_ = gdf_.to_crs("EPSG:4326")
    return gdf_

gdf = load_boundary(BOUNDARY_PATH)
south_sudan_geojson = gdf.__geo_interface__

minx, miny, maxx, maxy = gdf.total_bounds
centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

DEFAULT_CENTER = [CENTER_LAT, CENTER_LON]
DEFAULT_ZOOM = 6


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Layers")

# Basemap choices (Street, Esri Satellite, MODIS True Color)
basemap = st.sidebar.radio(
    "Basemap",
    ["Street map (Carto)", "Satellite (Esri World Imagery)", "MODIS True Color (daily) — NASA GIBS"],
    index=0,
)

st.sidebar.divider()

st.sidebar.subheader("Model layers (dashboard outputs)")
show_nowcast = st.sidebar.checkbox("Nowcast (current)", value=True)
show_fc7 = st.sidebar.checkbox("Forecast (7 days)", value=False)
show_fc30 = st.sidebar.checkbox("Forecast (30 days)", value=False)
show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)

st.sidebar.divider()

st.sidebar.subheader("Real public data overlay (auto-updating tiles)")
show_imerg = st.sidebar.checkbox("Weather: IMERG Precipitation Rate (30-min) — NASA GIBS", value=False)

overlay_opacity = st.sidebar.slider("Overlay opacity", min_value=0.0, max_value=1.0, value=0.70, step=0.05)

st.sidebar.divider()

marker_size = st.sidebar.slider("Marker size", min_value=6, max_value=30, value=14, step=1)
heat_opacity = st.sidebar.slider("Heat opacity", min_value=0.05, max_value=0.95, value=0.55, step=0.05)

st.sidebar.divider()

# Extra zoom buttons
colz1, colz2 = st.sidebar.columns(2)
with colz1:
    if st.button("Reset view"):
        st.session_state["map_center"] = DEFAULT_CENTER
        st.session_state["map_zoom"] = DEFAULT_ZOOM
        st.rerun()
with colz2:
    if st.button("Zoom to S. Sudan"):
        # Approximate “fit bounds” effect by centering + slightly closer zoom
        st.session_state["map_center"] = DEFAULT_CENTER
        st.session_state["map_zoom"] = 7
        st.rerun()


# ============================================================
# Engine (v1 heuristic) – produces points + forecasts
# ============================================================
def make_points(seed: int, step_deg: float = 0.6):
    rng = np.random.default_rng(seed)
    points = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)

    poly = gdf.unary_union

    for lat in lats:
        for lon in lons:
            jlat = lat + rng.uniform(-0.12, 0.12)
            jlon = lon + rng.uniform(-0.12, 0.12)

            # Keep only if inside South Sudan polygon
            try:
                pt = gpd.points_from_xy([jlon], [jlat])[0]
                if not poly.contains(pt):
                    continue
            except Exception:
                pass

            # Proxies (still "v1 heuristic", but deterministic and explainable)
            veg = 1.0 - min(1.0, abs(jlat - CENTER_LAT) / 6.0)
            rain = rng.random()
            access = 1.0 - min(1.0, (abs(jlat - CENTER_LAT) + abs(jlon - CENTER_LON)) / 10.0)

            score = 0.45 * veg + 0.35 * rain + 0.20 * access
            points.append((jlat, jlon, float(score), float(veg), float(rain), float(access)))

    points.sort(key=lambda x: x[2], reverse=True)
    return points[:30]


# week-seeded so it changes week-to-week automatically (no reboot needed)
week_seed = int(datetime.now(timezone.utc).strftime("%Y%U"))

nowcast_pts = make_points(seed=week_seed)
fc7_pts = make_points(seed=week_seed + 7)
fc30_pts = make_points(seed=week_seed + 30)


def top_alerts(now_pts, forecast_pts, k=6):
    """
    Alerts = places where forecast suitability increases vs nowcast.
    """
    now_index = {(round(p[0], 1), round(p[1], 1)): p for p in now_pts}
    alerts = []
    for p in forecast_pts:
        key = (round(p[0], 1), round(p[1], 1))
        base = now_index.get(key, None)
        delta = p[2] - (base[2] if base else 0.0)
        alerts.append((p, float(delta), base))
    alerts.sort(key=lambda x: x[1], reverse=True)
    return alerts[:k]


alerts_7 = top_alerts(nowcast_pts, fc7_pts, k=6)
alerts_30 = top_alerts(nowcast_pts, fc30_pts, k=6)


def alert_label(delta: float) -> str:
    if delta > 0.75:
        return "New hotspot likely forming"
    if delta > 0.45:
        return "Route deviation / shift"
    return "Moderate increase"


def explain_drivers(p):
    # p = (lat, lon, score, veg, rain, access)
    veg, rain, access = p[3], p[4], p[5]
    drivers = sorted(
        [("Rainfall", rain), ("Forage/vegetation", veg), ("Access/centrality", access)],
        key=lambda x: x[1],
        reverse=True,
    )
    # top 2 driver names + values
    return drivers[:2]


# ============================================================
# Persist map view (fixes “weird zoom” / resets)
# ============================================================
if "map_center" not in st.session_state:
    st.session_state["map_center"] = DEFAULT_CENTER
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = DEFAULT_ZOOM


# ============================================================
# Build folium map
# ============================================================
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=st.session_state["map_zoom"],
    tiles=None,
    control_scale=True,
    prefer_canvas=True,  # helps marker rendering look smoother
)

# --- Basemaps (radio-driven) ---
# Always add the three basemaps, but only one is "show=True" at a time.
show_carto = basemap.startswith("Street")
show_esri = basemap.startswith("Satellite")
show_modis = basemap.startswith("MODIS")

folium.TileLayer(
    "cartodbpositron",
    name="Street map (Carto)",
    overlay=False,
    control=True,
    show=show_carto,
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite (Esri World Imagery)",
    overlay=False,
    control=True,
    show=show_esri,
).add_to(m)

# REAL DATA basemap: MODIS True Color via NASA GIBS (auto-updating "default" time)
folium.TileLayer(
    tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
          "MODIS_Terra_CorrectedReflectance_TrueColor/default/default/"
          "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
    attr="NASA GIBS (MODIS Terra True Color)",
    name="MODIS True Color (daily) — NASA GIBS",
    overlay=False,
    control=True,
    show=show_modis,
).add_to(m)

# --- Boundary outline (always on) ---
folium.GeoJson(
    south_sudan_geojson,
    name="South Sudan boundary",
    style_function=lambda x: {"fillOpacity": 0.0, "color": "black", "weight": 3},
).add_to(m)

# --- REAL DATA overlay: IMERG precip rate (NASA GIBS) ---
if show_imerg and overlay_opacity > 0:
    folium.TileLayer(
        tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
              "IMERG_Precipitation_Rate/default/default/"
              "GoogleMapsCompatible_Level8/{z}/{y}/{x}.png",
        attr="NASA GIBS (GPM IMERG Precipitation Rate)",
        name="IMERG Precipitation Rate (30-min) — NASA GIBS",
        overlay=True,
        control=True,
        opacity=float(overlay_opacity),
    ).add_to(m)


def add_model_layer(points, layer_name: str, show_layer: bool):
    """
    Adds BOTH heat + markers inside a FeatureGroup so the LayerControl checkbox works.
    """
    group = folium.FeatureGroup(name=layer_name, show=show_layer)

    # Heat
    heat_data = [[p[0], p[1], p[2]] for p in points]
    HeatMap(
        heat_data,
        radius=30,
        blur=24,
        min_opacity=float(heat_opacity),
        name=f"{layer_name} heat",
    ).add_to(group)

    # Markers (more visible: thicker outline + tooltip)
    for i, p in enumerate(points[:14], start=1):
        lat, lon, score = p[0], p[1], p[2]
        top2 = explain_drivers(p)
        tooltip = (
            f"{layer_name} #{i} | score {score:.2f} | "
            f"{top2[0][0]} {top2[0][1]:.2f}, {top2[1][0]} {top2[1][1]:.2f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=int(marker_size),
            weight=3,               # thicker outline (easier to see)
            color="black",
            fill=True,
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=folium.Popup(
                f"<b>{layer_name} hotspot #{i}</b><br>"
                f"Total score: {score:.2f}<br>"
                f"Lat/Lon: {lat:.3f}, {lon:.3f}<br>"
                f"Forage/veg: {p[3]:.2f}<br>"
                f"Rainfall: {p[4]:.2f}<br>"
                f"Access/centrality: {p[5]:.2f}",
                max_width=320,
            ),
        ).add_to(group)

    group.add_to(m)


# Add model layers (these now truly respond to sidebar checkboxes)
if show_nowcast:
    add_model_layer(nowcast_pts, "Nowcast (current)", show_layer=True)
if show_fc7:
    add_model_layer(fc7_pts, "Forecast (7 days)", show_layer=True)
if show_fc30:
    add_model_layer(fc30_pts, "Forecast (30 days)", show_layer=True)

folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# Layout: map + alerts panel
# ============================================================
left, right = st.columns([2.2, 1], gap="large")

with left:
    # IMPORTANT: capturing center/zoom fixes “weird zoom reset”
    map_state = st_folium(
        m,
        height=680,
        use_container_width=True,
        key="main_map",
        returned_objects=["center", "zoom"],
    )

    # Persist user pan/zoom so it doesn't snap back on every interaction
    if isinstance(map_state, dict):
        center = map_state.get("center")
        zoom = map_state.get("zoom")
        if center and isinstance(center, dict) and "lat" in center and "lng" in center:
            st.session_state["map_center"] = [float(center["lat"]), float(center["lng"])]
        if zoom is not None:
            try:
                st.session_state["map_zoom"] = int(zoom)
            except Exception:
                pass


with right:
    st.header("Alerts")
    st.caption("Alerts highlight places where forecast suitability increases vs nowcast (delta score).")

    if not show_alerts:
        st.info("Turn on **Anomaly alerts** in the sidebar to view alerts.")
    else:
        def render_alert_block(title: str, alert_list, zoom_key_prefix: str):
            st.subheader(title)

            for idx, (p, delta, base) in enumerate(alert_list, start=1):
                label = alert_label(delta)
                top2 = explain_drivers(p)

                # Clean scoring layout
                r1 = st.container(border=True)
                with r1:
                    st.markdown(f"**{idx}. {label}**")
                    st.caption(f"Lat/Lon: {p[0]:.3f}, {p[1]:.3f}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Delta vs nowcast", f"{delta:+.2f}")
                    with c2:
                        st.metric("Forecast total score", f"{p[2]:.2f}")

                    # Variable scores (clean + explicit)
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("Rainfall", f"{p[4]:.2f}")
                    with s2:
                        st.metric("Forage/veg", f"{p[3]:.2f}")
                    with s3:
                        st.metric("Access", f"{p[5]:.2f}")

                    st.caption(
                        f"Top drivers: {top2[0][0]} ({top2[0][1]:.2f}), "
                        f"{top2[1][0]} ({top2[1][1]:.2f})"
                    )

                    # Mention real overlays status
                    overlay_bits = []
                    if basemap.startswith("MODIS"):
                        overlay_bits.append("MODIS basemap ON (real)")
                    if show_imerg and overlay_opacity > 0:
                        overlay_bits.append("IMERG overlay ON (real)")
                    if overlay_bits:
                        st.caption(" | ".join(overlay_bits))

                    # Zoom button (extra zoom button requested)
                    if st.button(f"Zoom to {title.lower()} alert #{idx}", key=f"{zoom_key_prefix}_{idx}"):
                        st.session_state["map_center"] = [p[0], p[1]]
                        st.session_state["map_zoom"] = 9
                        st.rerun()

        render_alert_block("7-day", alerts_7, "z7")
        st.divider()
        render_alert_block("30-day", alerts_30, "z30")

    st.caption(
        "Note: The **MODIS True Color** + **IMERG precipitation** layers are **real NASA GIBS tiles**. "
        "The cow suitability/forecasts are a **v1 heuristic model** you can later replace with a real model."
    )
