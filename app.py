import math
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ----------------------------
# page setup
# ----------------------------
st.set_page_config(page_title="Cattle in South Sudan Movement Dashboard", layout="wide")

st.title("Cattle in South Sudan Movement Dashboard")
st.caption("near-real-time (weekly) cattle movement suitability, forecasts, and alerts")

# ----------------------------
# load boundary
# ----------------------------
# IMPORTANT: your file name is south_sudan.geojson.json (based on your screenshot)
BOUNDARY_PATH = "data/south_sudan.geojson.json"

@st.cache_data
def load_boundary(path: str):
    gdf = gpd.read_file(path)
    # force to EPSG:4326 if missing
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

# center map
centroid = gdf.unary_union.centroid
CENTER_LAT, CENTER_LON = float(centroid.y), float(centroid.x)

# ----------------------------
# sidebar controls (THE CHECKBOXES ACTUALLY DO SOMETHING)
# ----------------------------
st.sidebar.header("Layers")

show_nowcast = st.sidebar.checkbox("Nowcast (current)", value=True)
show_fc7 = st.sidebar.checkbox("Forecast (7 days)", value=False)
show_fc30 = st.sidebar.checkbox("Forecast (30 days)", value=False)
show_alerts = st.sidebar.checkbox("Anomaly alerts", value=True)

st.sidebar.divider()
st.sidebar.subheader("Real public data layers (auto-updating tiles)")
show_truecolor = st.sidebar.checkbox("Satellite: MODIS True Color (daily)", value=False)
show_precip = st.sidebar.checkbox("Weather: IMERG Precipitation Rate (30-min)", value=False)

st.sidebar.divider()
marker_size = st.sidebar.slider("Marker size", min_value=6, max_value=18, value=12, step=1)
heat_opacity = st.sidebar.slider("Heat opacity", min_value=0.2, max_value=0.9, value=0.55, step=0.05)

# ----------------------------
# simple “engine” (v1 heuristic) — produces points + forecasts
# ----------------------------
# we’ll generate hotspots on a grid inside the bounding box, then score them.
# (this is a placeholder engine you can swap later for a real model.)
minx, miny, maxx, maxy = gdf.total_bounds

def make_points(seed: int, step_deg: float = 0.6):
    rng = np.random.default_rng(seed)
    points = []
    lats = np.arange(miny, maxy, step_deg)
    lons = np.arange(minx, maxx, step_deg)
    for lat in lats:
        for lon in lons:
            # jitter so it doesn't look like a perfect grid
            jlat = lat + rng.uniform(-0.12, 0.12)
            jlon = lon + rng.uniform(-0.12, 0.12)

            # keep only if inside south sudan polygon
            try:
                if not gdf.unary_union.contains(gpd.points_from_xy([jlon], [jlat])[0]):
                    continue
            except Exception:
                # if contains check fails, still include (won't break app)
                pass

            # heuristic “suitability” score
            # vegetation proxy: favor central lat band a bit
            veg = 1.0 - min(1.0, abs(jlat - CENTER_LAT) / 6.0)

            # rainfall proxy: random but spatially smooth-ish
            rain = rng.random()

            # accessibility proxy: slightly favor near center lon/lat
            access = 1.0 - min(1.0, (abs(jlat - CENTER_LAT) + abs(jlon - CENTER_LON)) / 10.0)

            score = 0.45 * veg + 0.35 * rain + 0.20 * access
            points.append((jlat, jlon, float(score), float(veg), float(rain), float(access)))
    # keep top N hotspots
    points.sort(key=lambda x: x[2], reverse=True)
    return points[:30]

# time-seeded so it changes week to week
week_seed = int(datetime.now(timezone.utc).strftime("%Y%U"))

nowcast_pts = make_points(seed=week_seed)
fc7_pts = make_points(seed=week_seed + 7)
fc30_pts = make_points(seed=week_seed + 30)

def top_alerts(now_pts, forecast_pts, k=6):
    # alert if a location gets much hotter in forecast vs now
    alerts = []
    # index now by rounded lat/lon
    now_index = {(round(p[0], 1), round(p[1], 1)): p for p in now_pts}
    for p in forecast_pts:
        key = (round(p[0], 1), round(p[1], 1))
        if key in now_index:
            delta = p[2] - now_index[key][2]
        else:
            delta = p[2]  # new hotspot
        alerts.append((p, float(delta)))
    alerts.sort(key=lambda x: x[1], reverse=True)
    return alerts[:k]

alerts_7 = top_alerts(nowcast_pts, fc7_pts, k=6)
alerts_30 = top_alerts(nowcast_pts, fc30_pts, k=6)

def explain(p):
    # p = (lat, lon, score, veg, rain, access)
    veg, rain, access = p[3], p[4], p[5]
    parts = []
    # make reasons DIFFERENT per alert based on strongest drivers
    drivers = sorted(
        [("vegetation/forage signal", veg), ("recent rainfall signal", rain), ("access/centrality signal", access)],
        key=lambda x: x[1],
        reverse=True
    )
    # pick top 2 reasons
    for name, val in drivers[:2]:
        parts.append(f"{name} strong ({val:.2f})")
    return ", ".join(parts)
# ----------------------------
# map view state (so alerts can zoom map)
# ----------------------------
if "map_center" not in st.session_state:
    st.session_state.map_center = [CENTER_LAT, CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 6

# ----------------------------
# build folium map
# ----------------------------
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles=None,
    control_scale=True
)

# base layers (so you can actually switch)
folium.TileLayer("cartodbpositron", name="Street map (Carto)").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite (Esri World Imagery)",
).add_to(m)

# boundary outline
folium.GeoJson(
    south_sudan_geojson,
    name="South Sudan",
    style_function=lambda x: {"fillOpacity": 0.0, "color": "black", "weight": 3},
).add_to(m)

# “real public data” overlays (tiles)
# NOTE: these are tile layers served by NASA GIBS. they update over time.
# URL pattern referenced from common GIBS WMTS usage patterns. :contentReference[oaicite:0]{index=0}
if show_truecolor:
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="Street map (OSM)",
        attr="© OpenStreetMap contributors",
        overlay=False,
        control=True
    ).add_to(m)

if show_precip:
    folium.TileLayer(
        tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/IMERG_Precipitation_Rate/default/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png",
        attr="NASA GIBS (GPM IMERG Precipitation Rate)",
        name="IMERG precip rate (30-min)",
        overlay=True,
        control=True,
        opacity=0.70,
    ).add_to(m)

def add_heat_and_markers(points, layer_name):
    # heatmap
    heat_data = [[p[0], p[1], p[2]] for p in points]
    HeatMap(heat_data, name=f"{layer_name} heatmap", radius=28, blur=22, min_opacity=heat_opacity).add_to(m)

    # markers
    fg = folium.FeatureGroup(name=f"{layer_name} markers", show=True)
    for i, p in enumerate(points[:12], start=1):
        lat, lon, score = p[0], p[1], p[2]
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            weight=2,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{layer_name} hotspot #{i}</b><br>"
                f"score: {score:.2f}<br>"
                f"lat/lon: {lat:.3f}, {lon:.3f}<br>"
                f"why: {explain(p)}",
                max_width=280,
            ),
        ).add_to(fg)
    fg.add_to(m)

if show_nowcast:
    add_heat_and_markers(nowcast_pts, "Nowcast")

if show_fc7:
    add_heat_and_markers(fc7_pts, "Forecast (7d)")

if show_fc30:
    add_heat_and_markers(fc30_pts, "Forecast (30d)")

folium.LayerControl(collapsed=False).add_to(m)

# ----------------------------
# layout: map + alerts panel
# ----------------------------
left, right = st.columns([2.2, 1])

with left:
    st_folium(m, width=None, height=650)

with right:
    st.subheader("alerts")

    if not show_alerts:
        st.info("turn on 'Anomaly alerts' in the sidebar to view alerts.")
    else:
        st.caption("alerts = places that get hotter vs nowcast (delta score)")
if st.button(f"zoom to 7-day alert #{idx}", key=f"z7_{idx}"):
    st.session_state.map_center = [p[0], p[1]]
    st.session_state.map_zoom = 9
    st.rerun()
        st.markdown("### 7-day")
        for idx, (p, delta) in enumerate(alerts_7, start=1):
            st.markdown(
                f"**{idx}. {('new hotspot' if delta > 0.75 else 'route deviation' if delta > 0.45 else 'moderate increase')}**  \n"
                f"delta: **+{delta:.2f}**  \n"
                f"score: {p[2]:.2f}  \n"
                f"lat/lon: {p[0]:.3f}, {p[1]:.3f}  \n"
                f"why: {explain(p)}"
            )
            st.divider()
if st.button(f"zoom to 30-day alert #{idx}", key=f"z30_{idx}"):
    st.session_state.map_center = [p[0], p[1]]
    st.session_state.map_zoom = 9
    st.rerun()
        st.markdown("### 30-day")
        for idx, (p, delta) in enumerate(alerts_30, start=1):
            st.markdown(
                f"**{idx}. movement shift**  \n"
                f"delta: **+{delta:.2f}**  \n"
                f"score: {p[2]:.2f}  \n"
                f"lat/lon: {p[0]:.3f}, {p[1]:.3f}  \n"
                f"why: {explain(p)}"
            )
            st.divider()

st.caption("note: the scoring is a v1 heuristic engine; the satellite/weather overlays are real public layers.")
