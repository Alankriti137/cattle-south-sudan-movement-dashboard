import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
def top_hotspots(points, k=5):
    # points: [lat, lon, score]
    pts = sorted(points, key=lambda x: x[2], reverse=True)[:200]
    hotspots = []
    used = []

    def far_enough(lat, lon):
        for la, lo in used:
            if abs(la - lat) < 0.7 and abs(lo - lon) < 0.7:
                return False
        return True

    for lat, lon, score in pts:
        if far_enough(lat, lon):
            hotspots.append({"lat": lat, "lon": lon, "score": float(score)})
            used.append((lat, lon))
        if len(hotspots) >= k:
            break
    return hotspots
from folium.plugins import HeatMap

st.set_page_config(
    st.sidebar.header("Layers")

show_nowcast = st.sidebar.checkbox("Nowcast (current)", True)
show_7d = st.sidebar.checkbox("Forecast (7 days)", False)
show_30d = st.sidebar.checkbox("Forecast (30 days)", False)
show_alerts = st.sidebar.checkbox("Anomaly alerts", False)
    page_title="Cattle in South Sudan Movement Dashboard",
    layout="wide"
)

st.title("Cattle in South Sudan Movement Dashboard")
st.caption("Near-real-time (weekly) cattle movement suitability, forecasts, and alerts")

# sidebar
st.sidebar.header("Layers")
show_nowcast = st.sidebar.checkbox("Nowcast (current)", value=True)
show_fc7 = st.sidebar.checkbox("Forecast (7 days)")
show_fc30 = st.sidebar.checkbox("Forecast (30 days)")
show_anom = st.sidebar.checkbox("Anomaly alerts")

# load south sudan boundary
gdf = gpd.read_file("data/south_sudan.geojson.json")
# --- DEMO DATA (replace later with real model outputs) ---

nowcast_points = [
    [7.1, 30.4],
    [7.2, 30.6],
    [6.9, 29.8],
]

forecast_7d_points = [
    [7.3, 30.8],
    [7.0, 30.2],
]

forecast_30d_points = [
    [7.6, 31.0],
    [6.7, 29.5],
]

alerts = [
    {
        "lat": 7.092,
        "lon": 30.510,
        "score": 0.98,
        "reason": "High vegetation and nearby water sources"
    },
    {
        "lat": 6.990,
        "lon": 29.765,
        "score": 0.92,
        "reason": "Unusual cattle concentration for this season"
    }
]
gdf = gdf.to_crs(epsg=4326)

# map
m = folium.Map(location=[7.5, 30], zoom_start=6, tiles="cartodbpositron")
from folium.plugins import MarkerCluster

marker_cluster = MarkerCluster().add_to(m)
folium.GeoJson(
    gdf,
    name="South Sudan",
    style_function=lambda x: {
        "fillOpacity": 0,
        "color": "black",
        "weight": 2
    }
).add_to(m)

# ---- nowcast heatmap (synthetic v1) ----
@st.cache_data
def build_nowcast_points(bounds, geom_wkt):
    minx, miny, maxx, maxy = bounds
    geom = gpd.GeoSeries.from_wkt([geom_wkt], crs="EPSG:4326").iloc[0]

    import random
    random.seed(42)
    pts = []
    for _ in range(800):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        if geom.contains(gpd.points_from_xy([lon], [lat])[0]):
            score = max(0, 1 - (abs(lat - 7.0) / 6) - (abs(lon - 30.5) / 10))
            pts.append([lat, lon, score])
    return pts

if show_nowcast:
    bounds = tuple(gdf.total_bounds)
    geom_wkt = gdf.geometry.unary_union.wkt
    points = build_nowcast_points(bounds, geom_wkt)
 # --- MAP LAYERS ---

if show_nowcast:
    HeatMap(
        nowcast_points,
        radius=35,
        blur=25,
        min_opacity=0.4,
        name="Nowcast"
    ).add_to(m)

if show_7d:
    HeatMap(
        forecast_7d_points,
        radius=30,
        blur=22,
        min_opacity=0.35,
        name="7-day Forecast"
    ).add_to(m)

if show_30d:
    HeatMap(
        forecast_30d_points,
        radius=30,
        blur=22,
        min_opacity=0.35,
        name="30-day Forecast"
    ).add_to(m)

if show_alerts:
    for alert in alerts:
        folium.CircleMarker(
            location=[alert["lat"], alert["lon"]],
            radius=14,          # BIGGER markers
            color="black",
            weight=2,
            fill=True,
            fill_color="yellow",
            fill_opacity=0.9,
            popup=f"""
            <b>Hotspot</b><br>
            Score: {alert['score']}<br>
            Reason: {alert['reason']}
            """
        ).add_to(m)

if show_fc7:
    folium.Marker([8.0, 30.0], tooltip="7-day forecast placeholder").add_to(m)
if show_fc30:
    folium.Marker([6.5, 29.5], tooltip="30-day forecast placeholder").add_to(m)
if show_anom:
    folium.Marker([7.8, 32.2], tooltip="Anomaly placeholder").add_to(m)
# --- signal markers ---
folium.Marker(
    location=[7.6, 31.2],
    popup="High vegetation availability (food)",
    icon=folium.Icon(color="green", icon="leaf")
).add_to(marker_cluster)

folium.Marker(
    location=[8.1, 32.6],
    popup="Permanent water access",
    icon=folium.Icon(color="blue", icon="tint")
).add_to(marker_cluster)

folium.Marker(
    location=[7.9, 30.8],
    popup="Herd convergence / grazing pressure",
    icon=folium.Icon(color="red", icon="warning-sign")
).add_to(marker_cluster)
folium.LayerControl(collapsed=False).add_to(m)

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    st_folium(m, width=900, height=650, key="ss_map")

with col2:
    st.subheader("alerts")
    if show_nowcast:
        hotspots = top_hotspots(points, k=5)
        for i, h in enumerate(hotspots, start=1):
            st.markdown(f"**{i}. hotspot**")
            st.write(f"score: {h['score']:.2f}")
            st.write(f"lat/lon: {h['lat']:.3f}, {h['lon']:.3f}")
            st.caption("flagged because suitability is high relative to other areas (v1 heuristic).")
            st.divider()
    else:
        st.caption("turn on nowcast to generate alerts.")
