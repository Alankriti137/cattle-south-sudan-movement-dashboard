import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(
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
gdf = gdf.to_crs(epsg=4326)

# map
m = folium.Map(location=[7.5, 30], zoom_start=6, tiles="cartodbpositron")

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
    HeatMap(points, name="Nowcast heatmap").add_to(m)

if show_fc7:
    folium.Marker([8.0, 30.0], tooltip="7-day forecast placeholder").add_to(m)
if show_fc30:
    folium.Marker([6.5, 29.5], tooltip="30-day forecast placeholder").add_to(m)
if show_anom:
    folium.Marker([7.8, 32.2], tooltip="Anomaly placeholder").add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=900, height=650, key="ss_map")
