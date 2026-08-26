"""
risk_overlay_map.py

Combines Stage 3's three layers into one interactive folium map for the
Thames catchment (station 39002):
    - Catchment boundary
    - OSM infrastructure: major roads + critical amenities (hospitals,
      schools, fire, police only -- buildings excluded from the map
      itself, since 310k polygons is not renderable in-browser at this
      scale; building density is reported as a summary statistic instead)
    - NDWI water-stress raster, rendered as a georeferenced image overlay
    - Flood-risk classifier output, shown as a banner (catchment-wide
      scalar, NOT a spatial layer -- see Stage 2/3 scope notes)

Explicitly labels the two real dates involved: the flood-risk score comes
from the 2015 CAMELS-GB test period, while NDWI is a 2026 snapshot -- the
map does not imply these represent the same moment in time.

Usage:
    python src/stage3_geospatial_overlay/risk_overlay_map.py
"""

import json
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb

from catchment_boundary import load_catchment_boundary

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
OVERLAY_DIR = PROJECT_ROOT / "data" / "stage3_overlay"
MODEL_DIR = PROJECT_ROOT / "models" / "stage2_xgboost"
FEATURES_DIR = PROJECT_ROOT / "data" / "stage2_features"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage3_geospatial_overlay"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATION_ID = "39002"
MAJOR_ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary"]

# Restricted to exactly the 4 categories originally queried for.
# NOTE: an earlier version filtered with amenity.notna(), which also
# swept up incidental amenity tags on buildings/roads queried for other
# reasons (pubs, cafes, churches, etc.) -- the same tag-leakage issue
# already seen with 'power' in osm_infrastructure.py. .isin() avoids that.
CRITICAL_AMENITY_TYPES = ["hospital", "school", "fire_station", "police"]


def load_flood_risk_summary(station_id: str) -> dict:
    """Load the Stage 2 classifier's saved test-period metadata (scores, dates)."""
    metadata_path = MODEL_DIR / f"xgb_{station_id}_metadata.json"
    with open(metadata_path) as f:
        return json.load(f)


def get_latest_test_period_prediction(station_id: str) -> dict:
    """
    Score the most recent day in the test period using the saved XGBoost
    checkpoint, as the headline "current" flood-risk figure for the
    banner. This is the most recent day the classifier has ever seen
    (2015-09-30-ish), not today -- labeled explicitly as such downstream.
    """
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"xgb_{station_id}.json")

    metadata_path = MODEL_DIR / f"xgb_{station_id}_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]

    df = pd.read_csv(FEATURES_DIR / f"features_{station_id}.csv", index_col=0, parse_dates=True)
    latest_row = df.iloc[[-1]]
    latest_date = latest_row.index[0].strftime("%Y-%m-%d")

    proba = model.predict_proba(latest_row[feature_cols])[0, 1]
    label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

    return {"date": latest_date, "risk_probability": round(float(proba), 3), "label": label}


def load_osm_layers(station_id: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load the saved OSM GeoPackage and split into major roads and critical
    amenities. Amenities are restricted to CRITICAL_AMENITY_TYPES via
    .isin(), not .notna(), to avoid pulling in incidental amenity tags
    from features that were queried for a different reason (see module
    docstring).
    """
    gdf = gpd.read_file(OVERLAY_DIR / f"osm_infrastructure_{station_id}.gpkg")

    roads = gdf[gdf["highway"].isin(MAJOR_ROAD_CLASSES)].copy()
    amenities = gdf[gdf["amenity"].isin(CRITICAL_AMENITY_TYPES)].copy()

    return roads, amenities


def render_ndwi_overlay_image(station_id: str) -> tuple[str, list]:
    """
    Convert the NDWI GeoTIFF into a colored PNG with a matplotlib
    colormap, for use as a folium ImageOverlay. Returns the PNG path and
    the [[south, west], [north, east]] bounds folium expects.
    """
    raster_path = OVERLAY_DIR / f"ndwi_raster_{station_id}.tif"
    with rasterio.open(raster_path) as src:
        ndwi = src.read(1)
        bounds = src.bounds

    ndwi_masked = np.ma.masked_invalid(ndwi)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(ndwi_masked, cmap="RdYlBu", vmin=-1, vmax=1)
    ax.axis("off")

    png_path = RESULTS_DIR / f"ndwi_overlay_{station_id}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    folium_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return str(png_path), folium_bounds


def build_map(station_id: str) -> folium.Map:
    """Assemble all Stage 3 layers into one interactive folium map."""
    boundary = load_catchment_boundary(station_id)
    centroid = boundary.centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles="OpenStreetMap")

    # Catchment boundary
    folium.GeoJson(
        boundary.__geo_interface__,
        name="Catchment boundary",
        style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0},
    ).add_to(fmap)

    # NDWI raster overlay
    ndwi_png_path, ndwi_bounds = render_ndwi_overlay_image(station_id)
    folium.raster_layers.ImageOverlay(
        image=ndwi_png_path,
        bounds=ndwi_bounds,
        opacity=0.55,
        name="NDWI water stress (2026 snapshot)",
    ).add_to(fmap)

    # OSM roads + critical amenities
    roads, amenities = load_osm_layers(station_id)

    folium.GeoJson(
        roads.__geo_interface__,
        name="Major roads",
        style_function=lambda x: {"color": "#555555", "weight": 1.5},
    ).add_to(fmap)

    amenity_group = folium.FeatureGroup(name="Critical amenities")
    for _, row in amenities.iterrows():
        if row.geometry.geom_type == "Point":
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color="darkred",
                fill=True,
                fill_opacity=0.8,
                popup=f"{row.get('amenity', 'unknown')}: {row.get('name', 'unnamed')}",
            ).add_to(amenity_group)
    amenity_group.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    # Flood-risk banner (scalar, NOT spatial -- deliberately a fixed HTML box, not a map layer)
    risk = get_latest_test_period_prediction(station_id)
    banner_html = f"""
    <div style="position: fixed; top: 10px; left: 60px; z-index: 9999;
                background: white; padding: 12px 18px; border: 2px solid black;
                border-radius: 6px; font-family: sans-serif; font-size: 14px;">
        <b>Thames Catchment Risk Overlay</b><br>
        Flood-risk classification (2015 test period, {risk['date']}):
        <b style="color: {'darkred' if risk['label'] == 'HIGH RISK' else 'darkgreen'};">
            {risk['label']}
        </b> (p={risk['risk_probability']})<br>
        <span style="font-size: 11px; color: #555;">
            NDWI water-stress layer: current 2026 Sentinel-2 imagery.
            These two layers represent different time periods.
        </span>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(banner_html))

    return fmap


if __name__ == "__main__":
    print(f"Building risk overlay map for station {STATION_ID}...")
    fmap = build_map(STATION_ID)

    output_path = RESULTS_DIR / f"risk_overlay_{STATION_ID}.html"
    fmap.save(str(output_path))
    print(f"Saved: {output_path}")