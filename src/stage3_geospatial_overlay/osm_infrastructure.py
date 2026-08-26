"""
osm_infrastructure.py

Pulls OpenStreetMap infrastructure data for the Thames catchment (station
39002) using its actual boundary polygon from catchment_boundary.py, not
a point buffer. Used as the infrastructure layer in Stage 3's risk
overlay map.

Queries each tag category separately rather than one combined request --
a combined query against a catchment this large returns a huge, sparse,
mixed-type response that pandas 3.x's string-dtype conversion handles
very slowly (this caused an apparent hang on an early attempt, even
though Overpass itself had already responded). Smaller per-category
queries avoid that path and make it clear which category is slow/failing.

Note: 'power' (substations/plants) was dropped from the tag set after
repeated Overpass connection failures on that specific query across two
separate runs -- the public Overpass API appears to rate-limit or drop
connections after the sustained heavy querying the larger highway/
building/amenity requests require. Roads, buildings, and critical
amenities are the layers that matter most for flood-risk exposure
anyway; power infrastructure can be added back later as a separate,
smaller, standalone pull if needed.

Usage:
    python src/stage3_geospatial_overlay/osm_infrastructure.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import osmnx as ox

from catchment_boundary import load_catchment_boundary

ox.settings.timeout = 300      # fail loudly after 5 min per request, don't hang silently
ox.settings.log_console = True  # print Overpass request/response activity as it happens

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
OUTPUT_DIR = PROJECT_ROOT / "data" / "stage3_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_ID = "39002"

# OSM tags for infrastructure genuinely relevant to flood risk exposure --
# 'power' deliberately excluded, see module docstring.
INFRASTRUCTURE_TAGS = {
    "highway": True,
    "building": True,
    "amenity": ["hospital", "school", "fire_station", "police"],
}


def fetch_infrastructure(geometry) -> gpd.GeoDataFrame:
    """
    Query OSM for each tag category separately. Isolating by category
    both avoids the pandas conversion slowdown seen with one huge combined
    query, and makes it obvious which category is slow/failing if
    something does go wrong.
    """
    all_features = []
    for tag_key, tag_value in INFRASTRUCTURE_TAGS.items():
        print(f"  Querying '{tag_key}'...")
        try:
            result = ox.features_from_polygon(geometry, tags={tag_key: tag_value})
            print(f"    Got {len(result)} features")
            all_features.append(result)
        except Exception as exc:
            print(f"    FAILED for '{tag_key}': {exc}")

    combined = pd.concat(all_features, ignore_index=False)
    return combined[~combined.index.duplicated(keep="first")]


def summarize_infrastructure(features: gpd.GeoDataFrame) -> dict:
    """Basic counts by category, useful both as a sanity check and as map summary stats."""
    summary = {}
    if "highway" in features.columns:
        summary["road_segments"] = features["highway"].notna().sum()
    if "building" in features.columns:
        summary["buildings"] = features["building"].notna().sum()
    if "amenity" in features.columns:
        summary["critical_amenities"] = features["amenity"].notna().sum()
        summary["amenity_breakdown"] = features["amenity"].value_counts().to_dict()
    return summary


def prepare_for_export(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Select only the columns relevant to the risk overlay and cast them to
    string. OSM's raw attribute table for a pull this broad includes
    hundreds of sparse tag columns (e.g. 'FIXME'), some with list-type or
    mixed-type values that GPKG/pyogrio cannot write directly. Restricting
    to relevant columns avoids both the earlier write failure and an
    unnecessarily huge, messy output file.
    """
    keep_cols = [c for c in ["highway", "building", "amenity", "name"] if c in features.columns]
    clean = features[keep_cols + ["geometry"]].copy()
    for col in keep_cols:
        clean[col] = clean[col].apply(lambda v: str(v) if v is not None else None)
    return clean


if __name__ == "__main__":
    print(f"Loading catchment boundary for station {STATION_ID}...")
    geometry = load_catchment_boundary(STATION_ID)

    print("Fetching OSM infrastructure by category...")
    features = fetch_infrastructure(geometry)

    print(f"Total features retrieved: {len(features)}")
    summary = summarize_infrastructure(features)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    clean_features = prepare_for_export(features)
    output_path = OUTPUT_DIR / f"osm_infrastructure_{STATION_ID}.gpkg"
    clean_features.to_file(output_path, driver="GPKG")
    print(f"Saved: {output_path}")