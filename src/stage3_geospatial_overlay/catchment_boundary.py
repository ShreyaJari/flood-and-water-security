"""
catchment_boundary.py

Shared utility for Stage 3: extracts a single catchment's boundary
polygon from the CAMELS-GB shapefile (671 catchments) and reprojects it
from British National Grid (EPSG:27700, the file's native CRS) to WGS84
(EPSG:4326), since both osmnx and Google Earth Engine expect lat/lon.

Used by osm_infrastructure.py and ndwi_water_stress.py so both pull the
exact same catchment geometry rather than deriving it separately.
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
BOUNDARIES_SHP = (
    PROJECT_ROOT / "data" / "camels_gb" / "camels_gb_v1_full" / "data"
    / "catchment_boundaries" / "CAMELS_GB_catchment_boundaries.shp"
)

SOURCE_CRS = "EPSG:27700"  # British National Grid, native to the shapefile
TARGET_CRS = "EPSG:4326"   # WGS84 lat/lon, required by osmnx and GEE


def load_catchment_boundary(station_id: str) -> BaseGeometry:
    """
    Load a single catchment's boundary polygon by station id, reprojected
    to WGS84. Raises if the station id isn't found -- no silent fallback.
    """
    gdf = gpd.read_file(BOUNDARIES_SHP)
    match = gdf[gdf["ID_STRING"] == station_id]

    if match.empty:
        raise ValueError(f"Station {station_id} not found in catchment boundaries shapefile")
    if len(match) > 1:
        raise ValueError(f"Multiple boundary rows matched station {station_id}: {len(match)}")

    match = match.set_crs(SOURCE_CRS, allow_override=False)  # confirm, don't silently assume
    match_wgs84 = match.to_crs(TARGET_CRS)

    return match_wgs84.geometry.iloc[0]


if __name__ == "__main__":
    # Sanity check -- confirm the Thames boundary loads, reprojects, and
    # produces a plausible bounding box (should fall within the UK's
    # rough lat/lon range: 49-61N, -8 to 2E).
    geom = load_catchment_boundary("39002")
    minx, miny, maxx, maxy = geom.bounds
    print(f"Thames (39002) boundary bounds (WGS84): "
          f"lon [{minx:.4f}, {maxx:.4f}], lat [{miny:.4f}, {maxy:.4f}]")
    print(f"Geometry type: {geom.geom_type}")
    print(f"Area (approx, via WGS84 degrees^2 -- not a real area unit): {geom.area:.6f}")