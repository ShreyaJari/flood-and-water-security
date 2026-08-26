"""
ndwi_water_stress.py

Computes NDWI (Normalized Difference Water Index) from Sentinel-2 L2A
imagery over the Thames catchment (station 39002), as the water-stress
layer for Stage 3's risk overlay map. Exports both a catchment-wide mean
(quick summary metadata) and a spatial NDWI raster (GeoTIFF, clipped to
the catchment boundary) -- the raster is what risk_overlay_map.py will
actually render, since NDWI is a genuinely spatial product straight from
the imagery, unlike the Stage 2 flood-risk classifier's catchment-wide
scalar score.

IMPORTANT TEMPORAL NOTE: CAMELS-GB (and therefore the Stage 2 classifier's
flood-risk score) ends 2015-09-30. Sentinel-2 only launched June 2015, so
there is no usable historical NDWI baseline from that period. This script
pulls RECENT Sentinel-2 imagery instead -- meaning the final Stage 3 map
combines a historical flood-risk classification (2015 test period) with a
present-day water-stress snapshot. risk_overlay_map.py must label both
layers with their actual dates, not imply they represent the same moment.

Prereqs:
    pip install earthengine-api requests
    earthengine authenticate   # one-time, if not already done

Usage:
    python src/stage3_geospatial_overlay/ndwi_water_stress.py
"""

import datetime
import json
from pathlib import Path

import ee
import requests

from catchment_boundary import load_catchment_boundary

GEE_PROJECT = "flood-water-security-toolkit"
STATION_ID = "39002"

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
OUTPUT_DIR = PROJECT_ROOT / "data" / "stage3_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

END_DATE = datetime.date.today()
START_DATE = END_DATE - datetime.timedelta(days=60)
MAX_CLOUD_PCT = 30

# Raster export scale -- coarser than the 20m used for the scalar mean,
# to stay well under Earth Engine's ~32MB direct-download limit for a
# catchment this large (3,445 km2). Sufficient for a catchment-overlay
# visualization, not intended for pixel-level analysis.
RASTER_EXPORT_SCALE_M = 100


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Cloud mask via Sentinel-2's SCL band: exclude cloud shadow(3), cloud medium(8), cloud high(9), cirrus(10)."""
    scl = image.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


def add_ndwi(image: ee.Image) -> ee.Image:
    """NDWI = (Green - NIR) / (Green + NIR), using Sentinel-2 bands B3 (green) and B8 (NIR)."""
    return image.addBands(image.normalizedDifference(["B3", "B8"]).rename("NDWI"))


def shapely_to_ee_geometry(shapely_geom) -> ee.Geometry:
    """Convert a shapely polygon (from catchment_boundary.py) to an EE Geometry via GeoJSON."""
    geojson = json.loads(json.dumps(shapely_geom.__geo_interface__))
    return ee.Geometry(geojson)


def get_most_recent_ndwi_image(station_id: str) -> tuple[ee.Image, ee.Geometry, dict]:
    """
    Build the cloud-masked NDWI collection over the catchment and return
    the most recent usable scene, the catchment geometry, and summary
    metadata (scene count, acquisition date, catchment-wide mean NDWI).
    """
    boundary = load_catchment_boundary(station_id)
    aoi = shapely_to_ee_geometry(boundary)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(str(START_DATE), str(END_DATE))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
        .select(["B3", "B8"])
        .map(add_ndwi)
    )

    n_scenes = collection.size().getInfo()
    if n_scenes == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for station {station_id} in "
            f"{START_DATE} to {END_DATE} under {MAX_CLOUD_PCT}% cloud cover."
        )

    most_recent = collection.sort("system:time_start", False).first()
    acquisition_ms = most_recent.get("system:time_start").getInfo()
    acquisition_date = datetime.datetime.utcfromtimestamp(acquisition_ms / 1000).strftime("%Y-%m-%d")

    mean_ndwi = most_recent.select("NDWI").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=20, maxPixels=1e9
    ).get("NDWI").getInfo()

    metadata = {
        "station_id": station_id,
        "n_scenes_in_window": n_scenes,
        "acquisition_date": acquisition_date,
        "mean_ndwi": round(mean_ndwi, 4) if mean_ndwi is not None else None,
        "window_start": str(START_DATE),
        "window_end": str(END_DATE),
        "raster_export_scale_m": RASTER_EXPORT_SCALE_M,
    }

    return most_recent, aoi, metadata


def export_ndwi_raster(image: ee.Image, aoi: ee.Geometry, station_id: str) -> Path:
    """
    Download the NDWI band as a GeoTIFF clipped to the catchment boundary,
    at RASTER_EXPORT_SCALE_M resolution to stay under Earth Engine's
    direct-download size limit.
    """
    ndwi_clipped = image.select("NDWI").clip(aoi)

    url = ndwi_clipped.getDownloadURL({
        "region": aoi,
        "scale": RASTER_EXPORT_SCALE_M,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    output_path = OUTPUT_DIR / f"ndwi_raster_{station_id}.tif"
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


if __name__ == "__main__":
    print(f"Initializing Earth Engine (project: {GEE_PROJECT})...")
    ee.Initialize(project=GEE_PROJECT)

    print(f"Computing NDWI for station {STATION_ID}, window {START_DATE} to {END_DATE}...")
    image, aoi, metadata = get_most_recent_ndwi_image(STATION_ID)

    print(f"  Scenes found in window: {metadata['n_scenes_in_window']}")
    print(f"  Most recent usable scene: {metadata['acquisition_date']}")
    print(f"  Mean NDWI (catchment-wide): {metadata['mean_ndwi']}")

    metadata_path = OUTPUT_DIR / f"ndwi_{STATION_ID}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print(f"  Exporting NDWI raster (scale={RASTER_EXPORT_SCALE_M}m)...")
    raster_path = export_ndwi_raster(image, aoi, STATION_ID)
    print(f"  Saved raster: {raster_path}")