from fastapi import FastAPI, HTTPException
import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from shapely.geometry import Point
from pyproj import Transformer
from cachetools import TTLCache

app = FastAPI()

# Cache DSM tiles (10 minutes)
tile_cache = TTLCache(maxsize=50, ttl=600)

# LINZ DSM template
DSM_BASE = "https://data.linz.govt.nz/services;key={api_key}/wms?"
DSM_LAYER = "NZ_8m_Digital_Surface_Model"
LINZ_API_KEY = "YOUR_LINZ_API_KEY"

@app.get("/pitch")
def get_pitch(lat: float, lon: float):
    try:
        # Convert WGS84 → NZTM2000
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        x, y = transformer.transform(lon, lat)

        # Build tile request
        params = {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": DSM_LAYER,
            "format": "image/geotiff",
            "width": 256,
            "height": 256,
            "crs": "EPSG:2193",
            "bbox": f"{x-20},{y-20},{x+20},{y+20}",
        }

        url = DSM_BASE.format(api_key=LINZ_API_KEY)

        # Cache lookup
        key = f"{round(x,1)}_{round(y,1)}"
        if key in tile_cache:
            dsm_bytes = tile_cache[key]
        else:
            r = requests.get(url, params=params)
            if r.status_code != 200:
                raise HTTPException(500, "DSM fetch failed")
            dsm_bytes = r.content
            tile_cache[key] = dsm_bytes

        # Read DSM pixel values
        with MemoryFile(dsm_bytes) as mem:
            with mem.open() as src:
                arr = src.read(1)
                transform = src.transform

        # Elevation sampling
        def elevation(px, py):
            row, col = ~transform * (px, py)
            return float(arr[int(col)][int(row)])

        center = elevation(x, y)
        north = elevation(x, y + 2)
        south = elevation(x, y - 2)

        slope = (north - south) / 4
        pitch_deg = abs(np.degrees(np.arctan(slope)))

        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "pitch_degrees": round(pitch_deg, 2),
            "method": "DSM 8m",
        }

    except Exception as e:
        raise HTTPException(500, f"Pitch calculation error: {str(e)}")
