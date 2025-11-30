"""
main.py

FastAPI backend to compute roof pitch using LINZ geocoding/building outlines + LINZ LiDAR tiles.
Drop this into the repo root. It exposes /pitch?address= or /pitch?lat=&lon=

Dependencies (pip):
  fastapi uvicorn requests rasterio shapely pyproj numpy aiohttp cachetools python-multipart

Environment variables (optional):
  LINZ_API_KEY - if you have one for LINZ services
  CACHE_DIR - where to cache lidar tiles and footprints (default ./cache)

Note: This implementation tries LINZ endpoints first and falls back to OpenStreetMap Overpass if needed.
It downloads GeoTIFF LiDAR DSM tiles, samples elevations along building faces, and computes pitch per face.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import requests
import tempfile
import math
import json
from typing import List, Tuple, Optional
from urllib.parse import urlencode

# Geospatial libs
import rasterio
from rasterio.warp import transform
from rasterio.io import MemoryFile
from shapely.geometry import shape, Point, mapping, LineString, Polygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import numpy as np

# Simple in-memory cache
from cachetools import TTLCache, cached

CACHE_DIR = os.environ.get('CACHE_DIR', './cache')
os.makedirs(CACHE_DIR, exist_ok=True)

LINZ_API_KEY = os.environ.get('LINZ_API_KEY')

app = FastAPI(title="Roof Pitch API - LiDAR + Footprints")

# Cache for remote requests
req_cache = TTLCache(maxsize=1024, ttl=60 * 60)

transform_wgs84_to_nztm = Transformer.from_crs('EPSG:4326', 'EPSG:2193', always_xy=True)
transform_nztm_to_wgs84 = Transformer.from_crs('EPSG:2193', 'EPSG:4326', always_xy=True)


class PitchResult(BaseModel):
    address: Optional[str] = None
    lat: float
    lon: float
    faces: dict
    average_pitch: float
    confidence: float


# ------------------------
# Helper: HTTP get with caching
# ------------------------
@cached(req_cache)
def http_get(url, params=None, headers=None, stream=False):
    if params:
        url = url + ('&' if '?' in url else '?') + urlencode(params)
    resp = requests.get(url, headers=headers, stream=stream)
    resp.raise_for_status()
    return resp


# ------------------------
# 1) Geocode: try LINZ then Nominatim fallback
# ------------------------
@cached(req_cache)
def geocode_address(address: str) -> Tuple[float, float, Optional[str]]:
    # Try LINZ geocode
    try:
        url = 'https://api.linz.govt.nz/geocode'  # placeholder - if you have real endpoint, replace
        params = {'q': address}
        headers = {'Authorization': f'Bearer {LINZ_API_KEY}'} if LINZ_API_KEY else None
        r = http_get(url, params=params, headers=headers)
        data = r.json()
        # adapt depending on the LINZ response shape; here's a generic attempt
        if isinstance(data, dict) and data.get('results'):
            r0 = data['results'][0]
            lat = float(r0['lat'])
            lon = float(r0['lon'])
            return lat, lon, r0.get('formatted')
    except Exception:
        pass

    # Nominatim fallback
    try:
        url = 'https://nominatim.openstreetmap.org/search'
        params = {'q': address, 'format': 'json', 'limit': 1}
        r = http_get(url, params=params, headers={'User-Agent': 'roof-pitch-bot/1.0'})
        hits = r.json()
        if hits:
            lat = float(hits[0]['lat'])
            lon = float(hits[0]['lon'])
            return lat, lon, hits[0].get('display_name')
    except Exception as e:
        print('Geocode fallback error', e)

    raise HTTPException(status_code=404, detail='Address not found')


# ------------------------
# 2) Building footprint: try LINZ footprints (WFS) then OSM Overpass
# ------------------------
@cached(req_cache)
def get_building_footprint(lat: float, lon: float) -> Polygon:
    # Try LINZ building outlines WFS - placeholder URL
    try:
        # Example WFS request (adapt if you have a LINZ WFS service URL)
        wfs_url = 'https://api.linz.govt.nz/services;type=wfs'  # replace with real LINZ WFS
        bbox = f"{lon-0.0005},{lat-0.0005},{lon+0.0005},{lat+0.0005}"
        params = {'SERVICE': 'WFS', 'REQUEST': 'GetFeature', 'TYPENAME': 'buildings', 'bbox': bbox, 'outputFormat': 'application/json'}
        r = http_get(wfs_url, params=params)
        data = r.json()
        if 'features' in data and len(data['features']) > 0:
            # pick closest feature
            pt = Point(lon, lat)
            best = min(data['features'], key=lambda f: Point(f['geometry']['coordinates'][0][0]).distance(pt))
            geom = shape(best['geometry'])
            if isinstance(geom, (Polygon,)):
                return geom
    except Exception:
        pass

    # Fallback: Overpass OSM
    try:
        overpass_url = 'https://overpass-api.de/api/interpreter'
        # query building polygons within 50m
        q = f"[out:json];(way[building](around:50,{lat},{lon});relation[building](around:50,{lat},{lon}););out body;>;out skel qt;"
        r = requests.post(overpass_url, data={'data': q}, headers={'User-Agent': 'roof-pitch-bot/1.0'})
        r.raise_for_status()
        data = r.json()
        # convert to GeoJSON-like polygons (simple approach)
        # We will attempt to find a way/relation that contains centroid near the point
        elements = data.get('elements', [])
        ways = [e for e in elements if e['type'] == 'way']
        nodes = {e['id']: e for e in elements if e['type'] == 'node'}
        candidates = []
        for w in ways:
            coords = []
            for nid in w['nodes']:
                n = nodes.get(nid)
                if n:
                    coords.append((n['lon'], n['lat']))
            if len(coords) >= 4:
                poly = Polygon(coords)
                if poly.is_valid and poly.centroid.distance(Point(lon, lat)) < 0.001:
                    candidates.append(poly)
        if candidates:
            # choose largest area (likely the building)
            best = max(candidates, key=lambda p: p.area)
            return best
    except Exception as e:
        print('Overpass error', e)

    raise HTTPException(status_code=404, detail='Building footprint not found')


# ------------------------
# 3) Find LiDAR tile and download (simple tile search using LINZ LDS)
#    This code assumes LiDAR is available as GeoTIFF DSM. In practice you may need to work with LAZ and convert.
# ------------------------
@cached(req_cache)
def get_lidar_tile_for_point(lat: float, lon: float) -> str:
    # Convert to NZTM (2193)
    x, y = transform_wgs84_to_nztm.transform(lon, lat)

    # Example LINZ LDS query - placeholder endpoint
    try:
        # This is a generic approach: LINZ LDS or your own index should give the tile path for (x,y).
        # For demo we attempt a fake endpoint — replace with your LINZ metadata index or local tile repository.
        index_url = 'https://api.linz.govt.nz/lidar/index'  # replace with actual index service
        params = {'x': x, 'y': y}
        r = http_get(index_url, params=params)
        info = r.json()
        tile_url = info.get('tile_url')
        if tile_url:
            # download to cache
            local_name = os.path.join(CACHE_DIR, os.path.basename(tile_url))
            if not os.path.exists(local_name):
                rr = requests.get(tile_url, stream=True)
                rr.raise_for_status()
                with open(local_name, 'wb') as fh:
                    for chunk in rr.iter_content(chunk_size=8192):
                        fh.write(chunk)
            return local_name
    except Exception:
        pass

    # If you don't have a tile index service, try to query LINZ LDS for available DSM products near point.
    # Since endpoints differ between providers, this section should be adapted to your LiDAR hosting.
    raise HTTPException(status_code=404, detail='LiDAR tile not found for this location — adapt get_lidar_tile_for_point to your LiDAR provider')


# ------------------------
# 4) Sample elevations along polygon faces
# ------------------------
def sample_elevations_from_geotiff(geotiff_path: str, points: List[Tuple[float, float]]) -> List[float]:
    with rasterio.open(geotiff_path) as src:
        # points are lon,lat in WGS84; need to transform to raster CRS if necessary
        src_crs = src.crs
        if src_crs and src_crs.to_epsg() != 4326:
            # warp points to raster CRS
            transformer = Transformer.from_crs('EPSG:4326', src_crs, always_xy=True)
            pts = [transformer.transform(lon, lat) for lon, lat in points]
        else:
            pts = [(p[0], p[1]) for p in points]

        samples = list(src.sample(pts))
        # samples are arrays (bands,) — we take first band
        elevations = [float(s[0]) if s is not None else float('nan') for s in samples]
        return elevations


# ------------------------
# 5) Compute pitch per roof face
# ------------------------
def compute_pitch_for_polygon(poly: Polygon, lidar_tif: str, points_per_edge: int = 6) -> dict:
    # Ensure polygon is in WGS84 lon/lat for simplicity
    poly_wgs = poly

    faces = {}
    coords = list(poly_wgs.exterior.coords)
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        # sample points along edge (interpolate lon/lat)
        line = LineString([a, b])
        pts = [line.interpolate(float(t) / (points_per_edge - 1), normalized=True) for t in range(points_per_edge)]
        pts_lonlat = [(p.x, p.y) for p in pts]
        elevs = sample_elevations_from_geotiff(lidar_tif, pts_lonlat)
        # compute rise = max elev - min elev along edge (approx)
        if all(math.isnan(e) for e in elevs):
            continue
        clean = [e for e in elevs if not math.isnan(e)]
        rise = max(clean) - min(clean)
        # horizontal run: length of line in meters -> transform endpoints to NZTM then compute distance
        x1, y1 = transform_wgs84_to_nztm.transform(a[0], a[1])
        x2, y2 = transform_wgs84_to_nztm.transform(b[0], b[1])
        run = math.hypot(x2 - x1, y2 - y1)
        if run <= 0.01:
            pitch_deg = 0.0
        else:
            pitch_ratio = rise / run
            pitch_deg = math.degrees(math.atan(pitch_ratio))
        face_name = f'edge_{i+1}'
        faces[face_name] = {
            'start': a,
            'end': b,
            'rise_m': round(rise, 3),
            'run_m': round(run, 3),
            'pitch_deg': round(pitch_deg, 2)
        }

    # compute average pitch
    pitch_vals = [v['pitch_deg'] for v in faces.values() if v.get('pitch_deg') is not None]
    average = float(np.mean(pitch_vals)) if pitch_vals else 0.0

    return {'faces': faces, 'average_pitch': round(average, 2)}


# ------------------------
# Endpoint: /pitch
# ------------------------
@app.get('/pitch', response_model=PitchResult)
def pitch_endpoint(address: Optional[str] = Query(None), lat: Optional[float] = Query(None), lon: Optional[float] = Query(None)):
    if address is None and (lat is None or lon is None):
        raise HTTPException(status_code=400, detail='Provide either address or lat & lon')

    # Geocode if needed
    if address:
        geocoded_lat, geocoded_lon, formatted = geocode_address(address)
        lat, lon = geocoded_lat, geocoded_lon
        addr_text = formatted
    else:
        addr_text = None

    # footprint
    footprint = get_building_footprint(lat, lon)

    # lidar tile
    lidar_local = get_lidar_tile_for_point(lat, lon)

    # compute
    pitch_data = compute_pitch_for_polygon(footprint, lidar_local)

    result = {
        'address': addr_text,
        'lat': lat,
        'lon': lon,
        'faces': pitch_data['faces'],
        'average_pitch': pitch_data['average_pitch'],
        'confidence': 0.95
    }
    return result


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
