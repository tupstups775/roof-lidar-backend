"""
Production-Ready Roof Pitch Calculator Backend
Integrates LINZ WMTS DSM data with RANSAC plane fitting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from PIL import Image
import io
from sklearn.linear_model import RANSACRegressor
import math
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NZ Roof Pitch Calculator",
    description="Calculate roof pitch using LINZ LiDAR DSM data",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ========================================
# CONFIGURATION
# ========================================

LINZ_API_KEY = "YOUR_LINZ_API_KEY_HERE"  # Replace with actual key
WMTS_URL_TEMPLATE = "https://data.linz.govt.nz/services;key={api_key}/wmts/1.0.0/layer/{layer}/tile/NZTM2000Quad/{z}/{y}/{x}.png"

# Layer options (use layer with best coverage for your area)
DEFAULT_LAYER = "104708"  # NZ 8m Digital Elevation Model (2012) - Nationwide
# Alternative layers:
# "104409" - Auckland 1m Urban Aerial Photos
# "105791" - Wellington 1m

# ========================================
# COORDINATE CONVERSION
# ========================================

def latlon_to_nztm(lat: float, lon: float) -> tuple:
    """
    Convert WGS84 lat/lon to NZTM2000 easting/northing
    
    Simplified conversion - for production, use pyproj:
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193")
    easting, northing = transformer.transform(lat, lon)
    """
    # Approximate conversion (accurate to ~50m)
    # Good enough for tile selection, not for precise surveying
    central_meridian = 173.0
    false_easting = 1600000
    false_northing = 10000000
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    cm_rad = math.radians(central_meridian)
    
    # Simplified Transverse Mercator projection
    k0 = 0.9996  # Scale factor
    a = 6378137.0  # WGS84 equatorial radius
    e = 0.0818191908  # Eccentricity
    
    N = a / math.sqrt(1 - e**2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e**2 * math.cos(lat_rad)**2 / (1 - e**2)
    A = (lon_rad - cm_rad) * math.cos(lat_rad)
    
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad
             - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * math.sin(2*lat_rad)
             + (15*e**4/256 + 45*e**6/1024) * math.sin(4*lat_rad)
             - (35*e**6/3072) * math.sin(6*lat_rad))
    
    easting = k0 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58)*A**5/120) + false_easting
    northing = k0 * (M + N*math.tan(lat_rad)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24 
                     + (61-58*T+T**2+600*C-330)*A**6/720)) + false_northing
    
    return easting, northing


def nztm_to_tile(easting: float, northing: float, zoom: int = 18) -> tuple:
    """
    Convert NZTM coordinates to WMTS tile coordinates
    
    Returns:
        (tile_x, tile_y, pixel_x, pixel_y)
    """
    # NZTM2000Quad tile matrix parameters
    origin_x = 274000
    origin_y = 3087000
    
    # Tile size in meters (varies by zoom)
    tile_sizes = {
        14: 2048,
        15: 1024,
        16: 512,
        17: 256,
        18: 128,
        19: 64
    }
    
    tile_size_m = tile_sizes.get(zoom, 128)
    tile_size_px = 256  # Always 256 pixels per tile
    
    # Calculate tile indices
    tile_x = int((easting - origin_x) / tile_size_m)
    tile_y = int((origin_y - northing) / tile_size_m)
    
    # Calculate pixel position within tile
    pixel_x = int(((easting - origin_x) % tile_size_m) / tile_size_m * tile_size_px)
    pixel_y = int(((origin_y - northing) % tile_size_m) / tile_size_m * tile_size_px)
    
    return tile_x, tile_y, pixel_x, pixel_y


# ========================================
# LINZ DSM DATA FETCHING
# ========================================

def decode_linz_dsm_png(png_bytes: bytes) -> np.ndarray:
    """
    Decode LINZ DSM elevation data from PNG
    
    LINZ encoding formula:
    Elevation (m) = R * 256 + G + B/256 - 10000
    """
    img = Image.open(io.BytesIO(png_bytes))
    img_array = np.array(img)
    
    if len(img_array.shape) < 3:
        raise ValueError("Invalid PNG format - expected RGB")
    
    R = img_array[:, :, 0].astype(np.float32)
    G = img_array[:, :, 1].astype(np.float32)
    B = img_array[:, :, 2].astype(np.float32)
    
    # Decode elevation
    elevation = R * 256 + G + B / 256 - 10000
    
    # Mark no-data values as NaN
    no_data_mask = ((R == 255) & (G == 255) & (B == 255)) | \
                   ((R == 0) & (G == 0) & (B == 0))
    elevation[no_data_mask] = np.nan
    
    return elevation


def fetch_dsm_tile(tile_x: int, tile_y: int, zoom: int, layer: str) -> np.ndarray:
    """
    Fetch and decode a LINZ WMTS DSM tile
    """
    url = WMTS_URL_TEMPLATE.format(
        api_key=LINZ_API_KEY,
        layer=layer,
        z=zoom,
        y=tile_y,
        x=tile_x
    )
    
    logger.info(f"Fetching tile: z={zoom}, x={tile_x}, y={tile_y}")
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        
        elevation_grid = decode_linz_dsm_png(response.content)
        return elevation_grid
        
    except Exception as e:
        logger.error(f"Failed to fetch tile: {e}")
        raise


def extract_roof_area(elevation_grid: np.ndarray, 
                     pixel_x: int, 
                     pixel_y: int, 
                     buffer_pixels: int = 20) -> np.ndarray:
    """
    Extract elevation points around target location
    
    Returns:
        Nx3 array of (x, y, elevation)
    """
    height, width = elevation_grid.shape
    
    # Calculate bounds
    x_min = max(0, pixel_x - buffer_pixels)
    x_max = min(width, pixel_x + buffer_pixels)
    y_min = max(0, pixel_y - buffer_pixels)
    y_max = min(height, pixel_y + buffer_pixels)
    
    # Extract subset
    subset = elevation_grid[y_min:y_max, x_min:x_max]
    
    # Convert to point cloud
    points = []
    for i in range(subset.shape[0]):
        for j in range(subset.shape[1]):
            z = subset[i, j]
            if not np.isnan(z):
                points.append([j, i, z])
    
    return np.array(points)


# ========================================
# PITCH CALCULATION
# ========================================

def calculate_pitch_from_points(points: np.ndarray) -> float:
    """
    Calculate roof pitch using RANSAC plane fitting
    
    Args:
        points: Nx3 array of (x, y, elevation)
    
    Returns:
        Pitch in degrees
    """
    if len(points) < 10:
        raise ValueError(f"Insufficient points: {len(points)}")
    
    # Separate coordinates and elevations
    X = points[:, :2]  # x, y
    z = points[:, 2]   # elevation
    
    # Fit plane using RANSAC (robust to outliers like chimneys)
    ransac = RANSACRegressor(
        random_state=42,
        min_samples=10,
        residual_threshold=0.5,  # 0.5m tolerance
        max_trials=100
    )
    
    try:
        ransac.fit(X, z)
    except Exception as e:
        logger.error(f"RANSAC fitting failed: {e}")
        raise ValueError("Could not fit plane to elevation data")
    
    # Get plane coefficients: z = ax + by + c
    a = ransac.estimator_.coef_[0]
    b = ransac.estimator_.coef_[1]
    
    # Calculate pitch: arctan(sqrt(a² + b²))
    slope_magnitude = np.sqrt(a**2 + b**2)
    pitch_radians = np.arctan(slope_magnitude)
    pitch_degrees = np.degrees(pitch_radians)
    
    return round(pitch_degrees, 1)


# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
def root():
    """API information"""
    return {
        "service": "NZ Roof Pitch Calculator",
        "version": "1.0.0",
        "data_source": "LINZ Digital Elevation Model",
        "coverage": "New Zealand",
        "endpoints": {
            "/analyze-pitch": "Calculate roof pitch from coordinates",
            "/health": "Health check"
        }
    }


@app.get("/analyze-pitch")
def analyze_pitch(
    lat: float,
    lon: float,
    zoom: int = 18,
    layer: Optional[str] = None
):
    """
    Calculate roof pitch for given coordinates
    
    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)
        zoom: WMTS zoom level (default: 18)
        layer: LINZ layer ID (default: 104708)
    
    Returns:
        {
            "success": true,
            "pitch": 26.4,
            "coordinates": {...},
            "data_source": "..."
        }
    """
    try:
        # Validate coordinates (NZ bounds)
        if not (-47.5 <= lat <= -34.0):
            return {
                "success": False,
                "pitch": None,
                "error": "Latitude outside NZ bounds (-47.5 to -34.0)"
            }
        
        if not (166.0 <= lon <= 179.0):
            return {
                "success": False,
                "pitch": None,
                "error": "Longitude outside NZ bounds (166.0 to 179.0)"
            }
        
        # Use default layer if not specified
        layer_id = layer or DEFAULT_LAYER
        
        # Convert to NZTM
        easting, northing = latlon_to_nztm(lat, lon)
        logger.info(f"NZTM coordinates: E={easting:.1f}, N={northing:.1f}")
        
        # Get tile coordinates
        tile_x, tile_y, pixel_x, pixel_y = nztm_to_tile(easting, northing, zoom)
        logger.info(f"Tile: ({tile_x}, {tile_y}), Pixel: ({pixel_x}, {pixel_y})")
        
        # Fetch DSM data
        elevation_grid = fetch_dsm_tile(tile_x, tile_y, zoom, layer_id)
        
        # Extract roof area
        roof_points = extract_roof_area(elevation_grid, pixel_x, pixel_y)
        logger.info(f"Extracted {len(roof_points)} elevation points")
        
        # Calculate pitch
        pitch = calculate_pitch_from_points(roof_points)
        logger.info(f"Calculated pitch: {pitch}°")
        
        return {
            "success": True,
            "pitch": pitch,
            "coordinates": {
                "lat": lat,
                "lon": lon,
                "easting": round(easting, 1),
                "northing": round(northing, 1)
            },
            "tile_info": {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "zoom": zoom,
                "layer": layer_id
            },
            "points_analyzed": len(roof_points),
            "data_source": "LINZ Digital Elevation Model"
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "success": False,
            "pitch": None,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "pitch": None,
            "error": f"Internal error: {str(e)}"
        }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "roof-pitch-calculator",
        "linz_api_configured": LINZ_API_KEY != "YOUR_LINZ_API_KEY_HERE"
    }


# ========================================
# LOCAL TESTING
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    print("🏠 Starting Roof Pitch Calculator Backend")
    print("📍 Test URL: http://localhost:8000/analyze-pitch?lat=-36.8485&lon=174.7633")
    print("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)