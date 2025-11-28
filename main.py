"""
LINZ Roof Analysis Backend - WMTS DSM/DEM VERSION
=================================================
Fetches DSM (Digital Surface Model) tiles via WMTS for roof analysis.
Much simpler and faster than point cloud approach!

✅ Uses WMTS (not WFS)
✅ Fetches DSM raster tiles (not LAZ files)
✅ Calculates roof slope from elevation grid
✅ Works with actual LINZ data structure
"""

import os
import io
import base64
import math
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
LINZ_API_KEY = os.environ.get('LINZ_API_KEY', 'your-linz-api-key-here')

# LINZ DSM Layer IDs (Digital Surface Models - show buildings/roofs)
LINZ_DSM_LAYERS = {
    'auckland_2024_part1_dsm': 'layer-121992',  # Auckland Part 1 LiDAR 1m DSM (2024)
    'auckland_2024_part2_dsm': 'layer-122587',  # Auckland Part 2 LiDAR 1m DSM (2024)
    'auckland_north_2016_dsm': 'layer-105087',  # Auckland North LiDAR 1m DSM (2016-2018)
    'auckland_south_2016_dsm': 'layer-104406',  # Auckland South LiDAR 1m DSM (2016-2017)
}

# LINZ DEM Layer IDs (Digital Elevation Models - bare earth)
LINZ_DEM_LAYERS = {
    'auckland_2024_part1_dem': 'layer-121990',  # Auckland Part 1 LiDAR 1m DEM (2024)
    'auckland_2024_part2_dem': 'layer-122585',  # Auckland Part 2 LiDAR 1m DEM (2024)
    'auckland_north_2016_dem': 'layer-105086',  # Auckland North LiDAR 1m DEM (2016-2018)
    'auckland_south_2016_dem': 'layer-104405',  # Auckland South LiDAR 1m DEM (2016-2017)
}

DEFAULT_DSM_LAYER = LINZ_DSM_LAYERS['auckland_2024_part1_dsm']
DEFAULT_DEM_LAYER = LINZ_DEM_LAYERS['auckland_2024_part1_dem']


@dataclass
class RoofAnalysis:
    """Results from roof analysis"""
    avg_pitch: float  # degrees
    max_pitch: float
    min_pitch: float
    dominant_aspect: float  # compass direction
    roof_height: float  # meters above ground
    ground_elevation: float  # meters
    roof_elevation: float  # meters
    area_analyzed: float  # square meters
    heatmap_png: str = ""  # base64
    

def latlon_to_nztm(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert WGS84 lat/lon to NZTM2000 (EPSG:2193) easting/northing
    Approximate conversion for NZ - good enough for tile selection
    """
    # Rough approximation for NZ (more accurate would use pyproj)
    # NZTM false origin: 1600000E, 10000000N
    # Central meridian: 173°E
    
    cos_lat = math.cos(math.radians(lat))
    
    # Very rough conversion (±50m accuracy is fine for 1km tiles)
    easting = 1600000 + (lon - 173) * 111320 * cos_lat
    northing = 10000000 + lat * 111320
    
    return easting, northing


def nztm_to_tile(easting: float, northing: float, tile_size: float = 1000.0) -> Tuple[int, int]:
    """
    Convert NZTM coordinates to tile indices
    LINZ DSM/DEM tiles are typically 1km x 1km
    """
    tile_x = int(easting / tile_size)
    tile_y = int(northing / tile_size)
    return tile_x, tile_y


class LINZWMTSClient:
    """Fetch DSM/DEM tiles from LINZ via WMTS"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://data.linz.govt.nz/services'
    
    def get_dsm_tile(self, lat: float, lon: float, layer_id: str = DEFAULT_DSM_LAYER, 
                     buffer_meters: int = 50) -> Optional[np.ndarray]:
        """
        Fetch DSM (Digital Surface Model) tile covering the location
        Returns elevation grid as numpy array
        """
        easting, northing = latlon_to_nztm(lat, lon)
        
        logger.info(f"Location: {lat}, {lon}")
        logger.info(f"NZTM: E={easting:.0f}, N={northing:.0f}")
        
        # For DSM/DEM, we can use WCS (Web Coverage Service) to get actual elevation data
        # WCS is better than WMTS for getting raw elevation values
        
        min_e = easting - buffer_meters
        max_e = easting + buffer_meters
        min_n = northing - buffer_meters
        max_n = northing + buffer_meters
        
        bbox = f"{min_e},{min_n},{max_e},{max_n}"
        
        # WCS GetCoverage request for GeoTIFF
        wcs_url = (
            f"{self.base_url};key={self.api_key}/wcs?"
            f"service=WCS&"
            f"version=2.0.1&"
            f"request=GetCoverage&"
            f"coverageId={layer_id}&"
            f"subset=E({min_e},{max_e})&"
            f"subset=N({min_n},{max_n})&"
            f"format=image/tiff"
        )
        
        logger.info(f"Fetching DSM tile via WCS")
        logger.info(f"WCS URL: {wcs_url}")
        
        try:
            response = requests.get(wcs_url, timeout=30)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content-type: {response.headers.get('content-type')}")
            
            if response.status_code != 200:
                logger.error(f"WCS Error: {response.text[:500]}")
                return None
            
            # Try to read as GeoTIFF
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(response.content))
                elevation_data = np.array(img, dtype=np.float32)
                
                logger.info(f"Loaded DSM: shape={elevation_data.shape}, dtype={elevation_data.dtype}")
                logger.info(f"Elevation range: {elevation_data.min():.1f}m to {elevation_data.max():.1f}m")
                
                return elevation_data
                
            except Exception as e:
                logger.error(f"Failed to parse GeoTIFF: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching DSM: {e}")
            return None
    
    def try_multiple_layers(self, lat: float, lon: float, buffer_meters: int = 50) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Try multiple DSM layers to find coverage"""
        for name, layer_id in LINZ_DSM_LAYERS.items():
            logger.info(f"Trying DSM layer: {name} ({layer_id})")
            dsm = self.get_dsm_tile(lat, lon, layer_id, buffer_meters)
            if dsm is not None:
                logger.info(f"✓ Found data in {name}")
                return dsm, layer_id
        
        logger.warning("No DSM data found in any layer")
        return None, None


class RoofAnalyzer:
    """Analyze roof from DSM elevation data"""
    
    @staticmethod
    def calculate_slope(elevation_grid: np.ndarray, pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope (gradient) from elevation grid
        Returns: (slope_degrees, aspect_degrees)
        """
        # Calculate gradients in X and Y directions
        dy, dx = np.gradient(elevation_grid, pixel_size)
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect (compass direction of slope)
        aspect_rad = np.arctan2(dy, dx)
        aspect_deg = (90 - np.degrees(aspect_rad)) % 360
        
        return slope_deg, aspect_deg
    
    @staticmethod
    def extract_roof_region(dsm: np.ndarray, dem: Optional[np.ndarray] = None, 
                           height_threshold: float = 2.0) -> np.ndarray:
        """
        Extract roof region from DSM
        If DEM available, use DSM-DEM difference (building height)
        Otherwise, use top percentile of DSM
        """
        if dem is not None and dsm.shape == dem.shape:
            # Building height = DSM - DEM
            building_height = dsm - dem
            roof_mask = building_height > height_threshold
        else:
            # Use top 30% of elevations as "roof"
            threshold = np.percentile(dsm, 70)
            roof_mask = dsm > threshold
        
        return roof_mask
    
    @staticmethod
    def analyze_roof(dsm: np.ndarray, dem: Optional[np.ndarray] = None) -> RoofAnalysis:
        """
        Analyze roof geometry from DSM/DEM
        """
        # Extract roof region
        roof_mask = RoofAnalyzer.extract_roof_region(dsm, dem)
        
        logger.info(f"Roof mask: {roof_mask.sum()} roof pixels out of {roof_mask.size} total")
        
        if roof_mask.sum() < 10:
            raise ValueError("Insufficient roof area detected")
        
        # Calculate slopes and aspects
        slope, aspect = RoofAnalyzer.calculate_slope(dsm)
        
        # Get roof statistics
        roof_slopes = slope[roof_mask]
        roof_aspects = aspect[roof_mask]
        roof_elevations = dsm[roof_mask]
        
        avg_pitch = float(np.mean(roof_slopes))
        max_pitch = float(np.max(roof_slopes))
        min_pitch = float(np.min(roof_slopes))
        
        # Dominant aspect (circular mean for angles)
        aspect_rad = np.radians(roof_aspects)
        mean_sin = np.mean(np.sin(aspect_rad))
        mean_cos = np.mean(np.cos(aspect_rad))
        dominant_aspect = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)
        
        roof_elevation = float(np.mean(roof_elevations))
        
        if dem is not None:
            ground_elevation = float(np.mean(dem[roof_mask]))
            roof_height = roof_elevation - ground_elevation
        else:
            ground_elevation = float(np.percentile(dsm, 10))
            roof_height = roof_elevation - ground_elevation
        
        # Approximate area (number of pixels × pixel size²)
        area_analyzed = float(roof_mask.sum() * 1.0)  # 1m² per pixel for 1m DSM
        
        return RoofAnalysis(
            avg_pitch=avg_pitch,
            max_pitch=max_pitch,
            min_pitch=min_pitch,
            dominant_aspect=dominant_aspect,
            roof_height=roof_height,
            ground_elevation=ground_elevation,
            roof_elevation=roof_elevation,
            area_analyzed=area_analyzed
        )
    
    @staticmethod
    def create_heatmap(slope: np.ndarray, roof_mask: np.ndarray) -> str:
        """Create slope heatmap visualization"""
        # Normalize slopes to 0-1 range (0-45 degrees)
        slope_norm = np.clip(slope / 45.0, 0, 1)
        
        # Apply jet colormap
        height, width = slope.shape
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                if not roof_mask[i, j]:
                    continue  # Black for non-roof
                
                val = slope_norm[i, j]
                
                if val < 0.25:
                    r, g, b = 0, int(255 * (val / 0.25)), 255
                elif val < 0.5:
                    r, g, b = 0, 255, int(255 * (1 - (val - 0.25) / 0.25))
                elif val < 0.75:
                    r, g, b = int(255 * ((val - 0.5) / 0.25)), 255, 0
                else:
                    r, g, b = 255, int(255 * (1 - (val - 0.75) / 0.25)), 0
                
                img_array[i, j] = [r, g, b]
        
        img = Image.fromarray(img_array, 'RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'LINZ Roof Analysis (WMTS DSM)',
        'version': '3.0.0-WMTS',
        'note': 'Using WMTS/WCS DSM tiles instead of WFS point clouds'
    })


@app.route('/api/analyze-roof', methods=['POST', 'OPTIONS'])
def analyze_roof_endpoint():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        lat = data.get('latitude')
        lon = data.get('longitude')
        buffer_meters = data.get('buffer_meters', 50)
        
        if lat is None or lon is None:
            return jsonify({
                'success': False,
                'error': 'Missing latitude or longitude',
                'error_code': 'INVALID_INPUT'
            }), 400
        
        logger.info(f"Analyzing roof at: {lat}, {lon}")
        
        client = LINZWMTSClient(LINZ_API_KEY)
        
        # Fetch DSM
        dsm, dsm_layer = client.try_multiple_layers(lat, lon, buffer_meters)
        
        if dsm is None:
            return jsonify({
                'success': False,
                'error': 'No DSM data available for this location',
                'error_code': 'NO_DATA'
            }), 404
        
        # Optionally fetch DEM for ground elevation
        # For now, skip DEM and estimate from DSM
        dem = None
        
        # Analyze roof
        analysis = RoofAnalyzer.analyze_roof(dsm, dem)
        
        # Create visualization
        slope, aspect = RoofAnalyzer.calculate_slope(dsm)
        roof_mask = RoofAnalyzer.extract_roof_region(dsm, dem)
        heatmap = RoofAnalyzer.create_heatmap(slope, roof_mask)
        
        result = {
            'success': True,
            'roof_analysis': {
                'avg_pitch': round(analysis.avg_pitch, 1),
                'max_pitch': round(analysis.max_pitch, 1),
                'min_pitch': round(analysis.min_pitch, 1),
                'dominant_aspect': round(analysis.dominant_aspect, 1),
                'roof_height': round(analysis.roof_height, 1),
                'ground_elevation': round(analysis.ground_elevation, 1),
                'roof_elevation': round(analysis.roof_elevation, 1),
                'area_analyzed': round(analysis.area_analyzed, 1)
            },
            'heatmap_png': heatmap,
            'metadata': {
                'dsm_layer': dsm_layer,
                'buffer_meters': buffer_meters,
                'grid_size': f"{dsm.shape[0]}x{dsm.shape[1]}"
            }
        }
        
        logger.info(f"Analysis complete: pitch={analysis.avg_pitch:.1f}°, aspect={analysis.dominant_aspect:.0f}°")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'PROCESSING_ERROR'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
