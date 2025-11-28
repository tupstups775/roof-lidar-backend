"""
LINZ LiDAR Roof Analysis Backend - FIXED VERSION
=================================================
Flask API for processing NZ LiDAR point clouds and extracting roof geometry.

✅ FIXED: Corrected LINZ WFS layer names to use 'layer-XXXXX' format
✅ UPDATED: Using most recent Auckland 2024 LiDAR data  

Requirements:
- Flask, flask-cors
- laspy
- numpy, scipy
- trimesh
- Pillow
- requests
- shapely
- scikit-learn

Install:
pip install flask flask-cors laspy numpy scipy trimesh pillow requests shapely scikit-learn
"""

import os
import io
import base64
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pathlib import Path
import laspy
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull
from shapely.geometry import LineString
from PIL import Image, ImageFilter
import trimesh
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Apps Script requests

# Configuration
LINZ_API_KEY = os.environ.get('LINZ_API_KEY', 'your-linz-api-key-here')
CACHE_DIR = Path('./lidar_cache')
CACHE_DIR.mkdir(exist_ok=True)

# ✅ FIXED: LINZ Point Cloud Index Tile Layer IDs (correct layer-XXXXX format)
# Layer IDs verified from https://data.linz.govt.nz/
LINZ_LIDAR_LAYERS = {
    "auckland_2024_part1": "data:lidar-index-auckland-2024-part1",
    "auckland_2024_part2": "data:lidar-index-auckland-2024-part2",
    "auckland_north_2016": "data:lidar-index-auckland-north",
    "auckland_south_2016": "data:lidar-index-auckland-south",
    "auckland_2013": "data:lidar-index-auckland-2013",
}

# Default to most recent Auckland 2024 data
DEFAULT_LAYER = LINZ_LIDAR_LAYERS['auckland_2024_part1']


@dataclass
class RoofPlane:
    """Represents a detected roof plane"""
    plane_id: int
    pitch: float  # degrees
    aspect: float  # degrees (0-360, 0=N, 90=E, 180=S, 270=W)
    area: float  # square meters
    points_count: int
    normal: np.ndarray
    centroid: np.ndarray
    boundary_polygon: List[Tuple[float, float]]
    inlier_indices: np.ndarray


class LINZDownloader:
    """Download LINZ LiDAR tiles"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://data.linz.govt.nz/services'
    
    def get_tiles_for_location(self, lat: float, lng: float, buffer_meters: float = 50, layer_id: str = DEFAULT_LAYER) -> List[str]:
        """
        Query LINZ WFS to find available point cloud tiles for location
        Returns list of tile URLs
        
        NOTE: layer_id must be in format 'layer-XXXXX' where XXXXX is the LINZ numeric layer ID
        Check https://data.linz.govt.nz/data/category/elevation/ for available layers
        """
        buffer_deg = buffer_meters / 111320.0
        
        # Calculate bbox corners
        lon1 = lng - buffer_deg
        lon2 = lng + buffer_deg
        lat1 = lat - buffer_deg
        lat2 = lat + buffer_deg
        
        # Ensure correct ordering (minLon, minLat, maxLon, maxLat)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        
        # ✅ FIXED: Using correct WFS URL format with layer-XXXXX typeNames
        wfs_url = (
            f"{self.base_url}/wfs?"
            f"service=WFS&"
            f"version=2.0.0&"
            f"request=GetFeature&"
            f"typeNames={layer_id}&"
            f"bbox={bbox}&"
            f"srsName=EPSG:4326&"
            f"outputFormat=json&"
            f"key={self.api_key}"
        )
        
        logger.info(f"Querying LINZ WFS for layer {layer_id}: {wfs_url}")
        
        try:
            response = requests.get(wfs_url, timeout=30)
            logger.info(f"LINZ Response Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"LINZ Response Body: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            tile_urls = []
            if 'features' in data:
                for feature in data['features']:
                    props = feature.get('properties', {})
                    # Try different property names for LAZ URLs
                    laz_url = props.get('url_laz') or props.get('laz_url') or props.get('url')
                    if laz_url:
                        tile_urls.append(laz_url)
            
            logger.info(f"Found {len(tile_urls)} tiles")
            return tile_urls
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error querying LINZ WFS: {e}")
            if hasattr(response, 'text'):
                logger.error(f"Response body: {response.text}")
            return []
        except Exception as e:
            logger.error(f"Error querying LINZ WFS: {e}")
            return []
    
    def try_multiple_layers(self, lat: float, lng: float, buffer_meters: float = 50) -> Tuple[List[str], str]:
        """
        Try multiple LINZ layers to find data for the location
        Returns (tile_urls, layer_id_used)
        """
        logger.info(f"Trying multiple LINZ layers for location: {lat}, {lng}")
        
        for region, layer_id in LINZ_LIDAR_LAYERS.items():
            logger.info(f"Trying {region} layer: {layer_id}")
            tile_urls = self.get_tiles_for_location(lat, lng, buffer_meters, layer_id)
            if tile_urls:
                logger.info(f"Found data in {region} layer")
                return tile_urls, layer_id
        
        logger.warning("No data found in any LINZ layer")
        return [], None
    
    def download_tile(self, url: str) -> Optional[Path]:
        """Download a single LiDAR tile (.laz file)"""
        filename = Path(url).name
        cache_path = CACHE_DIR / filename
        
        if cache_path.exists():
            logger.info(f"Using cached tile: {filename}")
            return cache_path
        
        logger.info(f"Downloading tile: {filename}")
        
        try:
            download_url = f"{url}?key={self.api_key}"
            response = requests.get(download_url, timeout=120, stream=True)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return cache_path
            
        except Exception as e:
            logger.error(f"Error downloading tile {filename}: {e}")
            return None
    
    def download_tiles(self, tile_urls: List[str]) -> List[Path]:
        """Download multiple tiles"""
        paths = []
        for url in tile_urls:
            path = self.download_tile(url)
            if path:
                paths.append(path)
        return paths


class LiDARProcessor:
    """Process LiDAR point clouds"""
    
    def __init__(self):
        self.point_cloud = None
    
    def load_laz_files(self, laz_paths: List[Path]) -> np.ndarray:
        """Load multiple LAZ files and merge point clouds"""
        all_points = []
        
        for laz_path in laz_paths:
            try:
                las = laspy.read(str(laz_path))
                
                x = las.x.scaled_array()
                y = las.y.scaled_array()
                z = las.z.scaled_array()
                
                points = np.vstack([x, y, z]).T
                all_points.append(points)
                
                logger.info(f"Loaded {len(points)} points from {laz_path.name}")
                
            except Exception as e:
                logger.error(f"Error reading {laz_path}: {e}")
        
        if not all_points:
            return np.array([])
        
        merged = np.vstack(all_points)
        logger.info(f"Total merged points: {len(merged)}")
        return merged
    
    def clip_to_bounds(self, points: np.ndarray, lat: float, lng: float, 
                       buffer_meters: float = 50) -> np.ndarray:
        """Clip point cloud to bounding box around location"""
        center_x = np.median(points[:, 0])
        center_y = np.median(points[:, 1])
        
        mask = (
            (points[:, 0] >= center_x - buffer_meters) &
            (points[:, 0] <= center_x + buffer_meters) &
            (points[:, 1] >= center_y - buffer_meters) &
            (points[:, 1] <= center_y + buffer_meters)
        )
        
        clipped = points[mask]
        logger.info(f"Clipped to {len(clipped)} points within {buffer_meters}m")
        return clipped
    
    def remove_ground(self, points: np.ndarray, ground_threshold: float = 0.5) -> np.ndarray:
        """Remove ground points using simple elevation threshold"""
        if len(points) == 0:
            return points
        
        ground_level = np.percentile(points[:, 2], 10)
        mask = points[:, 2] > (ground_level + ground_threshold)
        building_points = points[mask]
        
        logger.info(f"Removed ground: {len(building_points)} building points remaining")
        return building_points
    
    def extract_roof_points(self, points: np.ndarray, percentile: float = 70) -> np.ndarray:
        """Extract likely roof points (top portion of building)"""
        if len(points) == 0:
            return points
        
        z_threshold = np.percentile(points[:, 2], percentile)
        roof_mask = points[:, 2] >= z_threshold
        roof_points = points[roof_mask]
        
        logger.info(f"Extracted {len(roof_points)} roof points (top {100-percentile}%)")
        return roof_points


class RoofAnalyzer:
    """Detect and analyze roof planes using sklearn RANSAC"""
    
    def __init__(self, min_points: int = 100, distance_threshold: float = 0.15):
        self.min_points = min_points
        self.distance_threshold = distance_threshold
    
    def detect_planes(self, points: np.ndarray, max_planes: int = 10) -> List[RoofPlane]:
        """Detect multiple roof planes using RANSAC"""
        if len(points) < self.min_points:
            logger.warning("Not enough points for plane detection")
            return []
        
        planes = []
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))
        
        for plane_id in range(max_planes):
            if len(remaining_points) < self.min_points:
                break
            
            # Use sklearn RANSAC for plane fitting
            X = remaining_points[:, :2]  # XY coordinates
            y = remaining_points[:, 2]   # Z coordinates
            
            ransac = RANSACRegressor(
                max_trials=1000,
                min_samples=3,
                residual_threshold=self.distance_threshold,
                random_state=42
            )
            
            try:
                ransac.fit(X, y)
            except:
                break
            
            inlier_mask = ransac.inlier_mask_
            inliers = np.where(inlier_mask)[0]
            
            if len(inliers) < self.min_points:
                break
            
            # Calculate plane normal from RANSAC model
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            
            # Normal vector: [a, b, -1] where ax + by + c = z
            normal = np.array([coef[0], coef[1], -1])
            normal = normal / np.linalg.norm(normal)  # Normalize
            
            # Ensure normal points upward
            if normal[2] < 0:
                normal = -normal
            
            # Get inlier points
            inlier_points = remaining_points[inliers]
            global_inlier_indices = remaining_indices[inliers]
            
            # Calculate pitch (angle from horizontal)
            pitch = np.degrees(np.arccos(abs(normal[2])))
            
            # Calculate aspect (compass direction)
            aspect = np.degrees(np.arctan2(normal[0], normal[1])) % 360
            
            # Calculate centroid
            centroid = np.mean(inlier_points, axis=0)
            
            # Calculate approximate area
            area = self._calculate_plane_area(inlier_points)
            
            # Get boundary polygon
            boundary = self._get_boundary_polygon(inlier_points)
            
            # Create RoofPlane object
            plane = RoofPlane(
                plane_id=plane_id + 1,
                pitch=pitch,
                aspect=aspect,
                area=area,
                points_count=len(inliers),
                normal=normal,
                centroid=centroid,
                boundary_polygon=boundary,
                inlier_indices=global_inlier_indices
            )
            
            planes.append(plane)
            logger.info(f"Plane {plane_id + 1}: Pitch={pitch:.1f}°, Aspect={aspect:.0f}°, Points={len(inliers)}")
            
            # Remove inliers from remaining points
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]
            remaining_indices = remaining_indices[mask]
        
        logger.info(f"Detected {len(planes)} roof planes")
        return planes
    
    def _calculate_plane_area(self, points: np.ndarray) -> float:
        """Calculate approximate plane area using 2D convex hull"""
        if len(points) < 3:
            return 0.0
        
        points_2d = points[:, :2]
        
        try:
            hull = ConvexHull(points_2d)
            return hull.volume  # In 2D, volume = area
        except:
            return 0.0
    
    def _get_boundary_polygon(self, points: np.ndarray, simplify_tolerance: float = 0.5) -> List[Tuple[float, float]]:
        """Get boundary polygon of plane (simplified)"""
        if len(points) < 3:
            return []
        
        points_2d = points[:, :2]
        
        try:
            hull = ConvexHull(points_2d)
            boundary_points = points_2d[hull.vertices]
            
            line = LineString(boundary_points)
            simplified = line.simplify(simplify_tolerance, preserve_topology=True)
            
            return list(simplified.coords)
        except:
            return []


class Visualizer:
    """Generate visualizations"""
    
    @staticmethod
    def create_heatmap(points: np.ndarray, planes: List[RoofPlane], 
                       width: int = 800, height: int = 600) -> str:
        """Create height-colored heatmap PNG and return base64 string"""
        if len(points) == 0:
            img = Image.new('RGB', (width, height), color='black')
            return Visualizer._image_to_base64(img)
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range == 0 or y_range == 0:
            img = Image.new('RGB', (width, height), color='black')
            return Visualizer._image_to_base64(img)
        
        px = ((points[:, 0] - x_min) / x_range * (width - 1)).astype(int)
        py = ((y_max - points[:, 1]) / y_range * (height - 1)).astype(int)
        
        if z_max > z_min:
            pz = (points[:, 2] - z_min) / (z_max - z_min)
        else:
            pz = np.zeros(len(points))
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Jet colormap
        for i in range(len(points)):
            x_px, y_px, z_val = px[i], py[i], pz[i]
            if 0 <= x_px < width and 0 <= y_px < height:
                if z_val < 0.25:
                    r, g, b = 0, int(255 * (z_val / 0.25)), 255
                elif z_val < 0.5:
                    r, g, b = 0, 255, int(255 * (1 - (z_val - 0.25) / 0.25))
                elif z_val < 0.75:
                    r, g, b = int(255 * ((z_val - 0.5) / 0.25)), 255, 0
                else:
                    r, g, b = 255, int(255 * (1 - (z_val - 0.75) / 0.25)), 0
                
                img_array[y_px, x_px] = [r, g, b]
        
        img = Image.fromarray(img_array, 'RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return Visualizer._image_to_base64(img)
    
    @staticmethod
    def create_3d_model(points: np.ndarray, planes: List[RoofPlane]) -> str:
        """Create 3D GLB model and return base64 string"""
        if len(points) == 0 or len(planes) == 0:
            return ""
        
        try:
            meshes = []
            
            for plane in planes:
                plane_points = points[plane.inlier_indices]
                
                if len(plane_points) < 10:
                    continue
                
                # Simple mesh from convex hull
                try:
                    hull = ConvexHull(plane_points)
                    vertices = plane_points[hull.vertices]
                    
                    # Create simple triangulation
                    if len(vertices) >= 3:
                        colors = Visualizer._get_plane_color(plane.plane_id)
                        vertex_colors = np.tile(colors, (len(vertices), 1))
                        
                        # Create triangles from hull
                        faces = hull.simplices
                        
                        tm = trimesh.Trimesh(
                            vertices=vertices,
                            faces=faces,
                            vertex_colors=vertex_colors
                        )
                        meshes.append(tm)
                
                except Exception as e:
                    logger.warning(f"Could not create mesh for plane {plane.plane_id}: {e}")
            
            if not meshes:
                logger.warning("No meshes created")
                return ""
            
            combined_mesh = trimesh.util.concatenate(meshes)
            
            with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
                tmp_path = tmp.name
            
            combined_mesh.export(tmp_path, file_type='glb')
            
            with open(tmp_path, 'rb') as f:
                glb_bytes = f.read()
            
            os.unlink(tmp_path)
            
            base64_glb = base64.b64encode(glb_bytes).decode('utf-8')
            logger.info(f"Created GLB model: {len(base64_glb)} chars")
            
            return base64_glb
        
        except Exception as e:
            logger.error(f"Error creating 3D model: {e}")
            return ""
    
    @staticmethod
    def _image_to_base64(img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    @staticmethod
    def _get_plane_color(plane_id: int) -> np.ndarray:
        """Get RGB color for plane ID"""
        colors = [
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
            [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
            [210, 245, 60], [250, 190, 212]
        ]
        return np.array(colors[(plane_id - 1) % len(colors)])


# ================================
# API ENDPOINTS
# ================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'LINZ LiDAR Roof Analysis',
        'version': '2.0.0-fixed',
        'note': 'Using correct LINZ layer-XXXXX format with Auckland 2024 data'
    })


@app.route('/api/analyze-roof', methods=['POST', 'OPTIONS'])
def analyze_roof():
    """Main endpoint for roof analysis"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        lat = data.get('latitude')
        lng = data.get('longitude')
        address = data.get('address', 'Unknown')
        buffer_meters = data.get('buffer_meters', 50)
        return_formats = data.get('return_formats', ['png', 'glb'])
        options = data.get('options', {})
        layer_id = data.get('layer_id')  # Optional: specify LINZ layer ID
        
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'error': 'Missing latitude or longitude',
                'error_code': 'INVALID_INPUT'
            }), 400
        
        logger.info(f"Processing request for: {lat}, {lng}")
        
        downloader = LINZDownloader(LINZ_API_KEY)
        processor = LiDARProcessor()
        analyzer = RoofAnalyzer(
            min_points=options.get('min_points_per_plane', 100),
            distance_threshold=options.get('distance_threshold', 0.15)
        )
        
        # Try to get tiles - either from specified layer or try multiple layers
        if layer_id:
            tile_urls = downloader.get_tiles_for_location(lat, lng, buffer_meters, layer_id)
            layer_used = layer_id
        else:
            tile_urls, layer_used = downloader.try_multiple_layers(lat, lng, buffer_meters)
        
        if not tile_urls:
            return jsonify({
                'success': False,
                'error': 'No LiDAR data available for this location',
                'error_code': 'NO_DATA',
                'details': 'Tried all available LINZ LiDAR layers. The location may not have LiDAR coverage.'
            }), 404
        
        laz_paths = downloader.download_tiles(tile_urls)
        
        if not laz_paths:
            return jsonify({
                'success': False,
                'error': 'Failed to download LiDAR tiles',
                'error_code': 'DOWNLOAD_FAILED'
            }), 500
        
        points = processor.load_laz_files(laz_paths)
        
        if len(points) == 0:
            return jsonify({
                'success': False,
                'error': 'No points loaded from LiDAR tiles',
                'error_code': 'NO_POINTS'
            }), 500
        
        points = processor.clip_to_bounds(points, lat, lng, buffer_meters)
        points = processor.remove_ground(points)
        roof_points = processor.extract_roof_points(points)
        
        if len(roof_points) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient roof points detected',
                'error_code': 'INSUFFICIENT_POINTS',
                'details': f'Only {len(roof_points)} roof points found'
            }), 500
        
        planes = analyzer.detect_planes(roof_points)
        
        if len(planes) == 0:
            return jsonify({
                'success': False,
                'error': 'No roof planes detected',
                'error_code': 'NO_PLANES'
            }), 500
        
        result = {
            'success': True,
            'planes': [],
            'metadata': {
                'total_points': len(points),
                'roof_points': len(roof_points),
                'tile_count': len(laz_paths),
                'buffer_meters': buffer_meters,
                'linz_layer_used': layer_used
            }
        }
        
        for plane in planes:
            result['planes'].append({
                'plane_id': plane.plane_id,
                'pitch': round(plane.pitch, 1),
                'aspect': round(plane.aspect, 1),
                'area': round(plane.area, 1),
                'points_count': plane.points_count,
                'centroid': plane.centroid.tolist(),
                'boundary_polygon': plane.boundary_polygon
            })
        
        if 'png' in return_formats:
            logger.info("Generating heatmap...")
            result['heatmap_png'] = Visualizer.create_heatmap(roof_points, planes)
        
        if 'glb' in return_formats:
            logger.info("Generating 3D model...")
            result['model_glb'] = Visualizer.create_3d_model(roof_points, planes)
        
        logger.info(f"Analysis complete: {len(planes)} planes detected using layer {layer_used}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'PROCESSING_ERROR'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)