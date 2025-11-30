"""
LINZ DSM PNG Decoder
Decodes elevation data from LINZ WMTS PNG tiles
"""

import numpy as np
from PIL import Image
import io
import requests


def decode_linz_dsm_png(png_bytes):
    """
    Decode LINZ DSM elevation data from PNG tile
    
    LINZ encodes elevation in RGB channels:
    - Elevation (m) = R * 256 + G + B/256 - 10000
    
    Args:
        png_bytes: PNG image data as bytes
    
    Returns:
        numpy array of elevation values (height x width)
    """
    # Open PNG image
    img = Image.open(io.BytesIO(png_bytes))
    img_array = np.array(img)
    
    # Extract RGB channels
    R = img_array[:, :, 0].astype(np.float32)
    G = img_array[:, :, 1].astype(np.float32)
    B = img_array[:, :, 2].astype(np.float32)
    
    # Decode elevation using LINZ formula
    elevation = R * 256 + G + B / 256 - 10000
    
    # Handle no-data values (typically where RGB = (255, 255, 255) or (0, 0, 0))
    # Mark as NaN for filtering
    no_data_mask = (R == 255) & (G == 255) & (B == 255)
    elevation[no_data_mask] = np.nan
    
    return elevation


def fetch_and_decode_dsm_tile(tile_x, tile_y, zoom, api_key):
    """
    Fetch LINZ WMTS tile and decode elevation data
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        api_key: LINZ API key
    
    Returns:
        2D numpy array of elevation values
    """
    url = f"https://data.linz.govt.nz/services;key={api_key}/wmts/1.0.0/layer/104708/tile/NZTM2000Quad/{zoom}/{tile_y}/{tile_x}.png"
    
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch tile: HTTP {response.status_code}")
    
    elevation_grid = decode_linz_dsm_png(response.content)
    return elevation_grid


def extract_roof_area(elevation_grid, pixel_x, pixel_y, buffer_pixels=20):
    """
    Extract a buffer area around the target pixel from elevation grid
    
    Args:
        elevation_grid: 2D array of elevation values
        pixel_x: Target pixel X coordinate within tile
        pixel_y: Target pixel Y coordinate within tile
        buffer_pixels: Number of pixels to extract around target
    
    Returns:
        Nx3 array of (x, y, elevation) points
    """
    height, width = elevation_grid.shape
    
    # Calculate extraction bounds
    x_min = max(0, pixel_x - buffer_pixels)
    x_max = min(width, pixel_x + buffer_pixels)
    y_min = max(0, pixel_y - buffer_pixels)
    y_max = min(height, pixel_y + buffer_pixels)
    
    # Extract subset
    subset = elevation_grid[y_min:y_max, x_min:x_max]
    
    # Convert to point cloud format (x, y, z)
    points = []
    for i in range(subset.shape[0]):
        for j in range(subset.shape[1]):
            z = subset[i, j]
            if not np.isnan(z):  # Skip no-data values
                points.append([j, i, z])
    
    return np.array(points)


# Example usage
if __name__ == "__main__":
    # Test with a known tile
    api_key = "YOUR_LINZ_API_KEY"
    tile_x, tile_y, zoom = 12345, 23456, 18  # Replace with actual values
    
    try:
        elevation_grid = fetch_and_decode_dsm_tile(tile_x, tile_y, zoom, api_key)
        print(f"Elevation grid shape: {elevation_grid.shape}")
        print(f"Min elevation: {np.nanmin(elevation_grid):.2f} m")
        print(f"Max elevation: {np.nanmax(elevation_grid):.2f} m")
        print(f"Mean elevation: {np.nanmean(elevation_grid):.2f} m")
        
        # Extract roof area around center pixel
        center_x, center_y = 128, 128
        roof_points = extract_roof_area(elevation_grid, center_x, center_y, buffer_pixels=20)
        print(f"\nExtracted {len(roof_points)} roof points")
        
    except Exception as e:
        print(f"Error: {e}")