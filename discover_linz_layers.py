"""
LINZ WFS Layer Discovery Script
================================
Fetches GetCapabilities from LINZ and extracts all LiDAR-related layer names.
This will give us the CORRECT typeNames to use in WFS requests.
"""

import requests
import xml.etree.ElementTree as ET
import json

# Your LINZ API key
LINZ_API_KEY = '061f3225981a44d7944df25a0868557b'

# LINZ WFS GetCapabilities URL
capabilities_url = f'https://data.linz.govt.nz/services;key={LINZ_API_KEY}/wfs?service=WFS&version=2.0.0&request=GetCapabilities'

print("🔍 Fetching LINZ WFS GetCapabilities...")
print(f"URL: {capabilities_url}\n")

try:
    response = requests.get(capabilities_url, timeout=30)
    response.raise_for_status()
    
    print(f"✅ Response Status: {response.status_code}")
    print(f"📦 Response Size: {len(response.content)} bytes\n")
    
    # Parse XML
    root = ET.fromstring(response.content)
    
    # Define namespaces
    namespaces = {
        'wfs': 'http://www.opengis.net/wfs/2.0',
        'ows': 'http://www.opengis.net/ows/1.1'
    }
    
    # Find all FeatureType elements
    feature_types = root.findall('.//wfs:FeatureType', namespaces)
    
    print(f"📊 Found {len(feature_types)} total WFS layers\n")
    print("=" * 80)
    
    # Filter for LiDAR/elevation related layers
    lidar_layers = {}
    elevation_layers = {}
    all_layers = {}
    
    for ft in feature_types:
        # Get layer name (this is what goes in typeNames parameter)
        name_elem = ft.find('wfs:Name', namespaces)
        title_elem = ft.find('wfs:Title', namespaces)
        
        if name_elem is not None and title_elem is not None:
            layer_name = name_elem.text
            layer_title = title_elem.text
            
            all_layers[layer_name] = layer_title
            
            # Check if it's LiDAR or elevation related
            title_lower = layer_title.lower()
            name_lower = layer_name.lower()
            
            if 'lidar' in title_lower or 'lidar' in name_lower:
                lidar_layers[layer_name] = layer_title
                
            if 'elevation' in title_lower or 'dem' in title_lower or 'dsm' in title_lower:
                elevation_layers[layer_name] = layer_title
    
    # Print LiDAR layers
    print("\n🎯 LIDAR LAYERS:")
    print("-" * 80)
    for name, title in sorted(lidar_layers.items()):
        print(f"  '{name}': '{title}'")
    
    # Print Elevation layers
    print("\n⛰️  ELEVATION LAYERS:")
    print("-" * 80)
    for name, title in sorted(elevation_layers.items()):
        if name not in lidar_layers:  # Don't duplicate
            print(f"  '{name}': '{title}'")
    
    # Save to JSON file
    output = {
        'lidar_layers': lidar_layers,
        'elevation_layers': elevation_layers,
        'all_layers_count': len(all_layers)
    }
    
    with open('linz_layers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Found {len(lidar_layers)} LiDAR layers")
    print(f"✅ Found {len(elevation_layers)} elevation layers")
    print(f"💾 Full results saved to: linz_layers.json")
    
    # Print Python dictionary format for easy copy/paste
    print("\n" + "=" * 80)
    print("📋 COPY/PASTE THIS INTO YOUR main.py:")
    print("-" * 80)
    print("\nLINZ_LIDAR_LAYERS = {")
    for name, title in sorted(lidar_layers.items()):
        # Create a nice key from the title
        key = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        key = ''.join(c for c in key if c.isalnum() or c == '_')
        print(f"    '{key}': '{name}',  # {title}")
    print("}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
