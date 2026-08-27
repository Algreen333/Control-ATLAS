import argparse
import os
import cv2
import numpy as np
import subprocess

def generate_aruco_texture(marker_id, path):
    """Generates the ArUco PNG file."""
    # Use DICT_4X4_50 (Common standard)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    # Generate 500x500 pixel image
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500)
    
    # Add a white border (Gazebo often bleeds edges)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    cv2.imwrite(path, img)

def generate_sdf(model_name, texture_filename, size):
    """
    Creates the SDF string using PBR (Physically Based Rendering)
    compatible with Gazebo Sim / Ignition.
    """
    # Get absolute path to the texture so Gazebo finds it guaranteed
    # texture_abs_path = os.path.join(os.getcwd(), model_name, "materials", "textures", texture_filename)
    
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{size} {size} 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0 0 0 1</specular>
          <pbr>
            <metal>
              <albedo_map>materials/textures/{texture_filename}</albedo_map>
              <roughness>0.6</roughness>
              <metalness>0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>{size} {size} 0.01</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

def generate_material_script(model_name, texture_filename):
    """Creates the Ogre material script."""
    return f"""
material ArucoMarker/{model_name}
{{
  technique
  {{
    pass
    {{
      texture_unit
      {{
        texture {texture_filename}
      }}
    }}
  }}
}}
"""

def main():
    parser = argparse.ArgumentParser(description="Spawn an ArUco marker in Gazebo.")
    parser.add_argument("--id", type=int, required=True, help="ArUco ID")
    parser.add_argument("--size", type=float, default=0.2, help="Size in meters")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.0)
    
    args = parser.parse_args()
    
    model_name = f"aruco_marker_{args.id}"
    base_dir = os.path.join(os.getcwd(), model_name)
    texture_dir = os.path.join(base_dir, "materials", "textures")
    
    # Create Directory Structure
    os.makedirs(texture_dir, exist_ok=True)
    
    # Generate Texture
    texture_filename = f"marker_{args.id}.png"
    generate_aruco_texture(args.id, os.path.join(texture_dir, texture_filename))
      
    # Generate SDF
    sdf_content = generate_sdf(model_name, texture_filename, args.size)
    sdf_path = os.path.join(base_dir, "model.sdf")
    with open(sdf_path, "w") as f:
        f.write(sdf_content)
        
    # Create model.config (Required for Gazebo to find it)
    config_content = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.5">model.sdf</sdf>
  <description>Procedural ArUco Marker {args.id}</description>
</model>
"""
    with open(os.path.join(base_dir, "model.config"), "w") as f:
        f.write(config_content)

    print(f"Generated model at: {base_dir}")

if __name__ == "__main__":
    main()