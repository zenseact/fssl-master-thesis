import os
import json

# tqdm adds a progress bar to show loop progress in the terminal
try:
    from tqdm import tqdm
except ImportError:
    print("Tip: Install tqdm ('pip install tqdm') to see the progress bar.")
    tqdm = lambda x, **kwargs: x

def main():
    # Paths — override via env vars ZOD_DATA_DIR and PROJECT_ROOT
    zod_data_dir = os.environ.get("ZOD_DATA_DIR")
    base_dir = os.path.join(zod_data_dir, "single_frames")

    project_root = os.environ.get("PROJECT_ROOT")
    output_file = os.path.join(project_root, "src", "zod_metadata_cache.json")
    
    # 1. Find all frame folders
    print(f"Scanning folders in {base_dir}...")
    try:
        # Create a list of all subdirectories
        frame_folders = [
            os.path.join(base_dir, d) 
            for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        print(f"Found {len(frame_folders)} frames. Starting extraction...")
    except FileNotFoundError:
        print(f"ERROR: Directory {base_dir} not found. Please check the path.")
        return

    # 2. Sequential Extraction (Standard FOR Loop)
    cache = {}
    
    # The loop will iterate over all ~90k folders
    for frame_path in tqdm(frame_folders):
        frame_id = os.path.basename(frame_path)
        meta_file = os.path.join(frame_path, "metadata.json")
        
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
                
                # Save only the 4 necessary metadata fields to minimize RAM usage
                cache[frame_id] = {
                    "country_code": data.get("country_code", "unknown"),
                    "scraped_weather": data.get("scraped_weather", "unknown"),
                    "time_of_day": data.get("time_of_day", "unknown"),
                    "road_type": data.get("road_type", "unknown")
                }
        except Exception:
            # If the file does not exist or is corrupted, safely skip to the next one
            print(f"Warning: Could not read metadata for frame {frame_id}. Skipping.")
            pass
            
    # 3. Save to disk
    print(f"\nSaving cache to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(cache, f) 
        
    print(f"Done! Cache generated successfully. Total entries written: {len(cache)}")

if __name__ == "__main__":
    main()