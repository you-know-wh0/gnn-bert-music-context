import os
import csv
import json

METADATA_FILE = "data/raw/fma_medium/metadata/fma_metadata/tracks.csv"
AUDIO_DIR = "data/raw/fma_medium/fma_medium"
SPLITS_DIR = "data/splits"

os.makedirs(SPLITS_DIR, exist_ok=True)

def generate_splits():
    print("Reading metadata from:", METADATA_FILE)
    
    # Initialize split dicts
    splits = {
        "training": {},
        "validation": {},
        "test": {}
    }
    
    with open(METADATA_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Read headers
        header0 = next(reader) # Row 0 (e.g. '', 'comments', etc. for album)
        header1 = next(reader) # Row 1 (contains 'split', 'subset', 'genre_top')
        header2 = next(reader) # Row 2 (contains 'track_id')
        
        # Find indices dynamically
        try:
            split_idx = header1.index('split')
            subset_idx = header1.index('subset')
            genre_top_idx = header1.index('genre_top')
        except ValueError as e:
            print("Error finding header columns:", e)
            return
            
        print(f"Dynamic indices - split: {split_idx}, subset: {subset_idx}, genre_top: {genre_top_idx}")
        
        count_all = 0
        count_subset = 0
        count_valid = 0
        
        for row in reader:
            if not row or len(row) <= max(split_idx, subset_idx, genre_top_idx):
                continue
                
            track_id = row[0].strip()
            split_val = row[split_idx].strip()      # training, validation, test
            subset_val = row[subset_idx].strip()    # small, medium, large
            genre_top = row[genre_top_idx].strip()  # Genre name
            
            count_all += 1
            
            # FMA-medium contains both 'small' and 'medium' subset tracks
            if subset_val not in ['small', 'medium']:
                continue
                
            count_subset += 1
            
            # Pad track_id to 6 characters
            padded_id = track_id.zfill(6)
            folder = padded_id[:3]
            mp3_path = os.path.join(AUDIO_DIR, folder, f"{padded_id}.mp3")
            
            # Verify if the MP3 file actually exists locally
            if os.path.exists(mp3_path):
                if split_val in splits:
                    splits[split_val][padded_id] = {
                        "genre": genre_top,
                        "subset": subset_val
                    }
                    count_valid += 1
                    
        print(f"Total rows in metadata: {count_all}")
        print(f"Rows in small/medium subsets: {count_subset}")
        print(f"Verified files found locally: {count_valid}")
        
        # Save splits to JSON
        for split_name, data in splits.items():
            output_path = os.path.join(SPLITS_DIR, f"{split_name}.json")
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(data, out_f, indent=4)
            print(f"Saved {len(data)} items to {output_path}")

if __name__ == "__main__":
    generate_splits()
