import json
import os
import glob

# Set directory path
directory = './Training_Data'

# Get all JSON files
json_files = glob.glob(os.path.join(directory, '*.json'))

# Initialise the merged data list
merged_data = []

# Iterate through and merge all JSON files
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"Warning: {os.path.basename(file_path)} is not a list format, skipped")
        except json.JSONDecodeError:
            print(f"Error: {os.path.basename(file_path)} is not a valid JSON file, skipped")

# Save the merged results
output_file = 'merged_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"Successfully merged {len(json_files)} files -> {output_file}")
print(f"Total data volume: {len(merged_data)} records")