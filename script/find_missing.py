import os
import asyncio

# Paths to folder A and folder B
folder_b_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\combined"
folder_a_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\labels\val"

# Path to the missing files text file
missing_files_txt_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\missing_files.txt"

async def write_missing_files(missing_files):
    """Asynchronously write missing files to the text file."""
    async with asyncio.Lock():
        with open(missing_files_txt_path, "a") as missing_files_file:
            for missing_file in missing_files:
                missing_files_file.write(missing_file + "\n")
            print(f"{len(missing_files)} missing files written to '{missing_files_txt_path}'")

async def main():
    missing_files_batch = []
    batch_size = 50

    for folder_a_root, _, folder_a_files in os.walk(folder_a_path):
        for folder_a_file in folder_a_files:
            # Get the relative path of the file within folder A
            relative_path = os.path.relpath(os.path.join(folder_a_root, folder_a_file), start=folder_a_path)
            
            # Create the corresponding path in folder B
            folder_b_file_path = os.path.join(folder_b_path, relative_path)
            
            # Check if the file exists in folder B
            if not os.path.exists(folder_b_file_path):
                print(f"Missing file detected: {relative_path}")
                missing_files_batch.append(relative_path)

                # If batch size is reached, write to file
                if len(missing_files_batch) >= batch_size:
                    await write_missing_files(missing_files_batch)
                    missing_files_batch = []
    
    # Write remaining missing files (if any) after loop completion
    if missing_files_batch:
        await write_missing_files(missing_files_batch)

# Run the asyncio event loop
asyncio.run(main())
