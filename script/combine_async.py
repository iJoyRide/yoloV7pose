import os
import shutil
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

def sort_key(filename):
    """ Sort based on numerical values in the filename. """
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def process_file_pair(source_a_file, source_b_file, target_directory, source_b_path):
    # Check for None
    if source_b_file is None:
        print(f"source_b_file is None for source_a_file: {source_a_file}")
        return
    
    # Check if source_b_file exists
    if not os.path.exists(source_b_file):
        print(f"source_b_file does not exist: {source_b_file}")
        return
    
    # Calculate relative directory path
    relative_dir = os.path.relpath(os.path.dirname(source_b_file), source_b_path)
    target_dir_path = os.path.join(target_directory, relative_dir)
    os.makedirs(target_dir_path, exist_ok=True)  # Ensure the target directory exists

    # Define target file path
    target_file = os.path.join(target_dir_path, os.path.basename(source_b_file))
    
    # If source_a_file exists, process and combine the files
    if source_a_file and os.path.exists(source_a_file):
        print(f"Processing pair: {source_a_file}, {source_b_file}")
        with open(source_a_file, 'r') as file_a, open(source_b_file, 'r') as file_b, open(target_file, 'w') as file_target:
            lines_a = file_a.readlines()
            lines_b = file_b.readlines()
            
            # Combine files A and B
            combined_lines = sorted(lines_a + lines_b, key=lambda line: sort_key(line))
            file_target.writelines(combined_lines)
            
        print(f"Finished processing pair: {source_a_file}, {source_b_file}")
    else:
        # If source_a_file does not exist, copy source_b_file to the target directory
        print(f"source_a_file does not exist, copying B file: {source_b_file} -> {target_file}")
        shutil.copy(source_b_file, target_file)
        print(f"Copied {source_b_file} to {target_file}")


async def process_files_in_batches(executor, source_a_files, source_b_files, target_directory, source_b_path):
    loop = asyncio.get_running_loop()

    for i in range(0, len(source_b_files), 20):
        batch = source_b_files[i:i + 20]
        tasks = []
        for source_b_file in batch:
            relative_path = os.path.relpath(source_b_file, source_b_path)
            source_a_file = source_a_files.get(relative_path)

            task = loop.run_in_executor(
                executor, process_file_pair, source_a_file, source_b_file, target_directory, source_b_path
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        print(f"Finished batch {i // 20 + 1}")


async def main():
    
    # source_a_path = r"/app/Desktop/yolov7pose/runs/detect/exp7/extra_set_val"
    # source_b_path = r"/app/Desktop/yolov7_cy/runs/detect/exp7"
    # target_directory = r"/app/Desktop/yolov7pose/runs/detect/that"
    
    source_a_path = r"/app/Desktop/yolov7_cy/runs/detect/exp8"
    source_b_path = r"/app/Desktop/yolov7pose/runs/detect/that"
    target_directory = r"/app/data/labels/val/extra_set_val"

    # Get a list of txt files in source A
    source_a_files = {os.path.relpath(os.path.join(dp, f), source_a_path): os.path.join(dp, f)
                      for dp, dn, filenames in os.walk(source_a_path) for f in filenames if f.endswith('.txt')}
    print(f"Found {len(source_a_files)} files in source A.")

    # Get a list of txt files in source B
    source_b_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_b_path) for f in filenames if f.endswith('.txt')]
    print(f"Found {len(source_b_files)} files in source B.")

    # Sort the files in source B to maintain order
    source_b_files.sort(key=sort_key)

    print(f"Number of files in source A: {len(source_a_files)}")
    print(f"Number of files in source B: {len(source_b_files)}")

    with ThreadPoolExecutor() as executor:
        await process_files_in_batches(executor, source_a_files, source_b_files, target_directory, source_b_path)

if __name__ == "__main__":
    print("Starting to process files in batches.")
    asyncio.run(main())
    print("Finished processing files in batches.")
