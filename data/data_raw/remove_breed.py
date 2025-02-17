import os

def remove_breed_from_files(directory="."):
    # Walk through all files and directories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip the script itself and any .git directories
            if file == "remove_breed.py" or ".git" in root:
                continue
                
            # Check if "_breed" is in the filename
            if "_breed" in file:
                old_path = os.path.join(root, file)
                new_file = file.replace("_breed", "")
                new_path = os.path.join(root, new_file)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed '{file}' to '{new_file}'")
                except Exception as e:
                    print(f"Error renaming {file}: {str(e)}")

if __name__ == "__main__":
    remove_breed_from_files()
    print("Finished renaming files") 