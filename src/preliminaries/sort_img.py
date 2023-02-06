import os
import re
import shutil
from src.config_paths import RAW_DATA_PATH

## Extract all images out from raw folder
def extract_all(parent_folder, drop=None):
    print("--------------------------------------------------------------")
    print("Extracting all image files...")
    print("--------------------------------------------------------------")
    if os.path.exists(parent_folder):
        print("Parent folder is " + parent_folder)
        print("Moving files from subfolders.")
        for root, _, files in os.walk(parent_folder, topdown=True):
            for file in files:
                if file.endswith((".png", ".jpeg", ".jpg")):
                    source = os.path.join(root, file)
                    shutil.move(source, RAW_DATA_PATH)
        print("Extraction complete. Check files.")
        if drop:
            try:
                print("Trying to remove parent folder...")
                shutil.rmtree(parent_folder)
                print("Parent folder removed.")
            except PermissionError:
                print("Removal not permitted.")
    else:
        print("All files have been extracted from subfolders.")
        print("The parent folder " + parent_folder + " has been removed.")
    print("--------------------------------------------------------------")

## Sort images by regular expressions
def sort_files(directory, pattern, ext):
    destination = f"{directory}{pattern.upper()}/"
    file_list = []
    if not os.path.exists(destination):
        print("Creating directory " + destination)
        os.makedirs(destination)
    print("Checking image filenames for " + pattern.capitalize() + " images")
    for i in os.listdir(directory):
        if re.search(f"{pattern}.*{ext}$", i):
            file_list.append(i)
    for j in file_list:
        shutil.move(f"{directory}{j}", f"{destination}{j}")
    print(pattern.capitalize() + " files sorted.")
    print("--------------------------------------------------------------")

## Sort any remaining files
def sort_rem_files(directory, folder, ext):
    destination = f"{directory}{folder}/"
    for k in os.listdir(directory):
        if re.search(f".*{ext}$", k):
            shutil.move(f"{directory}{k}", f"{destination}")

def main(parent_folder, extract=None, sort=None, drop_parent=None):
    if extract:
        extract_all(parent_folder=parent_folder, drop=drop_parent)
    if sort:
        print("--------------------------------------------------------------")
        print("Sorting image files...")
        print("--------------------------------------------------------------")
        directory = f"./data/raw/"
        patterns = ["NORMAL", "bacteria", "virus"]
        ext = ".jpeg"
        for pattern in patterns:
            sort_files(directory=directory, pattern=pattern, ext=ext)
        sort_rem_files(directory=directory, folder="NORMAL", ext=ext)

if __name__ == "__main__":
    main(parent_folder="./data/raw/", 
         extract=True, 
         sort=False, 
         drop_parent=False)