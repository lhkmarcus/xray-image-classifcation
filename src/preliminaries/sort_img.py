import os
import re
import shutil
import zipfile

#### Unzip data folder
archive_path = "..\\data\\images\\archive.zip"
img_dir = "..\\data\\images"
subfolders = ("chest_xray/test", "chest_xray/train/", "chest_xray/val/")

def unzip(archive_path, img_dir):
    with zipfile.ZipFile(archive_path) as zip_object:
        for file in zip_object.namelist():
            if file.startswith(subfolders):
                zip_object.extract(file, img_dir)

#### Extract images from subfolders
main_dir = "..\\data\\images\\chest_xray"

def extract_img(main_dir):
    if os.path.exists(main_dir):
        try:
            print("Extracting images...")
            for root, _, files in os.walk(main_dir, topdown=True):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, file)
                        shutil.move(img_path, main_dir)
            print("Done.")
        except:
            print("All images have been extracted.")

    ## Remove subfolders
    try:
        for folder in ["test", "train", "val"]:
            shutil.rmtree(f"..\\data\\images\\chest_xray\\{folder}")
    except:
        pass

#### Sort images for each class
patterns = ["normal", "bacteria", "virus"]
ext = "(png|jpg|jpeg)"

def sort_img(main_dir, patterns, ext):
    print("Sorting images...")
    for pattern in patterns:
        new_dir = f"{main_dir}\\{pattern.upper()}\\"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for file in os.listdir(main_dir):
            if re.search(f"({pattern}|{pattern.upper()}).*{ext}$", file):
                shutil.move(f"{main_dir}\\{file}", f"{new_dir}{file}")
        print(f"{pattern.capitalize()}" + " images sorted.")

## Sort any residual files
folder = "NORMAL"

def sort_res_img(main_dir, folder, ext):
    print("Sorting residual images...")
    dest_path = f"{main_dir}\\{folder}\\"
    for file in os.listdir(main_dir):
        if re.search(f".*{ext}$", file):
            shutil.move(f"{main_dir}\\{file}", dest_path)
    print("Done.")

def main():
    unzip(archive_path, img_dir)
    extract_img(main_dir)
    sort_img(main_dir, patterns, ext)
    sort_res_img(main_dir, folder, ext)

if __name__ == "__main__":
    main()