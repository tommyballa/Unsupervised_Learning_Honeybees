import os
import shutil
import random

source_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_batch3+4"
val_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/val_split_batch3+4"
test_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_split_batch3+4"

os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

images = sorted(os.listdir(os.path.join(source_dir, "images")))
random.seed(42)
random.shuffle(images)

mid = len(images) // 2
val_images = images[:mid]
test_images = images[mid:]

def copy_files(image_list, dest_dir):
    for img in image_list:
        # copy image
        shutil.copy(os.path.join(source_dir, "images", img), os.path.join(dest_dir, "images", img))
        
        # copy label
        label_file = os.path.splitext(img)[0] + ".txt"
        src_label = os.path.join(source_dir, "labels", label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dest_dir, "labels", label_file))

copy_files(val_images, val_dir)
copy_files(test_images, test_dir)

print(f"Copied {len(val_images)} images to {val_dir}")
print(f"Copied {len(test_images)} images to {test_dir}")
