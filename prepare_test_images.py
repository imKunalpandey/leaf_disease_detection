import os
import shutil
import random

source_dir = 'dataset'
target_dir = 'test_images'
num_samples = 5

# Ensure target dir exists
os.makedirs(target_dir, exist_ok=True)

# Select random images from each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    sample_images = random.sample(images, min(num_samples, len(images)))

    for img in sample_images:
        source_img_path = os.path.join(class_path, img)
        target_img_path = os.path.join(target_dir, f'{class_name}_{img}')
        shutil.copyfile(source_img_path, target_img_path)

print(f"✅ {num_samples} test images copied to '{target_dir}' for prediction demo.")
