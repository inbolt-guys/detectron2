import os
import glob
import shutil
from PIL import Image

# Function to binarize the alpha channel
def binarize_alpha(image):
    # Split the image into RGB and alpha channels
    r, g, b, a = image.split()
    
    # Convert alpha channel: 255 means fully opaque, 0 means fully transparent
    a = a.point(lambda p: 255 if p > 0 else 0)
    
    # Merge back the modified alpha channel with RGB channels
    return Image.merge('RGBA', (r, g, b, a))

# Function to resize the image
def resize_image(image, scale=0.5):
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

datasets = ["4_instances_rocket_steel", 
            "4_instances_rocket_steel_with_random_objects", 
            "rocket_steel_1_instance_pose1",
            "rocket_wheel_bin_picking_1",
            "rocket_wheel_4_instances_4_poses"]

for dataset_name in datasets:
    dataset_path = f"/home/clara/detectronDocker/dataset_for_detectron/rocket_steel_all_datasets/{dataset_name}/rgbd"
    out_path = f"/home/clara/gifs/{dataset_name}"

    os.makedirs(out_path, exist_ok=True)
    images_paths = sorted(glob.glob(dataset_path+"/*.png"))

    """for i in range(0, len(images_paths), 5):
        ip = images_paths[i]
        name = os.path.basename(ip)
        shutil.copy(ip, out_path+name)"""

    # Load images, binarize the alpha channel, and resize
    processed_images = []
    for i, image_file in enumerate(images_paths):
        if i % 30 == 0:  # Take every 5th image
            img = Image.open(image_file).convert('RGBA')
            img = binarize_alpha(img)
            img = resize_image(img, scale=0.5)
            processed_images.append(img)
            img.save(f"{out_path}/fig-{i // 60}.png")

    """# Save as GIF
    processed_images[0].save(
        f"{out_path}/{dataset_name}.gif",  # Output file name
        save_all=True,
        append_images=processed_images[1:],  # Append the other images to the first
        duration=100,  # Duration between frames in milliseconds
        loop=0,  # Infinite loop
        transparency=0,  # Set transparency index to the first index (0)
        disposal=2  # Background will be transparent
    )"""
