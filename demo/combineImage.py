import os
from PIL import Image

# Define paths
folder_A = '/public/home/zhuyuchen530/projects/cvpr24/fake3/demo/ss1'
folder_B = '/public/home/zhuyuchen530/projects/cvpr24/fake3/demo/ss2'
folder_GT = '/public/home/zhuyuchen530/projects/ECLIPSE/ade_ps_base_gt_val'
save_root = './step2_drop_cates'

if not os.path.exists(save_root):
    os.makedirs(save_root)

# Process images
save_dir = save_root
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Construct paths for images in folders A, B, and GT
for img_name in os.listdir(folder_A):
    img_A_path = os.path.join(folder_A, img_name)
    img_B_path = os.path.join(folder_B, img_name)
    img_GT_path = os.path.join(folder_GT, img_name.split('.')[0] + '.jpg')

    # Check if corresponding images exist in B and GT
    if os.path.exists(img_B_path) and os.path.exists(img_GT_path):
        try:
            # Open images
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            img_GT = Image.open(img_GT_path)

            # Resize images to the same size (optional, if needed)
            # Assuming all images are the same size, otherwise uncomment the following lines
            # img_B = img_B.resize(img_A.size)
            # img_GT = img_GT.resize(img_A.size)

            # Create a new image with width of sum of the three images and the same height
            new_width = img_A.width + img_B.width + img_GT.width
            new_height = img_A.height
            combined_img = Image.new('RGB', (new_width, new_height))

            # Paste the three images into the combined image
            combined_img.paste(img_A, (0, 0))
            combined_img.paste(img_B, (img_A.width, 0))
            combined_img.paste(img_GT, (img_A.width + img_B.width, 0))

            # Save the combined image

            save_path = os.path.join(save_dir, img_name)
            combined_img.save(save_path)

            print(f"Saved combined image: {save_path}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    else:
        print(f"Missing corresponding image for {img_name} in either B or GT folder")

