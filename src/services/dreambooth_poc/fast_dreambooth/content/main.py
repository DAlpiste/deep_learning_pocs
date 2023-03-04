from PIL import Image
from tqdm import tqdm
from smart_crop import *

IMAGES_FOLDER_OPTIONAL = "./data/original/"
INSTANCE_DIR = "./data/reshaped/"
Crop_size = 512

def convert_images():
    for filename in tqdm(os.listdir(IMAGES_FOLDER_OPTIONAL), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
        extension = filename.split(".")[-1]
        new_path_with_file = os.path.join(INSTANCE_DIR, filename)
        file = Image.open(IMAGES_FOLDER_OPTIONAL+"/"+filename)
        width, height = file.size
        if file.size !=(Crop_size, Crop_size):
            image=crop_image(file, Crop_size)
            if extension.upper()=="JPG" or extension.lower()=="jpg":
                image[0] = image[0].convert("RGB")
                image[0].save(new_path_with_file, format="JPEG", quality = 100)
            else:
                image[0].save(new_path_with_file, format=extension.upper())

convert_images()