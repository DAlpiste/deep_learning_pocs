import cv2
import cvlib as cv
from cvlib.object_detection import detect_common_objects, draw_bbox

def reshape_photo_with_mainity_with_path_image(path):
    img = cv2.imread(path)
    bbox, label, conf = detect_common_objects(img, enable_gpu=True) # detect_common_objects(img, model="yolov3-tiny")
    max_area = 0
    max_box = None
    for i, (x, y, w, h) in enumerate(bbox):
        area = w * h
        if area > max_area:
            max_area = area
            max_box = bbox[i]
    if max_box is not None:
        try:
            print('Este es el maxbox: {0}'.format(max_box))
            x, y, w, h = max_box
            cropped_img = img[y:y+h, x:x+w]

            # Redimensionar la imagen a 512 x 512 píxeles
            resized_img = cv2.resize(cropped_img, (512, 512))

            # Guardar la imagen redimensionada
            cv2.imwrite('reshaped_{0}'.format(path), resized_img)
        except Exception as e:
            print(str(e))
    else:
        print("No se encontraron objetos en la imagen.")

import os

current_folder = os.getcwd()
file_list = os.listdir(current_folder)
image_list = [f for f in file_list if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
print(image_list)
for path in image_list:
    print(path)
    reshape_photo_with_mainity_with_path_image(path)