import cv2
import numpy as np
from PIL import Image
import os 
from PIL import ImageOps

image_size = 384

# Путь к папке, куда сохранять изображения
# dataset_path = "dataset_neyronka_30_meters"
cwd = os.getcwd()

print(cwd)


def image_processor(path):
    for root, dirs, files in os.walk(path):

        for file in files:
            # print(file)
            name, ext = os.path.splitext(file)
            find_10 = name.split("_") 

            if "10" in find_10[1]:
                print(name)

                if ext == ".png":

                    img_path = os.path.join(root, name + ext)
                    img = Image.open(img_path)
                    #Делаем глубину цвета - 1
                    img_resized = img.convert("RGB")

                    # img_path = os.path.join(root, name + ".png")

                    #Изменяем размер изображения
                    # img_resized = img_resized.resize((image_size,image_size))

                    # изменяем размер изображения до кратного 64
                    border = (1, 1, 1, 1) # left, top, right, bottom
                    # img_resized = ImageOps.crop(img_resized, border)
                    folder_name = os.path.basename(root)
                    print(img_path)
                    img_resized.save(img_path)


for root, dirs, files in os.walk(cwd):

    for dir in dirs:
        if dir.startswith("PH2 Dataset images"):
            path = os.path.join(root,dir)

            image_processor(path)
            exit()
        else :
            print(os.getcwd())

