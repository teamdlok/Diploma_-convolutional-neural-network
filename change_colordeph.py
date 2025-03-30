from PIL import Image

img_path = "D:/ДИПЛОМ/neyronka_dataset/neyronka_two/2018_10_1.png"

img = Image.open(img_path)
#Делаем глубину цвета - 1
img_resized = img.convert("RGB")
img_resized.save(img_path)