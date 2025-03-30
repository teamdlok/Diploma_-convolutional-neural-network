import cv2
import numpy as np
from prettytable import PrettyTable

# img = cv2.imread('2018_30_1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('2018_30_1.png')
print(img)
n_white_pix = np.sum(img == 255)
print('Number of white pixels:', n_white_pix)

x = PrettyTable()

x.field_names = ["white px", "file_name"]

x.add_row([n_white_pix, img])
x.add_row([n_white_pix, img])
x.add_row([n_white_pix, img])
print(x)