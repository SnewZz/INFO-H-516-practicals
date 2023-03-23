from PIL import Image
import numpy as np


def encode(path):   
    img = Image.open(path)

    img_array = np.array(img)
    block_img = np.zeros(img_array.shape)
    img_h, img_w = img_array.shape[:2]
    bl_h, bl_w = 8, 8
    for row in np.arange(img_h - bl_h + 1, step=bl_h):
        for col in np.arange(img_w - bl_w + 1, step=bl_w):
            dct_transform(img_array[row:row+bl_h, col:col+bl_w])

    print(img_array)
    #Image.fromarray(imgarray)


def dct_transform(block):
    print(block)