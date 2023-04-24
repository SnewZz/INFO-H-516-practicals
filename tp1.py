from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct


def encode(path):   
    img = Image.open(path)
    img.show()
    img_array = np.array(img)
    block_img = np.zeros(img_array.shape)
    img_h, img_w = img_array.shape[:2]
    bl_h, bl_w = 8, 8
    img_array = dct_transform(img_array)

def dct_transform(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct_transform(block):
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
