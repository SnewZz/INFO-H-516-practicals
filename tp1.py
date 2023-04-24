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


    #for row in np.arange(img_h - bl_h + 1, step=bl_h):
    #    for col in np.arange(img_w - bl_w + 1, step=bl_w):
    #        tmp = dct_transform(img_array[row:row+bl_h, col:col+bl_w])
    #        img_array[row:row+bl_h, col:col+bl_w] = tmp
    #
    #img_dct = Image.fromarray(img_array)
    #img_dct.show()
#
    #for row in np.arange(img_h - bl_h + 1, step=bl_h):
    #    for col in np.arange(img_w - bl_w + 1, step=bl_w):
    #        tmp = idct_transform(img_array[row:row+bl_h, col:col+bl_w])
    #        img_array[row:row+bl_h, col:col+bl_w] = tmp
    #
    #img2 = Image.fromarray(img_array)
    #img2.show()

def decode()

def dct_transform(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct_transform(block):
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
