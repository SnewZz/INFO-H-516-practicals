import numpy as np
import cv2

dct_img = cv2.imread('img/dct_bird.tif', cv2.IMREAD_GRAYSCALE)

h, w = np.shape(dct_img)
blocks = [np.split(row, w // 8, axis=1) for row in np.split(dct_img, h // 8)]

idct_blocks = []
for row in blocks:
    idct_row = []
    for block in row:
        idct = cv2.idct(block)
        idct_row.append(idct)
    idct_blocks.append(idct_row)

idct_img = np.concatenate([np.concatenate(row, axis=1) for row in idct_blocks], axis=0)
cv2.imwrite('img/idct_bird.tif', idct_img)