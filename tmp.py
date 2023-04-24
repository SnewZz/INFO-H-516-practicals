import numpy as np
import cv2

tiff_img = cv2.imread('img/bird.tif', cv2.IMREAD_UNCHANGED)
max_val = np.iinfo(tiff_img.dtype).max
tiff_img = (tiff_img / max_val * 255).astype(np.uint8)

h, w = np.shape(tiff_img)
blocks = [np.split(row, w // 8, axis=1) for row in np.split(tiff_img, h // 8)]
dct_blocks = []
for row in blocks:
    dct_row = []
    for block in row:
        dct = cv2.dct(np.float32(block))
        dct_row.append(dct)
    dct_blocks.append(dct_row)

dct_img = np.concatenate([np.concatenate(row, axis=1) for row in dct_blocks], axis=0)
cv2.imwrite('img/dct_bird.tif', dct_img)

idct_blocks = []
for row in blocks:
    idct_row = []
    for block in row:
        idct = cv2.idct(block.astype(np.float32))
        idct_row.append(idct)
    idct_blocks.append(idct_row)

idct_img = np.concatenate([np.concatenate(row, axis=1) for row in idct_blocks], axis=0)
cv2.imwrite('img/idct_bird.tif', idct_img)