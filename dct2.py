import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from PIL import Image
import graphlearning as gl

def dct2(f):
    """2D Discrete Cosine Transform

    Args:
        f: Square array

    Returns: 
        2D DCT of f
    """
    return dct(dct(f, axis=0, norm='ortho' ),axis=1, norm='ortho')




def idct2(f):
    """2D Inverse Discrete Cosine Transform

    Args:
        f: Square array

    Returns: 
        2D Inverse DCT of f
    """
    return idct(idct(f, axis=0 , norm='ortho'), axis=1 , norm='ortho')

I = plt.imread("img/bird.tif")

print('Data type: '+str(I.dtype))
print('Pixel intensity range: (%d,%d)'%(I.min(),I.max()))
print(I.shape)

plt.figure(figsize=(10,10))
plt.imshow(I, cmap = "gray")
plt.show()

patch_size = 8  #8x8 patch
I_dct = np.zeros_like(I)
for i in range(0,I.shape[0],patch_size):
    for j in range(0,I.shape[1],patch_size):
        I_dct[i:(i+patch_size),j:(j+patch_size)] = dct2(I[i:(i+patch_size),j:(j+patch_size)])

plt.figure(figsize=(10,10))
plt.imshow(I_dct,cmap='gray',vmin=0,vmax=np.max(I_dct)*0.01)

plt.show()

thresh = 0.1
I_thresh = I_dct * (np.absolute(I_dct) > thresh*np.max(np.absolute(I_dct)))

patch_size = 8  #8x8 patch
I_comp = np.zeros_like(I)
for i in range(0,I.shape[0],patch_size):
    for j in range(0,I.shape[1],patch_size):
        I_comp[i:(i+patch_size),j:(j+patch_size)] = idct2(I_dct[i:(i+patch_size),j:(j+patch_size)])

plt.figure(figsize=(10,10))
plt.imshow(I_comp,cmap='gray')
#plt.imshow(np.hstack((I,I_comp, I-I_comp+0.5)), cmap='gray', vmin=0, vmax=1)

#I_idct = np.zeros_like(I_dct)
#for i in range(0,I_dct.shape[0],patch_size):
#    for j in range(0,I_dct.shape[1],patch_size):
#        I_idct[i:(i+patch_size),j:(j+patch_size)] = idct2(I_dct[i:(i+patch_size),j:(j+patch_size)])
#
#plt.figure(figsize=(10,10))
#plt.imshow(I_idct,cmap='gray',vmin=0,vmax=np.max(I_idct)*0.01)
#
plt.show()