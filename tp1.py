from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct


def encode(path):   
    img = Image.open(path)
    img_np = np.array(img)

    block_size = 8

    if len(img_np.shape) == 2:
        height, width = img_np.shape
        channels = 1  #grayscale image
    else:
        height, width, channels = img_np.shape #color image

    h_blocks, w_blocks = height // block_size, width // block_size

    img_np = img_np[:h_blocks*block_size, :w_blocks*block_size]

    blocks = np.split(img_np, h_blocks, axis=0)
    blocks = [np.split(block, w_blocks, axis=1) for block in blocks]
    blocks = np.array(blocks).reshape(-1, block_size, block_size, channels)[:h_blocks*w_blocks]

    dct_blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = blocks[i*w_blocks+j]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_blocks.append(dct_block)

    inv_blocks = []
    for i in range(h_blocks*w_blocks):
        inv_block = idct(idct(dct_blocks[i].T, norm='ortho').T, norm='ortho')
        inv_blocks.append(inv_block)

    # Recombiner les blocs en une seule image
    inv_img_np = np.zeros_like(img_np)
    for i in range(h_blocks):
        for j in range(w_blocks):
            inv_img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = inv_blocks[i*w_blocks+j].squeeze()

    # Convertir le tableau numpy en image Pillow
    inv_img = Image.fromarray(inv_img_np)

    # Enregistrer l'image
    inv_img.save('img/inverse_bird.tga')
