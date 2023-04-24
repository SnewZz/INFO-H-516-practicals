from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct


def encode(path):   

    #Load the image and convert to numpy array
    img = Image.open(path)
    img_np = np.array(img)

    block_size = 8

    # Determine the height, width and channels of the image
    if len(img_np.shape) == 2:
        height, width = img_np.shape
        channels = 1  #grayscale image
    else:
        height, width, channels = img_np.shape #color image

    # Calculate the number of blocks along the height and width of the image
    h_blocks, w_blocks = height // block_size, width // block_size

    # Crop the image to ensure each block is of the same size
    img_np = img_np[:h_blocks*block_size, :w_blocks*block_size]

    # Split the image into blocks of size block_size x block_size
    blocks = np.split(img_np, h_blocks, axis=0)
    blocks = [np.split(block, w_blocks, axis=1) for block in blocks]

    # Convert the blocks to a numpy array of shape (n_blocks, block_size, block_size, channels)
    blocks = np.array(blocks).reshape(-1, block_size, block_size, channels)[:h_blocks*w_blocks]

    # Apply DCT to each block and store the result in a list
    dct_blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = blocks[i*w_blocks+j]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_blocks.append(dct_block)

    # Apply inverse DCT to each block and store the result in a list
    inv_blocks = []
    for i in range(h_blocks*w_blocks):
        inv_block = idct(idct(dct_blocks[i].T, norm='ortho').T, norm='ortho')
        inv_blocks.append(inv_block)

    inv_img_np = np.zeros_like(img_np)
    for i in range(h_blocks):
        for j in range(w_blocks):
            inv_img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = dct_blocks[i*w_blocks+j].squeeze()

    # Convertir le tableau numpy en image Pillow
    inv_img = Image.fromarray(inv_img_np)

    # Enregistrer l'image
    inv_img.save('img/dct_bird.tga')

    # Recombiner les blocs en une seule image
    inv_img_np = np.zeros_like(img_np)
    for i in range(h_blocks):
        for j in range(w_blocks):
            inv_img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = inv_blocks[i*w_blocks+j].squeeze()

    # Convertir le tableau numpy en image Pillow
    inv_img = Image.fromarray(inv_img_np)

    # Enregistrer l'image
    inv_img.save('img/inverse_bird.tga')
