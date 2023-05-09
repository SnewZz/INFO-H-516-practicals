from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct


def encode(path, quality):   

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

    # Define quantization matrix Q
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # Apply DCT to each block and store the result in a list
    dct_blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = blocks[i*w_blocks+j]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quantized_block = np.round(dct_block / (Q * quality))
            dct_blocks.append(quantized_block)

    # Apply inverse DCT to each block and store the result in a list
    inv_blocks = []
    for i in range(h_blocks*w_blocks):
        quantized_block = dct_blocks[i]
        inv_quantized_block = quantized_block * (Q * quality)
        inv_block = idct(idct(inv_quantized_block.T, norm='ortho').T, norm='ortho')
        inv_blocks.append(inv_block)

    inv_img_np = np.zeros_like(img_np)
    for i in range(h_blocks):
        for j in range(w_blocks):
            inv_img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = dct_blocks[i*w_blocks+j].squeeze()[:,:,0]


    # Convertir le tableau numpy en image Pillow
    inv_img = Image.fromarray(inv_img_np)

    # Enregistrer l'image
    inv_img.save('data/result/dct_bird.tga')

    # Recombiner les blocs en une seule image
    inv_img_np = np.zeros_like(img_np)
    for i in range(h_blocks):
        for j in range(w_blocks):
            inv_img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = inv_blocks[i*w_blocks+j].squeeze()[:,:,0]

    # Convertir le tableau numpy en image Pillow
    inv_img = Image.fromarray(inv_img_np)

    # Enregistrer l'image
    inv_img.save('data/result/inverse_bird.tga')
