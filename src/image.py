import cv2
import numpy as np
from block import Block
from PIL import Image as I

BLOCK_SIZE = 8

class Image:
    def __init__(self, path):
        self.path = path
        self.width, self.height, self.channel_bits, self.channel_count = self.get_image_info()
        self.blocks = self.get_blocks()

    def get_image_info(self):
        img = I.open(self.path)
        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 2:
            height, width = img.shape
            channel_count = 1  #grayscale image
        else:
            height, width, channel_count = img.shape
        channel_bits = img.dtype.itemsize * 8

        return height, width, channel_bits, channel_count

    def get_blocks(self):

        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

        # Reshape the data of the image into blocks
        num_blocks_h = self.height // BLOCK_SIZE
        num_blocks_w = self.width // BLOCK_SIZE
        if len(img.shape) == 2:
            block_data = np.reshape(img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE], (num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE, self.channel_count))
        else :
            block_data = np.reshape(img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE, :], (num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE, self.channel_count))

         #convert into float to avoid lost of precision in the calculation with DCT
        block_data = block_data.astype(np.float32)

        blocks = []
        for i in range(num_blocks_h):
            row_blocks = []
            for j in range(num_blocks_w):
                block = Block(block_data[i, j])
                row_blocks.append(block)
            blocks.append(row_blocks)

        return blocks
        
    def encode(self):

        for row_blocks in self.blocks:
            for block in row_blocks:
                block.dct()

    def decode(self):

        for row_blocks in self.blocks:
            for block in row_blocks:
                block.idct()

    def save_image(self, file_path):
        # Combine the blocks back into an array
        block_data = np.zeros((self.height, self.width, self.channel_count), dtype=np.float32)
        for i, row_blocks in enumerate(self.blocks):
            for j, block in enumerate(row_blocks):
                x_start, y_start = j * BLOCK_SIZE, i * BLOCK_SIZE
                block_data[y_start:y_start+BLOCK_SIZE, x_start:x_start+BLOCK_SIZE, :] = block.data

        # Convert the array to an unsigned 8-bit integer array
        image_data = np.clip(block_data, 0, 255).astype(np.uint8)

        # Save the image using the PIL Image module
        cv2.imwrite(file_path, image_data)