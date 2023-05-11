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

        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        print(img.dtype)

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

        img_np = img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE]

        blocks = np.split(img_np, num_blocks_h, axis=0)
        blocks = [np.split(block, num_blocks_w, axis=1) for block in blocks]
        blocks = np.array(blocks).reshape(-1, BLOCK_SIZE, BLOCK_SIZE, self.channel_count)[:num_blocks_h*num_blocks_w]

        block_array = []
        for i in range(num_blocks_h):
            row = []
            for j in range(num_blocks_w):
                # Create a Block object for this block
                block_data = blocks[i*num_blocks_w+j]
                block = Block(block_data)
                row.append(block)
            block_array.append(row)

        # blocks_list = []
        # for block in blocks:
        #     blocks_list.append(Block(block))
        #print(len(blocks_list))
        #===========================================
            
        # if len(img.shape) == 2:
        #     block_data = np.reshape(img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE], (num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE, self.channel_count))
        # else :
        #     block_data = np.reshape(img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE, :], (num_blocks_h, num_blocks_w, BLOCK_SIZE, BLOCK_SIZE, self.channel_count))

        #  #convert into float to avoid lost of precision in the calculation with DCT
        # block_data = block_data.astype(np.float32)

        # blocks = []
        #for i in range(num_blocks_h):
        #    row_blocks = []
        #    for j in range(num_blocks_w):
        #        block = Block(block_data[i, j])
        #        row_blocks.append(block)
        #    blocks.append(row_blocks)

        return block_array
        
    def encode(self):

        quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])

        for row_blocks in self.blocks:
            for block in row_blocks:
                block.dct()
                block.quantize(quantization_matrix)

    def decode(self):

        quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])

        for row_blocks in self.blocks:
            for block in row_blocks:
                block.idct()
                block.dequantize(quantization_matrix)

    def save_image(self, file_path):

        num_blocks_h = self.height // BLOCK_SIZE
        num_blocks_w = self.width // BLOCK_SIZE

        reconstructed = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = self.blocks[i][j]
                y, x = i*BLOCK_SIZE, j*BLOCK_SIZE
                reconstructed[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = block.get_data().squeeze().astype(np.uint8)

        # Save the image to file
        img = I.fromarray(reconstructed)
        img.save(file_path)

