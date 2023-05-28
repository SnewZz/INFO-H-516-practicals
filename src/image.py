import cv2
import os
import numpy as np
from block import MyBlock
from PIL import Image as I

BLOCK_SIZE = 8

class MyImage:

    #This is the constructor of an Image object
    def __init__(self, path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(current_dir, path)
        self.width, self.height, self.channel_count = self.get_image_info()
        self.blocks = self.get_blocks()

    def get_image_info(self):

        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            height, width = img.shape
            channel_count = 1  #grayscale image
        else:
            height, width, channel_count = img.shape

        return height, width, channel_count

    def get_blocks(self):

        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

        # Reshape the data of the image into blocks
        num_blocks_h = self.height // BLOCK_SIZE
        num_blocks_w = self.width // BLOCK_SIZE

        img_np = img[:num_blocks_h*BLOCK_SIZE, :num_blocks_w*BLOCK_SIZE]

        blocks = np.split(img_np, num_blocks_h, axis=0)
        blocks = [np.split(block, num_blocks_w, axis=1) for block in blocks]
        if(self.channel_count == 1):
            print(blocks.shape)
            blocks = np.array(blocks).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[:num_blocks_h*num_blocks_w]
        else:
            blocks = np.array(blocks).reshape(-1, BLOCK_SIZE, BLOCK_SIZE, self.channel_count)[:num_blocks_h*num_blocks_w]
        block_array = []
        for i in range(num_blocks_h):
            row = []
            for j in range(num_blocks_w):
                # Create a Block object for this block
                block_data = blocks[i*num_blocks_w+j]
                block = MyBlock(block_data)
                row.append(block)
            block_array.append(row)

        return block_array
        
    def encode(self):
        print("Compression of the image...")
        print(self.blocks[0][0].get_data().shape)
        for row_blocks in self.blocks:
            for block in row_blocks:
                block.convert_to("double")
                block.dct()
                # block.quantize()
        print(self.blocks[0][0].get_data().shape)

    def decode(self):
        print("Decompression of the image...")

        for row_blocks in self.blocks:
            for block in row_blocks:
                block.convert_to("double")
                # block.dequantize()
                block.idct()

    def save_image(self, file_path):

        num_blocks_h = self.height // BLOCK_SIZE
        num_blocks_w = self.width // BLOCK_SIZE

        reconstructed = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = self.blocks[i][j]
                y, x = i*BLOCK_SIZE, j*BLOCK_SIZE
                reconstructed[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = block.get_data()
                # if(self.channel_count == 1):
                    # reconstructed[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = block.get_data()
                # else:
                    # reconstructed[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = block.get_data()

        # Save the image to file
        img = I.fromarray(reconstructed)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img.save(os.path.join(current_dir,file_path))

