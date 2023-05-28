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
        blocks = []
        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        img_float32 = img.astype(np.float32)
        for i in range(0, img_float32.shape[0], BLOCK_SIZE):
            for j in range(0, img_float32.shape[1], BLOCK_SIZE):
                if(self.channel_count == 1):
                    blocks.append(MyBlock(img_float32[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]))
                else:
                    blocks.append(MyBlock(img_float32[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :]))
        return blocks
        
    def encode(self):
        print("Compression of the image...")
        for block in self.blocks:
            block.dct()
            block.quantize()

    def decode(self):
        print("Decompression of the image...")
        for block in self.blocks:
            block.dequantize()
            block.idct()


    def merge_blocks(self, file_path):
        num_blocks_h = self.height // BLOCK_SIZE
        num_blocks_w = self.width // BLOCK_SIZE
        # Créer une image vide pour stocker l'image reconstruite
        if self.channel_count == 1 :
            reconstructed = np.zeros((self.height, self.width), dtype=np.uint8)
        else :
            reconstructed = np.zeros((self.height, self.width, self.channel_count), dtype=np.uint8)

        # Parcourir tous les blocs
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = self.blocks[i * num_blocks_w + j]

                # Calculer les coordonnées du bloc dans l'image reconstruite
                y, x = i * BLOCK_SIZE, j * BLOCK_SIZE

                # Copier les données du bloc dans l'image reconstruite
                if self.channel_count == 1:
                    reconstructed[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = block.get_data()
                else:
                    reconstructed[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE, :] = block.get_data()

        img = I.fromarray(reconstructed)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img.save(os.path.join(current_dir,file_path))
        return reconstructed

