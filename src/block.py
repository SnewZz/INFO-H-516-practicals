from scipy.fftpack import dct, idct
import numpy as np
import cv2

QUANTIZATION_MATRIX = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])

class Block:
    def __init__(self, data):
        self.data = data

    def dct(self):
        self.data = dct(dct(self.data.T, norm='ortho').T, norm='ortho')
    
    def quantize(self):
        self.data = np.divide(self.data, QUANTIZATION_MATRIX).astype(int)

    def idct(self):
        self.data = idct(idct(self.data.T, norm='ortho').T, norm='ortho')

    def dequantize(self):
        self.data = np.multiply(self.data, QUANTIZATION_MATRIX)

    def get_data(self):
        return self.data
    
    def convert_to(self, dtype, scale=1.0):
        self.data = cv2.convertScaleAbs(self.data, alpha=scale, beta=0.0)

        if dtype == "float":
            self.data = self.data.astype(np.float32)
        elif dtype == "double":
            self.data = self.data.astype(np.float64)
        elif dtype == "int":
            self.data = self.data.astype(np.int32)
        elif dtype == "uint8":
            self.data = self.data.astype(np.uint8)
        elif dtype == "uint16":
            self.data = self.data.astype(np.uint16)
        else:
            raise ValueError(f"Invalid data type: {dtype}")