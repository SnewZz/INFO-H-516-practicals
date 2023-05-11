from scipy.fftpack import dct, idct
import numpy as np

class Block:
    def __init__(self, data):
        self.data = data

    def dct(self):
        self.data = dct(dct(self.data.T, norm='ortho').T, norm='ortho')
    
    def quantize(self, quantization_matrix):
        self.data = np.round(np.divide(self.data, quantization_matrix)).astype(int)

    def idct(self):
        self.data = idct(idct(self.data.T, norm='ortho').T, norm='ortho')

    def dequantize(self, quantization_matrix):
        self.data = np.multiply(self.data, quantization_matrix)

    def get_data(self):
        return self.data