from scipy.fftpack import dct, idct
import numpy as np
import cv2

QUANTIZATION_MATRIX = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


class MyBlock:
    def __init__(self, data):
        if len(data.shape) == 2:
            self.channel_count = 1
        else:
            self.channel_count = data.shape[2]
        self.data = data

    def dct(self):
        # self.data = dct(dct(self.data.T, norm='ortho').T, norm='ortho')
        if self.channel_count == 1:
            self.data = cv2.dct(self.data.astype(np.float32))
        else:
            for k in range(self.data.shape[2]):
                self.data[:, :, k] = cv2.dct(self.data[:, :, k].astype(np.float32))

    def quantize(self):
        if self.channel_count == 1:
            self.data = np.round(self.data / QUANTIZATION_MATRIX) * QUANTIZATION_MATRIX
        else:
            for k in range(self.data.shape[2]):
                self.data[:, :, k] = (
                    np.round(self.data[:, :, k] / QUANTIZATION_MATRIX)
                    * QUANTIZATION_MATRIX
                )

    def idct(self):
        # self.data = idct(idct(self.data.T, norm='ortho').T, norm='ortho')
        if self.channel_count == 1:
            self.data = cv2.idct(self.data.astype(np.float32))
        else:
            for k in range(self.data.shape[2]):
                self.data[:, :, k] = cv2.idct(self.data[:, :, k].astype(np.float32))

    def dequantize(self):
        if self.channel_count == 1:
            self.data = self.data * QUANTIZATION_MATRIX
        else:
            for k in range(self.data.shape[2]):
                self.data[:, :, k] = self.data[:, :, k] * QUANTIZATION_MATRIX

    def get_data(self):
        return self.data
