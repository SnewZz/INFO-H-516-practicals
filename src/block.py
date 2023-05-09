import cv2

class Block:
    def __init__(self, data):
        self.data = data

    def dct(self):
        return cv2.dct(self.data)
    
    def idct(self):
        return cv2.idct(self.data)