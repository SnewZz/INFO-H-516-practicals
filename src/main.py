import image
import tp1
if __name__ == '__main__':
    img = image.Image(".\\data\\bird.tif")
    img.encode()
    img.save_image(".\\data\\result\\encode_bird.tif")
    img.decode()
    img.save_image(".\\data\\result\\new_bird.tif")
    # tp1.encode("data/bird.tga", 95)