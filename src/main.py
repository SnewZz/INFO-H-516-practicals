import image
if __name__ == '__main__':
    original_img = image.MyImage("data\\bird.tif")
    original_img.encode()
    original_img.save_image(".\\data\\result\\encode_bird.tif")
    encoded_img = image.MyImage(".\\data\\result\\encode_bird.tif")
    encoded_img.decode()
    encoded_img.save_image(".\\data\\result\\new_bird.tif")
    # tp1.encode("data/bird.tga", 95)