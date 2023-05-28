import image
if __name__ == '__main__':
    original_img = image.MyImage("data\\bird.tif")
    original_img.encode()
    original_img.merge_blocks(".\\data\\result\\encode_bird.tif")
    original_img.decode()
    original_img.merge_blocks(".\\data\\result\\new_bird.tif")
    # encoded_img = image.MyImage(".\\data\\result\\encode_bird.tif")
    # encoded_img.decode()
    # encoded_img.merge_blocks(".\\data\\result\\new_bird.tif")