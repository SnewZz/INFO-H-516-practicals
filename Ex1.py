import cv2
import numpy as np
import utils

BLOCK_SIZE = 8

psnr_result = []
new_images = []

# Load of the TIFF image
image = cv2.imread("data/ex1/bird.tif", cv2.IMREAD_UNCHANGED)


# Conversion to np.float32 to avoid loss of precision
image_float32 = image.astype(np.float32)

# Loop on different quantization matrix
for i in range(3):
    # Apply of the DCT and the quantization by block.
    encoded_image = utils.encode(image_float32, BLOCK_SIZE, i)

    # Save of the coded image
    cv2.imwrite("result/ex1/encoded_bird_mode"+str(i)+".tif", encoded_image.astype(np.float32))


    #######################Separate Encodeur and Decodeur###############################
    # Load of the compressed image for decoding
    # encoded_image_loaded = cv2.imread("result/ex1/encoded_lena3_mode"+str(i)+".tif", cv2.IMREAD_UNCHANGED)

    # Convert into float32 for IDCT
    # encoded_image= encoded_image_loaded.astype(np.float32)
    ####################################################################################


    # Apply of the dequantization and the IDCT by block.
    reconstructed_image = utils.decode(encoded_image, BLOCK_SIZE, i)

    # Final conversion to uint8 for display and save
    reconstructed_image_uint8 = reconstructed_image.astype(np.uint8)

    # Save of the decoded image
    cv2.imwrite("result/ex1/decoded_bird_mode"+str(i)+".tif", reconstructed_image_uint8)
    new_images.append(reconstructed_image_uint8)

    # PSNR calculation between original and decompressed image
    psnr_result.append(cv2.PSNR(image, reconstructed_image_uint8))


# Display of the original picture and the different decoded pictures.
cv2.imshow("Original Image", image)

for i, img in enumerate(new_images):
    cv2.imshow("Reconstructed Image "+str(i), img)
print(psnr_result)
utils.plot_psnr_result(psnr_result, "Bird (in grey scale)")
cv2.waitKey(0)
cv2.destroyAllWindows()
