import cv2
import numpy as np

BLOCK_SIZE = 8

QUANTIZATION_MATRIX = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])

HIGH_QUANTIZATION_MATRIX = np.array([[32, 24, 22, 32, 48, 80, 102, 122],
                                     [24, 24, 28, 38, 52, 116, 120, 110],
                                     [28, 26, 32, 48, 80, 114, 138, 112],
                                     [28, 34, 44, 58, 102, 174, 160, 124],
                                     [36, 44, 74, 112, 136, 218, 206, 154],
                                     [48, 70, 110, 128, 162, 208, 226, 184],
                                     [98, 128, 156, 174, 206, 242, 240, 202],
                                     [144, 184, 190, 196, 224, 200, 206, 198]])

LOW_QUANTIZATION_MATRIX = np.array([[8, 5, 5, 8, 12, 20, 26, 31],
                                    [6, 6, 7, 10, 13, 29, 30, 27],
                                    [7, 6, 8, 12, 20, 29, 35, 28],
                                    [7, 9, 11, 15, 26, 44, 40, 31],
                                    [9, 11, 19, 28, 34, 55, 52, 39],
                                    [12, 18, 28, 33, 41, 52, 57, 46],
                                    [25, 32, 39, 43, 51, 60, 59, 50],
                                    [36, 46, 48, 50, 57, 50, 52, 50]])



# Fonction pour appliquer la DCT par bloc de 8x8
def encode(image):
    result = np.zeros_like(image, dtype=np.float32)

    # Parcours des blocs de 8x8
    for i in range(0, image.shape[0], BLOCK_SIZE):
        for j in range(0, image.shape[1], BLOCK_SIZE):
            # Extraction du bloc de 8x8
            block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :]

            # Application de la DCT sur chaque canal de couleur
            dct_block = np.zeros_like(block, dtype=np.float32)
            for k in range(block.shape[2]):
                dct_block[:, :, k] = cv2.dct(block[:, :, k].astype(np.float32))
                # Application de la quantization
                # dct_block[:, :, k] = np.divide(dct_block[:, :, k], QUANTIZATION_MATRIX)

                dct_block[:, :, k] = np.round(dct_block[:, :, k] / HIGH_QUANTIZATION_MATRIX) * HIGH_QUANTIZATION_MATRIX


            # Stockage du résultat
            result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :] = dct_block
    print(result.dtype)
    return result

# Fonction pour appliquer l'IDCT par bloc de 8x8
def decode(result):
    image = np.zeros_like(result, dtype=np.float32)

    # Parcours des blocs de 8x8
    for i in range(0, result.shape[0], BLOCK_SIZE):
        for j in range(0, result.shape[1], BLOCK_SIZE):
            # Extraction du bloc de 8x8
            dct_block = result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :]

            # Application de l'IDCT sur chaque canal de couleur
            block = np.zeros_like(dct_block, dtype=np.float32)
            
            for k in range(dct_block.shape[2]):
                block[:, :, k] = dct_block[:, :, k] * HIGH_QUANTIZATION_MATRIX
                block[:, :, k] = cv2.idct(dct_block[:, :, k].astype(np.float32))

            # Conversion des valeurs de block en uint8
            block = np.clip(block, 0, 255)

            # Stockage du résultat
            image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :] = block

    return image


# Chargement de l'image TIFF en couleur
image = cv2.imread("src/data/lena3.tif", cv2.IMREAD_UNCHANGED)
print(type(image))
# Conversion en np.float64 pour éviter la perte de précision
image_float32 = image.astype(np.float32)

# Application de la DCT pour l'encodage
encoded_image = encode(image_float32)

# Sauvegarde de l'image compressée
# cv2.imwrite("encoded_image.tif", encoded_image.astype(np.float32))

# Chargement de l'image compressée pour le décodage
# encoded_image_loaded = cv2.imread("encoded_image.tif", cv2.IMREAD_UNCHANGED) # ptet mauvais flag

# Conversion en np.float64 pour l'IDCT
# encoded_image_float32 = encoded_image_loaded.astype(np.float32)

# Application de l'IDCT pour la reconstruction
reconstructed_image = decode(encoded_image)

# Conversion finale en uint8 pour l'affichage et l'enregistrement
reconstructed_image_uint8 = reconstructed_image.astype(np.uint8)

# Sauvegarde de l'image décompressée
cv2.imwrite("decoded_image.tif", reconstructed_image_uint8)

# Calcul du PSNR entre l'image originale et l'image décompressée
psnr = cv2.PSNR(image, reconstructed_image_uint8)
print("PSNR:", psnr)

# Affichage des images
cv2.imshow("Original Image", image)
cv2.imshow("Reconstructed Image", reconstructed_image_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
