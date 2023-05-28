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


# Fonction pour appliquer la DCT par bloc de 8x8
def apply_dct(image):
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

                dct_block[:, :, k] = np.round(dct_block[:, :, k] / QUANTIZATION_MATRIX) * QUANTIZATION_MATRIX


            # Stockage du résultat
            result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :] = dct_block
    print(result.dtype)
    return result

# Fonction pour appliquer l'IDCT par bloc de 8x8
def apply_idct(result):
    image = np.zeros_like(result, dtype=np.float32)

    # Parcours des blocs de 8x8
    for i in range(0, result.shape[0], BLOCK_SIZE):
        for j in range(0, result.shape[1], BLOCK_SIZE):
            # Extraction du bloc de 8x8
            dct_block = result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, :]

            # Application de l'IDCT sur chaque canal de couleur
            block = np.zeros_like(dct_block, dtype=np.float32)
            
            for k in range(dct_block.shape[2]):
                block[:, :, k] = dct_block[:, :, k] * QUANTIZATION_MATRIX
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
encoded_image = apply_dct(image_float32)

# Sauvegarde de l'image compressée
# cv2.imwrite("encoded_image.tif", encoded_image.astype(np.float32))

# Chargement de l'image compressée pour le décodage
# encoded_image_loaded = cv2.imread("encoded_image.tif", cv2.IMREAD_UNCHANGED) # ptet mauvais flag

# Conversion en np.float64 pour l'IDCT
# encoded_image_float32 = encoded_image_loaded.astype(np.float32)

# Application de l'IDCT pour la reconstruction
reconstructed_image = apply_idct(encoded_image)

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
