import cv2
import numpy as np

BLOCK_SIZE = 8

# Fonction pour appliquer la DCT par bloc de 8x8
def apply_dct(image):
    result = np.zeros_like(image, dtype=np.float32)

    # Parcours des blocs de 8x8
    for i in range(0, image.shape[0], BLOCK_SIZE):
        for j in range(0, image.shape[1], BLOCK_SIZE):
            # Extraction du bloc de 8x8
            block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

            # Application de la DCT sur chaque canal de couleur
            dct_block = np.zeros_like(block, dtype=np.float32)
            for k in range(block.shape[2]):
                dct_block[:, :, k] = cv2.dct(block[:, :, k].astype(np.float32))

            # Stockage du résultat
            result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = dct_block

    return result

# Fonction pour appliquer l'IDCT par bloc de 8x8
def apply_idct(result):
    image = np.zeros_like(result, dtype=np.uint8)

    # Parcours des blocs de 8x8
    for i in range(0, result.shape[0], BLOCK_SIZE):
        for j in range(0, result.shape[1], BLOCK_SIZE):
            # Extraction du bloc de 8x8
            dct_block = result[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

            # Application de l'IDCT sur chaque canal de couleur
            block = np.zeros_like(dct_block, dtype=np.float32)
            for k in range(dct_block.shape[2]):
                block[:, :, k] = cv2.idct(dct_block[:, :, k].astype(np.float32))

            # Stockage du résultat
            image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = block.astype(np.uint8)

    return image

# Chargement de l'image TIFF en couleur
image = cv2.imread("src/data/lena3.tif", cv2.IMREAD_COLOR)

# Application de la DCT
dct_result = apply_dct(image)

# Sauvegarde de l'image compressée
cv2.imwrite("encoded_image.tif", dct_result)

# Chargement de l'image compressée
encoded_image = cv2.imread("encoded_image.tif", cv2.IMREAD_UNCHANGED)

# Application de l'IDCT
reconstructed_image = apply_idct(encoded_image)

# Sauvegarde de l'image décompressée
cv2.imwrite("decoded_image.tif", reconstructed_image)

# Affichage des images
cv2.imshow("Original Image", image)
cv2.imshow("Reconstructed Image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
