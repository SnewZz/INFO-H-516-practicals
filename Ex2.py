import cv2
import numpy as np
import utils

BLOCK_SIZE = 8

# Ouvrir la vidéo en utilisant cv2.VideoCapture
video = cv2.VideoCapture("data/bus_cif.y4m")

# Obtenir les propriétés de la vidéo
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Créer un objet pour écrire la vidéo décodée
output_video = cv2.VideoWriter("decoded_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# Lire les frames de la vidéo une par une
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Conversion en np.float64 pour éviter la perte de précision
    frame_float32 = frame.astype(np.float32)

    # Afficher la vidéo originale
    cv2.imshow("Original Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Application de la DCT pour l'encodage
    encoded_frame = utils.encode(frame_float32, BLOCK_SIZE)

    # Afficher la vidéo encodée
    encoded_frame_uint8 = encoded_frame.astype(np.uint8)
    cv2.imshow("Encoded Frame", encoded_frame_uint8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Application de l'IDCT pour la reconstruction
    reconstructed_frame = utils.decode(encoded_frame, BLOCK_SIZE)

    # Conversion finale en uint8 pour l'enregistrement
    reconstructed_frame_uint8 = reconstructed_frame.astype(np.uint8)

    # Écrire le frame décodé dans la vidéo de sortie
    output_video.write(reconstructed_frame_uint8)

    # Afficher le frame décodé
    cv2.imshow("Reconstructed Frame", reconstructed_frame_uint8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
output_video.release()
cv2.destroyAllWindows()
