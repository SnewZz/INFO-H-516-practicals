import cv2
import numpy as np
import utils

BLOCK_SIZE = 8

# Open the video file
video = cv2.VideoCapture("data/ex2/bus_cif.y4m")

# Get the video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

original_frames = []

psnr_result = []

# List to stock the encoded frames (index 0 contains frames for LOW_QUANTIZATION_MATRIX, 1 for QUANTIZATION_MATRIx, 2 for HIGH_QUANTIZATION_MATRIX and 3 for BAD_QUANTIZATION_MATRIX)
encoded_frames = [[],[],[],[]]


showVideo = True
# Read video frames one by one and encode it.
for i in range(frame_count):
    ret, frame = video.read()

    if not ret:
        print("Error")
        break
    original_frames.append(frame)

    # Conversion to np.float32 to avoid loss of precision
    frame_float32 = frame.astype(np.float32)

    # Loop for the test on different QUANTIZATION_MATRIX
    for k in range(4):

        # Apply of the DCT and the quantization by block.
        encoded_frame = utils.encode(frame_float32, BLOCK_SIZE, k)

        # Store encoded frame in list
        encoded_frames[k].append(encoded_frame)

    # View original video
    if showVideo :
        cv2.imshow("Original Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        showVideo=False
        cv2.destroyWindow("Original Frame")

if showVideo : 
    # Close original video window
    cv2.destroyWindow("Original Frame")

for i in range(4):
    # Create an object to write the decoded video
    output_video = cv2.VideoWriter("result/ex2/decoded_bus_cif_mode"+str(i)+".avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

    psnr = 0
    # Browse encoded frames for decompression and display
    for j, encoded_frame in enumerate(encoded_frames[i]):
        # Apply dequantization and IDCT to get the decoded frame.
        decoded_frame = utils.decode(encoded_frame, BLOCK_SIZE, i)

        # Final conversion to uint8 for registration
        decoded_frame_uint8 = decoded_frame.astype(np.uint8)

        psnr+=cv2.PSNR(original_frames[j], decoded_frame_uint8)

        # Write decoded frame to output video
        output_video.write(decoded_frame_uint8)

        # Display decoded frame
        cv2.imshow("Reconstructed Frame mode"+str(i), decoded_frame_uint8)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    psnr_result.append(psnr/frame_count)

    output_video.release()

utils.plot_psnr_result(psnr_result, "bus_cif")
# Freeing up resources
video.release()
cv2.destroyAllWindows()
