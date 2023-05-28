import cv2
import numpy as np
import utils

BLOCK_SIZE = 8

# Open the video file
video = cv2.VideoCapture("data/bus_cif.y4m")

# Get the video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# List to stock the encoded frames
encoded_frames = []

# Read video frames one by one and encode it.
while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    # Conversion to np.float32 to avoid loss of precision
    frame_float32 = frame.astype(np.float32)

    # Apply of the DCT and the quantization by block.
    encoded_frame = utils.encode(frame_float32, BLOCK_SIZE, 3)

    # Store encoded frame in list
    encoded_frames.append(encoded_frame)

    # View original video
    cv2.imshow("Original Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close original video window
cv2.destroyWindow("Original Frame")

# Create an object to write the decoded video
output_video = cv2.VideoWriter("decoded_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# Browse encoded frames for decompression and display
for encoded_frame in encoded_frames:
    # Apply dequantization and IDCT to get the decoded frame.
    decoded_frame = utils.decode(encoded_frame, BLOCK_SIZE, 3)

    # Final conversion to uint8 for registration
    decoded_frame_uint8 = decoded_frame.astype(np.uint8)

    # Write decoded frame to output video
    output_video.write(decoded_frame_uint8)

    # Display decoded frame
    cv2.imshow("Reconstructed Frame", decoded_frame_uint8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freeing up resources
video.release()
output_video.release()
cv2.destroyAllWindows()
