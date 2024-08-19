import numpy as np
import os
import cv2

filename = 'ball.avi'
frames_per_second = 24.0
res = '720p'

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # Change the current capture device to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Start video capture
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, res))

while True:
    ret, frame = cap.read()
    
    # Check if frame is successfully captured
    if not ret:
        break

    # Add text overlay
    text = "Recording..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2
    position = (50, 50)  # Position of the text

    # Apply the text overlay on the frame
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Write the frame with the overlay to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
