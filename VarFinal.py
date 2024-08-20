import cv2
import numpy as np
import os
import time
from datetime import datetime

# Define color ranges in HSV format
greenLower = (35, 100, 100)
greenUpper = (85, 255, 255)
lower_red1 = np.array([0, 150, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 50])
upper_red2 = np.array([180, 255, 255])

filename = 'ball.mp4'  # Change the file extension to .mp4
frames_per_second = 24.0
res = '720p'  # Set resolution to 720p

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Grab resolution dimensions and set video capture to it.
def get_dims(cap, res='720p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # Change the current capture device to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'mp4v'),  # Use mp4v for MP4 format
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']  # Default to avi if not found

# Function to save screenshot with overlays to a specific directory
def save_screenshot_with_overlays(frame, xmin2, xxx, save_dir="screenshots"):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/goal_screenshot_{timestamp}.png"

    # Draw the red line
    cv2.line(frame, (xmin2, 0), (xmin2, frame.shape[0]), (0, 0, 255), 3)

    # Draw the "Goal" text if a goal was detected
    if xxx == "Goal":
        text_position = (50, 75)  # Position text on the left side
        cv2.putText(frame, "Goal", text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

    # Save the frame with overlays
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as {filename}")


# Start video capture
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, res))

# Create a window and a trackbar for brightness adjustment
cv2.namedWindow("Ball Detection and Goal Line")

xmin = 1000
xmin2 = 1000
flag = 1
final = None
xxx = "No Goal"
recording = True

while cap.isOpened():
    flag2 = 0
    ret, frame = cap.read()

    if not ret:
        break


    # # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # if flag == 1:
    #     # Create masks for red color to detect the goal line
    #     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    #     mask = cv2.bitwise_or(mask1, mask2)

    #     kernel = np.ones((4, 4), np.uint8)
    #     mask = cv2.erode(mask, kernel, iterations=1)
    #     mask = cv2.dilate(mask, kernel, iterations=1)

    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #     if len(contours) > 0:
    #         cv2.drawContours(frame, [contours[0]], -1, (0, 255, 0), 5)
    #         for i in range(len(contours[0])):
    #             if contours[0][i][0][0] < xmin2:
    #                 xmin2 = contours[0][i][0][0]
    #         cv2.line(frame, (xmin2, 0), (xmin2, frame.shape[0]), (0, 0, 255), 5)
    #     flag = 0

    largest_radius = 0
    largest_contour = None

    # Detect the green ball
    lower_bound = np.array(greenLower)
    upper_bound = np.array(greenUpper)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Perform morphological operations to remove small noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Proceed if any contours are found
    if contours:
        # Find the largest contour for the green ball
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle around the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        
        # Check if this is the largest detected ball so far
        if radius > largest_radius:
            largest_radius = radius
            largest_contour = (int(x), int(y), int(radius))

    # Check if the detected ball crosses the goal line
    if largest_contour is not None and largest_radius > 10:
        x, y, radius = largest_contour
        cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
        cv2.putText(frame, "Green Ball", (x - radius, y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if x < xmin and x > 50:
            xmin = x
            flag2 = 1

        if radius < 70:
            if x + radius < xmin2:
                xxx = "Goal"
                save_screenshot_with_overlays(frame, xmin2, xxx)  # Save screenshot with overlays
            else:
                xxx = "No Goal"

    # Set position of the text to the right side of the frame
    text_position = (600, 70)

    if xxx == "No Goal":
        cv2.putText(frame, "No Goal", text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    else:
        cv2.putText(frame, "Goal", text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

    cv2.line(frame, (xmin2, 0), (xmin2, frame.shape[0]), (0, 0, 255), 3)

    if flag2 == 1:
        final = frame.copy()

    # Add text overlay for recording status and current time
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2
    position = (50, 50)  # Position of the text

    # Add current time to the right side of the frame
    current_time = datetime.now().strftime("%H:%M:%S")
    text_position_time = (frame.shape[1] - 450, frame.shape[0] - 20)  # Adjust position as needed
    cv2.putText(frame, f"Time: {current_time}", text_position_time, font, 0.6, color, thickness)

    # Write the frame with the overlay to the video file
    if recording:
        out.write(frame)

    # Display the frame
    cv2.imshow("Ball Detection and Goal Line", frame)

    # Handle user input
    key = cv2.waitKey(1) & 0xFF

    # Stop recording if 's' is pressed
    if key == ord('s'):
        recording = False
        print("Recording stopped")

    # Break the loop if the 'q' key is pressed
    if key == ord('q'):
        break

# Release the capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
