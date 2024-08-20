import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import time

# Define the lower and upper boundaries of the "green" ball in the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# Initialize video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # Grab the current frame
    frame = vs.read()

    # Resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Construct a mask for the color "green"
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        # Find the largest contour and compute the minimum enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Determine if the ball is on the left or right side
            frame_center_x = frame.shape[1] // 2
            if center[0] > frame_center_x:
                position = "Tidak Goal"
            else:
                position = "Goal"

            # Draw the position status on the frame
            cv2.putText(frame, position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Draw the center line
    goal_line_own = frame.shape[1] // 4  # Garis 1/4 dari lebar frame
    goal_line_opponent = 3 * frame.shape[1] // 4  # Garis 3/4 dari lebar frame

    cv2.line(frame, (goal_line_own, 0), (goal_line_own, 480), (255, 0, 0), 2)
    cv2.line(frame, (goal_line_opponent, 0), (goal_line_opponent, 480), (0, 255, 0), 2)    

    # Show the frame to our screen
    cv2.imshow("Frame", frame)
    
    # If the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Stop the video stream and close all windows
vs.stop()
cv2.destroyAllWindows()

#catatanku
#penentuan warna
#Versioning >main >level >future1 >future2