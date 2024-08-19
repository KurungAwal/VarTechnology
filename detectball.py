import cv2
import numpy as np

# Define color ranges in HSV format
color_ranges = {
    "blue": ([100, 150, 0], [140, 255, 255]),
    "red": ([0, 150, 50], [10, 255, 255]),  # Red can also be detected with another range [170, 150, 50], [180, 255, 255]
    "orange": ([10, 150, 150], [25, 255, 255]),
    "yellow": ([25, 150, 150], [35, 255, 255]),
    "green": ([35, 100, 100], [85, 255, 255])
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    largest_radius = 0
    largest_contour = None
    detected_color = None

    # Loop through each color range and detect the ball
    for color, (lower, upper) in color_ranges.items():
        # Convert bounds to numpy arrays
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)

        # Create a mask for the current color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Perform morphological operations to remove small noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed if any contours are found
        if contours:
            # Find the largest contour for the current color
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum enclosing circle around the largest contour
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            
            # Check if this is the largest detected ball so far
            if radius > largest_radius:
                largest_radius = radius
                largest_contour = (int(x), int(y), int(radius))
                detected_color = color

    # Draw the largest detected ball if any
    if largest_contour is not None and largest_radius > 10:
        x, y, radius = largest_contour
        cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
        cv2.putText(frame, f"{detected_color.capitalize()} Ball", (x - radius, y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Ball Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
