import cv2
import numpy as np

# Inisialisasi variabel
goal_line_x_left = 100  # X-coordinate batas goal kiri
goal_line_x_right = 540  # X-coordinate batas goal kanan
ball_radius = 20  # Radius bola (sesuaikan dengan ukuran bola)
frame_width = 640  # Lebar frame video (sesuaikan dengan resolusi video)

# Buka video atau kamera
cap = cv2.VideoCapture(0)  # 0 untuk webcam, ganti dengan path video jika menggunakan file

# Inisialisasi skor
score = {'left': 0, 'right': 0}
goal_detected = {'left': False, 'right': False}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi gambar ke ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold untuk mendeteksi warna bola (misalnya merah)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Temukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 2 * ball_radius and h > 2 * ball_radius:  # Filter ukuran kontur
            ball_center_x = x + w // 2
            ball_center_y = y + h // 2

            # Gambar kontur dan posisi bola
            cv2.circle(frame, (ball_center_x, ball_center_y), ball_radius, (0, 255, 0), 2)

            # Cek jika bola melewati garis goal
            if ball_center_x < goal_line_x_left and not goal_detected['left']:
                score['left'] += 1
                print(f"Goal Left! Score: {score['left']}")
                goal_detected['left'] = True
                goal_detected['right'] = False  # Reset detection for the other goal
            elif ball_center_x > goal_line_x_right and not goal_detected['right']:
                score['right'] += 1
                print(f"Goal Right! Score: {score['right']}")
                goal_detected['right'] = True
                goal_detected['left'] = False  # Reset detection for the other goal
            elif goal_line_x_left <= ball_center_x <= goal_line_x_right:
                # Reset goal detection when ball is within goal area (if needed)
                goal_detected['left'] = False
                goal_detected['right'] = False

    # Gambar garis goal
    cv2.line(frame, (goal_line_x_left, 0), (goal_line_x_left, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (goal_line_x_right, 0), (goal_line_x_right, frame.shape[0]), (0, 0, 255), 2)

    # Tampilkan hasil
    cv2.putText(frame, f"Score Left: {score['left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Score Right: {score['right']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Ball Detection', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()