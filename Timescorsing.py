import cv2
import numpy as np
import time

# Inisialisasi variabel
score = {'left': 0, 'right': 0}
goal_left_x = 100  # X-coordinate batas goal kiri
goal_right_x = 540  # X-coordinate batas goal kanan
ball_radius = 20  # Radius bola (sesuaikan dengan ukuran bola)
frame_width = 640  # Lebar frame video (sesuaikan dengan resolusi video)

# Buka video atau kamera
cap = cv2.VideoCapture(0)  # 0 untuk webcam, ganti dengan path video jika menggunakan file

start_time = time.time()

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

    ball_detected = False

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 2 * ball_radius and h > 2 * ball_radius:  # Filter ukuran kontur
            ball_center_x = x + w // 2
            ball_center_y = y + h // 2

            # Gambar kontur dan posisi bola
            cv2.circle(frame, (ball_center_x, ball_center_y), ball_radius, (0, 255, 0), 2)

            # Cek jika bola masuk ke goal
            if ball_center_x < goal_left_x:
                if not ball_detected:
                    score['left'] += 1
                    print(f"Goal Left! Score: {score['left']}")
                    ball_detected = True
            elif ball_center_x > goal_right_x:
                if not ball_detected:
                    score['right'] += 1
                    print(f"Goal Right! Score: {score['right']}")
                    ball_detected = True
            else:
                ball_detected = False

    # Hitung waktu yang telah berlalu
    elapsed_time = time.time() - start_time
    cv2.putText(frame, f"Time: {elapsed_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Score Left: {score['left']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Score Right: {score['right']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan hasil
    cv2.imshow('Ball Detection', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
