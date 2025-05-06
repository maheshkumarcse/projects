import cv2
import mysql.connector
from datetime import datetime
from deepface import DeepFace
import time

# === MySQL connection ===
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="face_analysis"
    )
    cursor = conn.cursor()
    print("[INFO] Connected to MySQL.")
except mysql.connector.Error as err:
    print("[ERROR] MySQL connection failed:", err)
    exit(1)

# === Webcam setup ===
cap = cv2.VideoCapture(0)

# === Timer ===
last_process_time = time.time()

print("[INFO] Starting 5-second emotion detection loop...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Webcam read failed.")
        break

    # Get current time
    current_time = time.time()

    # Check if 5 seconds have passed
    if current_time - last_process_time >= 5:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]['dominant_emotion']
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Print to terminal
            print(f"[{timestamp}] Detected Emotion: {current_emotion}")

            # Store in database
            cursor.execute(
                "INSERT INTO emotion_results (timestamp, emotion) VALUES (%s, %s)",
                (timestamp, current_emotion)
            )
            conn.commit()
            print(f"[INFO] Stored emotion: {current_emotion} at {timestamp}")

            # Reset timer
            last_process_time = current_time

        except Exception as e:
            print("[ERROR] Emotion detection or DB insert failed:", e)

    # Show live video (optional)
    cv2.imshow("Live Emotion Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit signal received.")
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
print("[INFO] Program ended.")
