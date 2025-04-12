import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for hand tracking
import time  # Time module for calculating FPS

# Initialize webcam capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Variables to calculate Frames Per Second (FPS)
prev_time = 0
curr_time = 0

while True:
    success, frame = video_capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = hand_detector.process(frame_rgb)

    if detection_results.multi_hand_landmarks:
        for hand_landmarks in detection_results.multi_hand_landmarks:
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                frame_height, frame_width, _ = frame.shape
                pixel_x = int(landmark.x * frame_width)
                pixel_y = int(landmark.y * frame_height)

                # Simple pinch detection between thumb tip (4) and index tip (8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                if pinch_distance < 0.05:
                    cv2.putText(frame, "Pinch Detected", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                # Index finger Y controls vertical bar height
                if landmark_id == 8:
                    normalized_y = landmark.y
                    bar_height = int((1 - normalized_y) * 300)

                    bar_x, bar_y = 50, 400
                    cv2.rectangle(frame, (bar_x, bar_y - bar_height), (bar_x + 50, bar_y), (0, 255, 0), cv2.FILLED)

                    cv2.putText(frame, f"{int((1 - normalized_y) * 100)}%", (bar_x, bar_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)
    cv2.waitKey(1)
