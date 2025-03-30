import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width to 640 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4830) # Set the height to 480 pixels

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # This class only uses RGB images (Default values set in hands.py)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert img into RGB
    results = hands.process(imgRGB)
    

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Extract from each hand (since it's a 'for', it will extract each one)
            for id, lm in enumerate(handLms.landmark):
                print(id,lm)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # Display img (not imgRGB) since we show img

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS
    cv2.putText(img, str(int(fps)), (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
