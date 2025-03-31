import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for hand tracking
import time  # Time module for calculating FPS

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video frame width to 640 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video frame height to 480 pixels

# Initialize MediaPipe Hand Tracking
mpHands = mp.solutions.hands  # Load the hand tracking module
hands = mpHands.Hands()  # Create an instance of Hands (defaults to using RGB images)
mpDraw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

# Variables to calculate Frames Per Second (FPS)
pTime = 0  # Previous time
cTime = 0  # Current time

while True:
    success, img = cap.read()  # Capture a frame from the webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame to RGB (MediaPipe requires RGB format)
    results = hands.process(imgRGB)  # Process the RGB frame to detect hands

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Loop through detected hands
            for id, lm in enumerate(handLms.landmark):  # Loop through hand landmarks
                # Extract the pixel coordinates of the landmark
                h, w, c = img.shape  # Get image dimensions (height, width, channels)
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                print(id, cx, cy)  # Print landmark ID and its pixel coordinates
                
                # If landmark ID is 4 (thumb tip), draw a circle at its position
                if id == 4: 
                    cv2.circle(img, (cx, cy), 10, (255, 124, 23), cv2.FILLED)

            # Draw hand landmarks and connections on the frame
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)  # FPS formula
    pTime = cTime  # Update previous time

    # Display FPS on the frame
    cv2.putText(img, str(int(fps)), (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with landmarks and FPS
    cv2.imshow("Image", img)

    # Press 'q' to exit the loop and close the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
