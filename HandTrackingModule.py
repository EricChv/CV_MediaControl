import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for hand tracking
import time  # Time module for calculating FPS


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5): 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        # Initialize MediaPipe Hand Tracking
        self.mpHands = mp.solutions.hands  # Load the hand tracking module
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf
            )  # Create an instance of Hands (defaults to using RGB images)
        self.mpDraw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame to RGB (MediaPipe requires RGB format)
        self.results = self.hands.process(imgRGB)  # Process the RGB frame to detect hands

        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # Loop through detected hands=
                if draw:
                    # Draw hand landmarks and connections on the frame
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img # Return img if drawn
    
    def findPosition(self, img, handNum=0, draw=True):

        lmList = [] # Landmark list
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(myHand.landmark):  # Loop through hand landmarks
                # Extract the pixel coordinates of the landmark
                h, w, c = img.shape  # Get image dimensions (height, width, channels)
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                #print(id, cx, cy)  # Print landmark ID and its pixel coordinates
                lmList.append([id, cx, cy])
                # If landmark id == 4 (thumb tip), draw a circle at its position
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

        

        return lmList # Return landmark list

def main():
    # Variables to calculate Frames Per Second (FPS)
    pTime = 0  # Previous time
    cTime = 0  # Current time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video frame width to 640 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video frame height to 480 pixels
    detector = handDetector() # object
    while True:
        success, img = cap.read()  # Capture a frame from the webcam
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4]) # only show coordinate for lm4

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # FPS formula
        pTime = cTime  # Update previous time

        # Display FPS on the frame
        cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame with landmarks and FPS
        cv2.imshow("Image", img)
        # Waits for 1 millisecond before proceeding to the next frame.
        cv2.waitKey(1)


# If running script in __main__ execute this
if __name__ == "__main__":
    main()