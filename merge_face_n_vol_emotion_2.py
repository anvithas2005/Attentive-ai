import cv2
import mediapipe as mp
import pyautogui
from deepface import DeepFace

# Initialize face detector and hand detector
face_cascade = cv2.CascadeClassifier('haarcascade.xml')
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Capture video from the webcam
webcam = cv2.VideoCapture(0)

x1 = y1 = x2 = y2 = 0

while True:
    # Read the frame from the webcam
    ret, img = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip the image for better hand gesture detection
    frame_height, frame_width, _ = img.shape

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Draw rectangles around detected faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Extract the face region
        face_region = img[y:y + h, x:x + w]

        # Detect emotion on the face
        try:
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(img, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Example response based on detected emotion
            if dominant_emotion == "happy":
                print("User is happy!")
            elif dominant_emotion == "sad":
                print("User is sad.")
            elif dominant_emotion == "angry":
                print("User is angry.")
            # Add more emotions as needed
            
        except Exception as e:
            print(f"Emotion detection error: {e}")

    # Convert frame to RGB for hand detection
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(img, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Forefinger
                    cv2.circle(img, (x, y), 8, (0, 255, 255), 3)
                    x1 = x
                    y1 = y
                if id == 4:  # Thumb
                    cv2.circle(img, (x, y), 8, (0, 0, 255), 3)
                    x2 = x
                    y2 = y

            # Calculate the distance between thumb and forefinger
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4

            # Draw a line between thumb and forefinger
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # Adjust volume based on distance
            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    # Display the result
    cv2.imshow("Face and Hand Detection", img)

    # Break the loop if 'Esc' is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()