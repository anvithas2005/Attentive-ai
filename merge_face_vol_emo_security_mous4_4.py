import cv2
import winsound
import mediapipe as mp
import pyautogui
from deepface import DeepFace

# Initialize face detector, hand detector, and PyAutoGUI screen size
face_cascade = cv2.CascadeClassifier('haarcascade.xml')
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Capture video from the webcam
camera = cv2.VideoCapture(0)

# Variables for hand tracking and mouse/volume control
x1 = y1 = x2 = y2 = 0
mouse_x = mouse_y = 0

while camera.isOpened():
    # Read frames from the webcam for motion detection and hand tracking
    ret1, frame1 = camera.read()
    ret2, frame2 = camera.read()

    if not ret1 or not ret2:
        print("Failed to grab frame")
        break

    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

    # Flip the image for better hand gesture detection
    frame1 = cv2.flip(frame1, 1)
    frame_height, frame_width, _ = frame1.shape

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_region = frame1[y:y + h, x:x + w]

        # Detect emotion on the face
        try:
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(frame1, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if dominant_emotion == "happy":
                print("Student is happy!")
            elif dominant_emotion == "sad":
                print("Student is sad.")
            elif dominant_emotion == "angry":
                print("Student is angry.")
        except Exception as e:
            print(f"Emotion detection error: {e}")

    # Hand detection
    rgb_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame1, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index Finger (Mouse Control)
                    mouse_x = int(screen_width / frame_width * x)
                    mouse_y = int(screen_height / frame_height * y)
                    cv2.circle(frame1, (x, y), 8, (0, 255, 255), 3)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1 = x
                    y1 = y

                if id == 4:  # Thumb (For Gesture Control)
                    cv2.circle(frame1, (x, y), 8, (0, 0, 255), 3)
                    x2 = x
                    y2 = y

            # Calculate distance between thumb and index finger for clicks or volume control
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Hand gesture-based click (distance < 40 triggers a click)
            if dist < 40:
                pyautogui.click()

            # Adjust volume based on distance between thumb and forefinger
            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    # Display the combined video feed with face, emotion, hand, and motion detection
    cv2.imshow("Merged Detection", frame1)

    # Break the loop if 'Esc' is pressed
    if cv2.waitKey(10) == 27:
        break

# Release the webcam and close windows
camera.release()
cv2.destroyAllWindows()
