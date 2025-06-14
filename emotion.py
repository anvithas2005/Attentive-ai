import cv2
from deepface import DeepFace

# Initialize face detector
face_cascade = cv2.CascadeClassifier('haarcascade.xml')

# Capture video from the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, img = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region
        face_region = img[y:y + h, x:x + w]

        # Detect emotion on the face
        try:
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']  # Use the first result
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

    # Display the frame
    cv2.imshow('Emotion Detection', img)

    # Break the loop with 'Esc'
    if cv2.waitKey(10) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
