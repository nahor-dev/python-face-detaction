import cv2

def run_face_detector():
    # 1. Load the pre-trained Haar Cascade face detection model
    # OpenCV provides built-in access to this data file path
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 2. Initialize video capture from the default webcam (0)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Face detection active. Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 3. Convert the frame to grayscale for faster processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Detect faces within the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 5. Draw bounding boxes around each detected face
        for (x, y, w, h) in faces:
            # Parameters: frame, top-left (x,y), bottom-right (x+w, y+h), color (BGR format), thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 6. Display the resulting frame in a window
        cv2.imshow('Real-Time Face Detection', frame)

        # Stop the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Release the camera hardware and clean up UI windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detector()
