from scipy.spatial import distance
from imutils import face_utils
import pygame
import time
import dlib
import cv2
import os
from sendGmail import sendAlertEmail as sendAlertEmail
from threading import Thread
import Constant as Constants

def detectDrowsiness():

    # Initialize Pygame and load sound
    volume = 0.10
    people_info = {}
    dataset_path = 'datasets'
    
    # Inform user about face recognition initialization
    print('Recognizing Faces, Ensure Sufficient Lighting...')

    # Create lists for images, labels, names, and ids
    (face_images, labels, names, identifier) = ([], [], {}, 0)

    pygame.mixer.init()
    pygame.mixer.music.load('audio/alert.wav')

    # Set the minimum threshold of eye aspect ratio to trigger an alarm
    EYE_ASPECT_RATIO_THRESHOLD = Constants.threashold

    # Set the minimum consecutive frames below threshold for the alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

    # Count the number of consecutive frames below the threshold value
    frame_counter = 0
    warning_counter = 0

    # Load face cascade for drawing rectangles around detected faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Define a function to calculate and return eye aspect ratio
    def calculate_eye_aspect_ratio(eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])

        ear = (A + B) / (2 * C)
        return ear

    # Load face detector and predictor, using a dlib shape predictor file
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("dataset\shape_predictor_68_face_landmarks.dat")

    # Extract indexes of facial landmarks for the left and right eye
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    # Start webcam video capture
    video_capture = cv2.VideoCapture(Constants.source)

    while True:
        # Read each frame, flip it, and convert to grayscale
        ret, current_frame = video_capture.read()
        cv2.imshow('Video Feed', current_frame)

        current_frame = cv2.flip(current_frame, 1)
        grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Detect facial points using the detector function
        detected_faces = face_detector(grayscale_frame, 0)

        # Detect faces using haarcascade_frontalface_default.xml
        face_rectangles = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)

        for face in detected_faces:
            shape = shape_predictor(grayscale_frame, face)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(current_frame, (x, y), 2, (0, 255, 0), -1)

            # Draw eye contours around eyes
            left_eye_landmarks = shape[left_eye_start:left_eye_end]
            right_eye_landmarks = shape[right_eye_start:right_eye_end]

            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye_landmarks)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye_landmarks)

            eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

            left_eye_hull = cv2.convexHull(left_eye_landmarks)
            right_eye_hull = cv2.convexHull(right_eye_landmarks)
            cv2.drawContours(current_frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(current_frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # Drowsiness Detection
            for (x, y, w, h) in face_rectangles:
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 128, 255), 2)

            if eye_aspect_ratio < EYE_ASPECT_RATIO_THRESHOLD:
                frame_counter += 1
                warning_counter += 1

                pygame.mixer.music.set_volume(volume)
                pygame.mixer.music.play(-1)
                time.sleep(0.1)
                volume += 0.01

                cv2.putText(current_frame, "Drowsiness Detected", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(current_frame, "Eyes Closed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                if warning_counter > Constants.limit:
                    print('\nLimit Exceeded! Sending alert message to family members...')
                    pygame.mixer.music.stop()

                    try:
                        Thread(target=sendAlertEmail).start()
                    except Exception as e:
                        print('Error: Message Not Sent')

                    warning_counter = 0

            else:
                cv2.putText(current_frame, "Eyes Open", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                pygame.mixer.music.stop()
                frame_counter = 0
                warning_counter = 0
                volume = 0.10

        # Show video feed
        cv2.imshow('Video Feed', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            pygame.quit()
            break
