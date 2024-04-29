import cv2
from yolo import YOLO  # Assuming YOLO class is defined in a module called 'yolo'
from PIL import Image
import numpy as np

def start_distance_detector(video_path):
    # Create an instance of the YOLO detector
    yolo_detector = YOLO()

    # Open the video stream
    video_stream = cv2.VideoCapture(video_path)
    
    frame_counter = 0
    frame_rate = 10

    try:
        while video_stream.isOpened():
            try:
                # Read a frame from the video stream
                ret, image = video_stream.read()

                # Check if the frame is successfully read and process every 'frame_rate'-th frame
                if ret and frame_counter == frame_rate:
                    # Detect objects in the image using YOLO
                    result_image = yolo_detector.detect_image(Image.fromarray(image))

                    # Display the input video with YOLO detection
                    cv2.imshow("Input Video", np.asarray(result_image))

                    # Check for user input to exit the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_counter = 0
            except Exception as e:
                print("Error processing frames: " + str(e))
                pass
            else:
                pass
                # print("Unable to stream the video")
            
            frame_counter += 1

    except Exception as e:
        video_stream.release()
        print("Error: " + str(e))
        return

    # Release the video stream
    video_stream.release()
