import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Model:
    @staticmethod
    def loadmodel(model_path):
        return load_model(model_path)

    def __init__(self, model_path):
        self.loaded_model = self.loadmodel(model_path)
        self.graph = tf.compat.v1.get_default_graph()

    def predict(self, image):
        with self.graph.as_default():
            processed_image = preprocess_image_for_keras(image)
            steering_angle_raw = float(self.loaded_model.predict(processed_image, batch_size=1))
            # Adjusting the steering angle for better control (if needed)
            steering_angle = steering_angle_raw * 60
            return steering_angle

def laneDetector(image, display_result):
    image_shape = image.shape
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 9
    smoothed_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)
    min_val = 60
    max_val = 150
    edges_image = cv2.Canny(smoothed_image, min_val, max_val)

    # Create mask to keep only the region of interest
    vertices = np.array([[(0, image_shape[0]), (465, 320), (475, 320), (image_shape[1], image_shape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges_image)
    color = 255
    cv2.fillPoly(mask, vertices, color)

    # Apply mask to image
    masked_image = cv2.bitwise_and(edges_image, mask)

    # Hough lines detection
    rho = 2
    theta = np.pi / 180
    threshold = 45
    min_line_len = 40
    max_line_gap = 100
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    if lines is not None and len(lines) > 2:
        all_lines = np.zeros_like(masked_image)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(all_lines, (x1, y1), (x2, y2), (255, 255, 0), 2)

        slope_positive_lines = []
        slope_negative_lines = []
        y_values = []

        for current_line in lines:
            for x1, y1, x2, y2 in current_line:
                line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if line_length > 30:
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        if slope > 0:
                            tan_theta = np.tan((abs(y2 - y1)) / (abs(x2 - x1)))
                            angle = np.arctan(tan_theta) * 180 / np.pi
                            if 20 < abs(angle) < 85:
                                slope_negative_lines.append([x1, y1, x2, y2, -slope])
                                y_values.append(y1)
                                y_values.append(y2)
                        if slope < 0:
                            tan_theta = np.tan((abs(y2 - y1)) / (abs(x2 - x1)))
                            angle = np.arctan(tan_theta) * 180 / np.pi
                            if 20 < abs(angle) < 85:
                                slope_positive_lines.append([x1, y1, x2, y2, -slope])
                                y_values.append(y1)
                                y_values.append(y2)

        if not slope_positive_lines or not slope_negative_lines:
            print('Not enough lines found')

        # Average position of lines and extrapolate to the top and bottom of the lane.
        pos_slopes = np.asarray(slope_positive_lines)[:, 4]
        pos_slope_median = np.median(pos_slopes)
        pos_slope_std_dev = np.std(pos_slopes)
        pos_slopes_good = []

        for slope in pos_slopes:
            if abs(slope - pos_slope_median) < pos_slope_median * 0.2:
                pos_slopes_good.append(slope)

        pos_slope_mean = np.mean(np.asarray(pos_slopes_good))

        neg_slopes = np.asarray(slope_negative_lines)[:, 4]
        neg_slope_median = np.median(neg_slopes)
        neg_slope_std_dev = np.std(neg_slopes)
        neg_slopes_good = []

        for slope in neg_slopes:
            if abs(slope - neg_slope_median) < 0.9:
                neg_slopes_good.append(slope)

        neg_slope_mean = np.mean(np.asarray(neg_slopes_good))

        # Positive Lines
        x_intercept_pos = []
        for line in slope_positive_lines:
            x1 = line[0]
            y1 = image_shape[0] - line[1]
            slope = line[4]
            y_intercept = y1 - slope * x1
            x_intercept = -y_intercept / slope
            if x_intercept == x_intercept:
                x_intercept_pos.append(x_intercept)

        x_int_pos_median = np.median(x_intercept_pos)
        x_int_pos_good = []

        for line in slope_positive_lines:
            x1 = line[0]
            y1 = image_shape[0] - line[1]
            slope = line[4]
            y_intercept = y1 - slope * x1
            x_intercept = -y_intercept / slope
            if abs(x_intercept - x_int_pos_median) < 0.35 * x_int_pos_median:
                x_int_pos_good.append(x_intercept)

        x_intercept_pos_mean = np.mean(np.asarray(x_int_pos_good))

        # Negative Lines
        x_intercept_neg = []
        for line in slope_negative_lines:
            x1 = line[0]
            y1 = image_shape[0] - line[1]
            slope = line[4]
            y_intercept = y1 - slope * x1
            x_intercept = -y_intercept / slope
            if x_intercept == x_intercept:
                x_intercept_neg.append(x_intercept)

        x_int_neg_median = np.median(x_intercept_neg)
        x_int_neg_good = []

        for line in slope_negative_lines:
            x1 = line[0]
            y1 = image_shape[0] - line[1]
            slope = line[4]
            y_intercept = y1 - slope * x1
            x_intercept = -y_intercept / slope
            if abs(x_intercept - x_int_neg_median) < 0.35 * x_int_neg_median:
                x_int_neg_good.append(x_intercept)

        x_intercept_neg_mean = np.mean(np.asarray(x_int_neg_good))

        # Create new black image
        lane_lines = np.zeros_like(edges_image)
        color_lines = image.copy()

        # Positive Slope Line
        slope = pos_slope_mean
        x1 = x_intercept_pos_mean
        y1 = 0
        y2 = image_shape[0] - (image_shape[0] - image_shape[0] * 0.35)
        x2 = (y2 - y1) / slope + x1

        # Plot positive slope line
        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
        cv2.line(lane_lines, (x1, image_shape[0] - y1), (x2, image_shape[0] - y2), (255, 255, 0), 2)
        cv2.line(color_lines, (x1, image_shape[0] - y1), (x2, image_shape[0] - y2), (0, 255, 0), 4)

        # Negative Slope Line
        slope = neg_slope_mean
        x1_neg = x_intercept_neg_mean
        y1_neg = 0
        x2_neg = (y2 - y1_neg) / slope + x1_neg

        # Plot negative Slope Line
        x1_neg = int(round(x1_neg))
        x2_neg = int(round(x2_neg))
        y1_neg = int(round(y1_neg))
        cv2.line(lane_lines, (x1_neg, image_shape[0] - y1_neg), (x2_neg, image_shape[0] - y2), (255, 255, 0), 2)
        cv2.line(color_lines, (x1_neg, image_shape[0] - y1_neg), (x2_neg, image_shape[0] - y2), (0, 255, 0), 4)

        # Blend Image
        lane_fill = image.copy()
        vertices = np.array([[(x1, image_shape[0] - y1), (x2, image_shape[0] - y2), (x2_neg, image_shape[0] - y2),
                              (x1_neg, image_shape[0] - y1_neg)]], dtype=np.int32)
        color = [241, 255, 1]
        cv2.fillPoly(lane_fill, vertices, color)
        opacity = 0.25
        blended_image = cv2.addWeighted(lane_fill, opacity, image, 1 - opacity, 0, image)
        cv2.line(blended_image, (x1, image_shape[0] - y1), (x2, image_shape[0] - y2), (0, 255, 0), 4)
        cv2.line(blended_image, (x1_neg, image_shape[0] - y1_neg), (x2_neg, image_shape[0] - y2), (0, 255, 0), 4)
        b, g, r = cv2.split(blended_image)
        output_image = cv2.merge((r, g, b))

        # Display the result
        if display_result:
            cv2.imshow('Lane Detection', cv2.resize(blended_image, (600, 400), interpolation=cv2.INTER_AREA))

        return blended_image



def estimate_vehicle_distance(input_image, yolo_detector):
    result_image = yolo_detector.detect_image(Image.fromarray(input_image))
    # cv2.imshow('Vehicle Distance Estimator', cv2.resize(np.asarray(result_image), (600, 400), interpolation=cv2.INTER_AREA))
    return np.asarray(result_image)


def preprocess_image_for_keras(input_img):
    target_image_size_x = 100
    target_image_size_y = 100
    processed_img = cv2.resize(input_img, (target_image_size_x, target_image_size_y))
    processed_img = np.array(processed_img, dtype=np.float32)
    processed_img = np.reshape(processed_img, (-1, target_image_size_x, target_image_size_y, 1))
    return processed_img


def detect_video(video_path, yolo_detector):
    # Load the steering wheel image
    steering_wheel = cv2.imread('resources/steering_wheel_image.jpg', 0)
    rows, cols = steering_wheel.shape

    smoothed_angle = 0
    frame_counter = 0
    frame_rate = 10

    # Load the autopilot model
    autopilot_model = Model(r"models\Stearing.h5")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            break

        if frame_counter == frame_rate:
            # Preprocess the frame for steering prediction
            gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (150, 150))
            steering_angle = autopilot_model.predict(gray)

            # Smoothen the steering angle
            smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
                    steering_angle - smoothed_angle) / abs(
                    steering_angle - smoothed_angle)

            # Rotate the steering wheel image
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
            rotated_steering_wheel = cv2.warpAffine(steering_wheel, rotation_matrix, (cols, rows))
            cv2.imshow("Steering Wheel", rotated_steering_wheel)

            # Estimate vehicle distance using YOLO detector
            vehicle_frame = estimate_vehicle_distance(frame.copy(), yolo_detector)
            final_frame = vehicle_frame

            try:
                laneDetector(frame, True)
                final_frame = laneDetector(vehicle_frame, False)
            except Exception as e:
                print(e)

            # Display the final output
            cv2.imshow('Final Output', cv2.resize(final_frame, (800, 500), interpolation=cv2.INTER_AREA))
            frame_counter = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


def detect_image(file_path, yolo_detector):
    # Load the steering wheel image
    steering_wheel = cv2.imread('resources/steering_wheel_image.jpg', 0)
    rows, cols = steering_wheel.shape

    # Load the autopilot model
    autopilot_model = Model(r"models\Stearing.h5")

    # Read the image
    frame = cv2.imread(file_path)

    # Preprocess the frame for steering prediction
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (150, 150))
    steering_angle = autopilot_model.predict(gray)

    # Smoothen the steering angle
    smoothed_angle = 0.2 * pow(abs((steering_angle - 0)), 2.0 / 3.0) * (
            steering_angle - 0) / abs(steering_angle - 0)

    # Rotate the steering wheel image
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    rotated_steering_wheel = cv2.warpAffine(steering_wheel, rotation_matrix, (cols, rows))
    cv2.imshow("Steering Wheel", rotated_steering_wheel)

    # Estimate vehicle distance using YOLO detector
    vehicle_frame = estimate_vehicle_distance(frame.copy(), yolo_detector)
    final_frame = vehicle_frame

    try:
        laneDetector(frame, True)
        final_frame = laneDetector(vehicle_frame, False)
    except Exception as e:
        print(e)

    # Display the final output
    cv2.imshow('Final Output', cv2.resize(final_frame, (800, 500), interpolation=cv2.INTER_AREA))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
