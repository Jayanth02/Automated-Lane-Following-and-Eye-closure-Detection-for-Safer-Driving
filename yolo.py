
import colorsys
import numpy as np
import simpleaudio as sa
from keras.models import load_model
from keras.layers import Input
from PIL import ImageFont, ImageDraw
from yolo_backend.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo_backend.utils import letterbox_image
import os
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

tf.compat.v1.disable_eager_execution()

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, attribute):
        if attribute in cls._defaults:
            return cls._defaults[attribute]
        else:
            return "Unrecognized attribute name '" + attribute + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.setup_fonts()
        self.car_original_height = 60
        self.f = 250  # focal length
        self.alert_audio = sa.WaveObject.from_wave_file('alert_audio/alert.wav')
        self.alert_audio_playing_obj = None
        self.alert_audio_playing_obj_initialized = False
        self.alert_message = "Alert"
        self.alert_color = 'rgb(255, 0, 0)'
        self.high_alert_message = "High Alert"
        self.high_alert_color = 'rgb(0, 0, 255)'
        self.car_original_height = 60

    def setup_fonts(self):
        self.distance_font = ImageFont.truetype('text_font/Roboto-Light.ttf', size=20)
        self.other_vehicle_distance_font = ImageFont.truetype('text_font/Roboto-Light.ttf', size=10)
        self.alert_font = ImageFont.truetype('text_font/Roboto-Bold.ttf', size=60)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as file:
            class_names = file.readlines()
        return [name.strip() for name in class_names]

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as file:
            anchors = file.readline()
        return np.array([float(x) for x in anchors.split(',')]).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('Successfully Loaded {} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(*t)) for t in hsv_tuples]
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32') / 255.0
        image_data = np.expand_dims(image_data, 0)

        out_boxes, _, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found total {} vehicle images'.format(len(out_boxes)))



        max_area = 0
        max_left = 0
        max_right = 0
        max_top = 0
        max_bottom = 0
        id = 0
        height = 1
        draw = ImageDraw.Draw(image)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if predicted_class == 'car' or predicted_class == 'motorbike' or predicted_class == "truck" or predicted_class == "person" or predicted_class == "bicycle":
                w = right - left
                h = top - bottom
                area1 = abs(w * h)

                if area1 > max_area:
                    max_area = area1
                    id = i
                    max_left = left
                    max_right = right
                    max_top = top
                    max_bottom = bottom
                    height = abs(h)
                else:
                    d = (self.car_original_height * self.f) / abs(h)
                    d = d / 12

                    text = '{} {:.2f} {}'.format('d =', d, 'ft')
                    x, y = (left + id + 10, top + id - 25)
                    w, h = self.distance_font.getsize(text)
                    draw.rectangle([left + id, top + id, right - id, bottom - id], outline=(255, 0, 0))
                    draw.rectangle((x, y, x + w, y + h), fill='black')
                    draw.text((x, y), text, fill='white', font=self.other_vehicle_distance_font)

        car_original_height = self.car_original_height
        f = self.f
        d = (car_original_height * f) / height
        d = d / 12
        int_d = int(d)

        if 10 < int_d < 15:
            draw.text((25, 25), self.alert_message, fill=self.alert_color, font=self.alert_font)
        elif int_d < 10:
            draw.text((25, 25), self.high_alert_message, fill=self.high_alert_color, font=self.alert_font)
            if not self.alert_audio_playing_obj_initialized:
                self.alert_audio_playing_obj = self.alert_audio.play()
            else:
                if not self.alert_audio_playing_obj.is_playing():
                    self.alert_audio_playing_obj = self.alert_audio.play()
        else:
            if self.alert_audio_playing_obj is not None:
                if self.alert_audio_playing_obj.is_playing():
                    self.alert_audio_playing_obj.stop()

        text = '{} {:.2f} {}'.format('D = ', d, 'ft')
        x, y = (max_left + id + 10, max_top + id - 25)
        w, h = self.distance_font.getsize(text)
        draw.rectangle([max_left + id, max_top + id, max_right - id, max_bottom - id], outline=(0, 0, 255))
        draw.rectangle((x, y, x + w, y + h), fill='black')
        draw.text((x, y), text, fill='white', font=self.distance_font)

        return image

    def close_session(self):
        self.sess.close()
