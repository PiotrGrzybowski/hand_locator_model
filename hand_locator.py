import numpy as np
import tensorflow as tf

from config import GRAPH, OUTPUT_TENSORS, INPUT_TENSOR, THRESHOLD


class HandLocator:
    def __init__(self):
        self.detection_graph = self.load_detection_graph()
        self.input_tensor = self.load_input_tensor()
        self.output_tensors = self.load_output_tensors()
        self.session = tf.Session(graph=self.detection_graph)

    @staticmethod
    def load_detection_graph():
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH, 'rb') as graph:
                serialized_graph = graph.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def get_all_hands_localization_with_score(self, image):
        return self.predict(image)

    def get_all_hands_localization(self, image):
        return [localization[0] for localization in self.get_all_hands_localization_with_score(image)]

    def get_best_hand_localization_with_score(self, image):
        return max(self.get_all_hands_localization_with_score(image), key=lambda x: x[1])

    def get_best_hand_localization(self, image):
        return self.get_best_hand_localization_with_score(image)[0]

    def cut_all_hands(self, image):
        return [self.cut_area(image, localization) for localization in self.get_all_hands_localization(image)]

    def cut_best_hand(self, image):
        return self.cut_area(image, self.get_best_hand_localization(image))

    def cut_area(self, image, localization):
        xmin, ymin, xmax, ymax = map(int, localization)
        return image[ymin:ymax, xmin:xmax]

    def predict(self, image):
        (boxes, scores, classes, num) = self.session.run(self.output_tensors, feed_dict=self.get_feed_dict(image))
        if len(np.squeeze(boxes)) > 0:
            return self.process_output(image, np.squeeze(boxes), np.squeeze(scores))
        else:
            return []

    def process_output(self, image, boxes, scores):
        height, width, _ = image.shape
        return [(self.get_coordinates(box, height, width), score) for box, score in zip(boxes, scores) if score > THRESHOLD]

    @staticmethod
    def get_coordinates(box, height, width):
        ymin, xmin, ymax, xmax = box
        return xmin * width, ymin * height, xmax * width, ymax * height

    def get_feed_dict(self, image):
        return {self.input_tensor: np.expand_dims(image, axis=0)}

    def __del__(self):
        self.session.close()

    def load_output_tensors(self):
        return [self.detection_graph.get_tensor_by_name(tensor) for tensor in OUTPUT_TENSORS]

    def load_input_tensor(self):
        return self.detection_graph.get_tensor_by_name(INPUT_TENSOR)
