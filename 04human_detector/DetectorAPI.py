# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

# Researcher: Lei Wang
# 1 October 2019

import numpy as np
import tensorflow as tf
import cv2
import time

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        # show all related nodes
        # [print(n.name) for n in self.detection_graph.as_graph_def().node]

        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        # add feature extraction here
        # for inception resnet v2
        # self.feat_conv = self.detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionResnetV2/Conv2d_7b_1x1/Relu:0')
        self.feat_avg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')

        '''
        # for inception v2
        # self.feat_conv = self.detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionV2/Mixed_5c/concat:0')
        self.feat_avg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
        '''

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num, feat_avg) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.feat_avg],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()
        
        print("Elapsed Time:", end_time-start_time)

        return int(num[0]), np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), np.mean(np.squeeze(feat_avg), axis=0)

    def close(self):
        self.sess.close()
        self.default_graph.close()
