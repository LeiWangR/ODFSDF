# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) * 1.0
    
    return e_x / e_x.sum(axis=0) # only difference


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
        # count how many nodes in total
        count_node = 0
        for n in self.detection_graph.as_graph_def().node:
            count_node = count_node + 1
            if 'SecondStageBoxPredictor' in n.name:
                print(n.name, '----------------------')
            if 'SecondStageFeatureExtractor' in n.name:
                print(n.name, 'xxxxxxxxxxxxxxxxxxxxxx')
            # if 'SecondStagePostprocessor' in n.name:
            #     print('**********************', n.name)

            # I think this one is for 90 classes and that's why there are 90 'add' in the nodes
            # if 'MultiClassNonMaxSuppression/add' in n.name:
            #     print('**********************', n.name)
            # if count_node > 8466 and count_node <=10000:
            #     print(n.name)
            # if 'SecondStageBoxPredictor' in n.name:
            #     print(count_node, '  xxoo')
            #     print(n.name)
        print(count_node)

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
        # Researcher: Lei Wang
        # 30 September 2019
        
        # for inception resnet v2
        self.feat_conv = self.detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionResnetV2/Conv2d_7b_1x1/Relu:0')
        self.feat_avg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
        self.reg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/Reshape:0') 
        self.cls = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/Reshape_1:0')

        '''
        # for inception v2
        self.feat_conv = self.detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionV2/Mixed_5c/concat:0')
        self.feat_avg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
        self.reg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/Reshape:0')
        self.cls = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/Reshape_1:0')
        '''
        # self.reg = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0')
        # self.cls = self.detection_graph.get_tensor_by_name('SecondStageBoxPredictor/ClassPredictor/BiasAdd:0')



    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num, feat_conv, feat_avg, reg, cls) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.feat_conv, self.feat_avg, self.reg, self.cls],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)
        print('scores: ', scores, '--- shape: ', scores.shape)
        print('classes: ', classes, '--- shape: ', classes.shape)
        print('num: ', num)
        print('feat. conv. : ', feat_conv.shape)
        print('feat. avg. : ', feat_avg.shape)

        # select only one region I think
        print('-----------------------------------------------')
        print('reg: ', reg.shape)
        # print(reg)
        print('cls:', cls.shape)
        # print(cls)
        
        print('bounding box normalized: ', boxes.shape)

        # test softmax
        X = np.zeros((cls.shape[0], 91))
        for ii in range(cls.shape[0]):
            # one_sample = cls[ii, :]
            one_sample = cls[ii, :, :].reshape(91, )
            vot = softmax(one_sample)
            # print(vot.shape)
            X[ii, :] = vot.reshape(1, 91)
            # print('index: ', ii, ' --- ', np.argmax(vot))
        # sort the softmax
        T = np.max(X, axis = 1)
        votM = np.argmax(X, axis = 1)
        # print(votM)
        
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]

        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))
        # print the bounding box mapped to the original input / resolution
        # print(boxes_list)

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
    # model_path = '/home/wan305/research/ongoing/lei/04human_detector/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    # model_path = '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    # model_path = '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('/home/wan305/research/ongoing/lei/04human_detector/TownCentreXVID.avi')
    # cap = cv2.VideoCapture('/home/wan305/research/ongoing/lei/04human_detector/S001C001P001R001A001_rgb.avi')

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        # img = cv2.resize(img, (224, 224))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            # class 2 represents bicycle
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

