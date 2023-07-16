import numpy as np
import os
import tensorflow as tf
import cv2

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim
import h5py
import json
import imageio

# given a video name
videoname = 's20-d07-cam-002_05495_042'
# dataset hdf5
dataset_hdf5 = 'mpii_sample.hdf5'
# object det. info. hdf5
object_det_hdf5 = 'par_part_feat_sample.hdf5'
# 'object_detect_vf_ava_resnet101_feat_sample.hdf5'

image_size = inception.inception_resnet_v2.default_image_size
print(image_size, ' --- default image size')
with tf.Graph().as_default():
    image = tf.placeholder(tf.uint8, (None, None, 3))
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn('inception_resnet_v2_2016_08_30.ckpt', slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)
        
        # load the hdf5 file
        with h5py.File(dataset_hdf5, 'r') as f_data:
            with h5py.File(object_det_hdf5, 'r') as f_det:
                count = f_det['rgb/' + videoname + '/count'][()]
                height = f_det['rgb/' + videoname + '/height'][()]
                width = f_det['rgb/' + videoname + '/width'][()]
                # print(count, height, width)
                for ii in range(count):
                    # load the rgb frame image
                    frame = f_data['rgb/' + videoname + '/frame' + str(ii).zfill(8)][:]
                    im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    # print('im: ', im.shape)
                    # lod the detection information
                    dic_info = json.loads(f_det['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
                    num = dic_info['num_detections']
                    boxes = np.asarray(dic_info['detection_boxes'])
                    print(boxes.shape, num)
                    if num == 0:
                        print('0!!!!!!')
                        continue;
                    # define an array to store the prediction score of per bb mini image
                    pred_scores = np.zeros((num, 1001))
                    print(pred_scores)
                    for jj in range(num):
                        box = boxes[jj, :]
                        # print('one bb: ', box.shape)
                        scale_box = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
                        # print('scaled: ', scale_box)
                        mini_bb_im = im[scale_box[0]:scale_box[2], scale_box[1]:scale_box[3], :]
                        # print('mini bb image size: ', mini_bb_im.shape)

                        # for checking whether the bb is applied correctly (saved to test folder)
                        # imageio.imwrite('test/' + str(ii) + '-' + str(jj) + '.jpg', mini_bb_im)

                        _, probability = sess.run([image, probabilities], feed_dict={image:mini_bb_im})
                        pred_scores[jj, :] = probability
                        print('-------- ', jj, ' | prob.: ', probability.shape)
                        # print(probability)
                        '''
                        # The rest codes are only for class prediction output
                        prob = probability[0, 0:]
                        sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]

                        names = imagenet.create_readable_names_for_imagenet_labels()
                        for i in range(5):
                            index = sorted_inds[i]
                            print('Probability %0.2f%% => [%s]' % (prob[index] * 100, names[index]))
                        '''
                    # print(pred_scores.shape, ' -------------- ')
                    # print(pred_scores)
                    # print(np.sum(pred_scores, axis = 1), np.sum(pred_scores, axis = 1).shape) # sum up to 1, and the number of bbs

