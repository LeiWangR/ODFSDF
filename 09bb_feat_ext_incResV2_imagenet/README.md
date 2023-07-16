lei_inceptionv1.py and lei_inceptionresnetv2.py are for some small demos (runnable on data61 computer, enter myenv conda environment first).

bb_feat_ext_incresv2_test.py is just Lei's trying codes.

'bb_feat_ext.py' is used for extracting the feature from per bounding box (from object detection pipeline) of frame image. The features are then saved into one single hdf5 file (keep only 4 float points)

'no_bb_feat_ext.py' is used for extracting the features from the whole video frame images. The features are then saved into one single hdf5 file as well.

Please note that you might need to change the index for loading the jpeg streams, for example, hmdb51 and yup all 6 digits behind 'frame' and both start from 1; whereas MPII and Charades are all 8 digits and both start from 0.
