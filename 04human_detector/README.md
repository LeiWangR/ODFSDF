To run the codes on data61 cluster, please load all the modules in the given list;
To run the codes on Linux, you might need to install opencv inside the proper conda environment (e.g., Lei's conda environment is myenv) using pip: `python3 -m pip install opencv-python`, this is to show the preview / GUI interface of detection results!

Finally, we decide to use faster R-CNN inception ResNet v2 as the feature detector.
The codes running in both data61 cluster and Lei's data61 desktop (feature extraction checking included). Note that if the model cannot run only because of running out of memory (for example faster_rcnn_nas on data61 cluster the codes all work fine). Don't use ssd model as the accuracy of these models are too low and mostly never show the boundary box of human subjects.

- `tensorflow-human-detection_explore_Lei.py`: this is just Lei's draft codes for playing with human detector and feature extraction from tensorflow graph

- `DetectorAPI.py`: this is the detection API used in the object_det_video_feat.py, this module will return #num_of_effect_detection, bounding_boxes (100 x 4), confidence scores (100, ), object_classes (100, ), average_pool_features (region proposal average pooling features, original dimension is 300 x 1 x 1 x 1536, and the pooling is applied along the 0-dimension, which is 300 proposals)

- `object_det_video_frame_feat.py`: for information extraction from video frame images and stored in one single hdf5 file
- `object_det_video_feat.py`: for information extraction from videos and stored in one single hdf5 file
- `test_hdf5.py` is to check the sample hdf5 file to see if all data are in the correct format
