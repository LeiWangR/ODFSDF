00opt_rgb_prep: For computing optical flow from rgb videos, and saved as grayscale video for each u and v components. We also generate a single hdf5 file for both optical flow and rgb videos in order to store videos much more efficiently.

01video2soundwave: shell script for generating the soundwave from rgb videos and saved as wav file (each video has its own generated wav file only if it contains the sound information).

02ext_soundfeat: extract the sound features using pre-trained soundnet8. The features are extracted from the pooling 5 layer based on the authors' NIPS2016 paper.

03saliency_ext: extract the saliency maps using pre-trained ACLNet. The output dimension of saliency tensor from ACLNet is (#total_frames/5, 5, 128, 160, 1). The dimension of the attention tensor (coarse/mosaic square/rectangle highlight the region of interest) is (#total_frame/5, 5, 64, 80, 1). The saliency information are saved into one single hdf5 file that contains the small saliency maps and the attention maps.

04human_detector: in fact not only for human detection but also other object detections as well. The inputs are videos/video frame images, pre-trained model, and specified hdf5 location. The code will save #detections, detection classes, confidence scores, bounding boxes (normalized) and region proposal average pooling features.

05ldof_opt: for large displacement of optical flow computation, the codes haven't been modified for video level and the current codes only work on two frames

06hal_pipeline: for hallucinating the optical flow features, the current codes are just very basic pipeline.

07unsupervised_sal_jing: the codes are obtained through Peter Koniusz from Jing Zhang's CVPR2018 paper. The generated saliency information are saved into one single hdf5 file and it contains the low resolution saliency maps and the big saliency maps (original video resolution).

08feat_ext_i3d: this codes are used to extract the prediction scores for both rgb and opt from the pre-trained I3D RGB and OPT models on Kinetics-400. The output feture dimension is 400D (for both rgo and opt videos)

09bb_feat_ext_incResV2_imagenet: this codes are used to extract the prediction score per bounding box (from the object detection model/pipeline), and the InceptionResnetV2 model is pre-trained on imagenet dataset.

10feature_aggr: this codes are used to aggregate all the extracted information together. This include the object detection information, two saliencies and the imagenet prediction scores per bounding box. We also apply the SIFT/HoG on top of bounding box of mini saliency maps
