For training

1. prepare data

1.1 generate gray saliency map
- open matlab and locat to the "data_prepare" directory
- open "demo.m"
- change "SRC = '/home/jing-zhang/jing_file/fixation_dataset/train/SALICON/img/';       %Path of input images" to your image path
- run demo.m, saliency map will be save in the "RBD" directory

1.2 get binarized saliency map
- in matlab, locate to the main folder directory
- open "norm_gt.m"
- change "img_dir = './RBD/'; %% gray saliency map directory
- save_dir = './normed/RBD/'; %% bianry saliency map directory" according to your seeting.

2. train the model
- open "model_train" in pycharm
- change "image_root = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/'
- gt_root = '/home/jing-zhang/jing_file/CVPR2020/single_noise/data/DUTS/normed/RBD/'"
- in train.py as needed
- run train.py

------

For testing

- open model_train in pycharm
- open test.py
- change "dataset_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/test/img/'" (testing image directory) and "test_datasets = ['HKU-IS']" (testing dataset name) according to your setting
- run test.py
- results will be saved in './results/VGG16/' + dataset + '/'


------ 

- To use the pre-trained model for saliency maps extraction, please go to the 'model_train' folder. Lei has rewritten the codes so that you can generate the video daliency images and save them into one single hdf5 file.

- All other codes are not directly relevant to our video saliency extraction, please ignore.

- The codes can run on both data61 computer and cluster. Prior to run the codes, please load all the modules first on cluster (just normal modules), and to run the codes on data61 computer, please enter Lei's conda environment: myenv.
