The jupyter notebook written by Lei is for visualization of video frame images, corresponding fixation (binary eye fixation maps in .mat format) and continuous saliency maps.

Authors provided files:

- 'video': 1000 videos (videoname.AVI)
- 'annotation/videoname/maps': continuous saliency maps in '.png' format
- 'annotation/videoname/fixation': binary eye fixation maps in '.png' format
- 'annotation/videoname/maps': binary eye fixation maps stored in mat file
- 'generate_frame.m': used for extracting the frame images from AVI videos.

You need to install keras packages.
Lei uses 'conda activate myenvclone' to activate the environment to run the ACL model codes (testing stage only by loading pre-trained model weights).
Note that you must use scipy 1.1.0, otherwise it will give you a mistake 'Cannot import scipy.misc.imread'. Solution is reinstall old scipy using: python3 -m pip install scipy==1.1.0 inside conda environment (say lei's myenvclone, for example).

The output of the network is a list with 6 tensors, the first 3 are all (#frames/5, 5, 128, 160, 1)-dimensional and the rest 3 are (#frames/5, 5, 64, 80, 1)-dimensional. #frames/5 must be round up, e.g., 59/5 = 12 and 54/5 = 11.

The author uses (#frames/5, 5, 128, 160, 1) for post-processing of saliency maps, and the generated saliency maps have exactly the same resolution as the resolution of the original video frame.

To extract saliency and attention tensors, please enter folder ACL for Lei's codes
