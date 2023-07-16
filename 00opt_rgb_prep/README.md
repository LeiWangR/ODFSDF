To generate optical flow grayscale videos, using:

`python opt_grayscale_gen.py -rgb_dir 'moments_in_time' -opt_u_dir 'flow_images/u' -opt_v_dir 'flow_images/v'`

To put rgb and optical flow information into one single hdf5 file, using:

`python rgb_opt_hdf5.py -hdf5_dir 'sample.hdf5' -rgb_dir 'moments_in_time' -opt_u_dir 'flow_images/u' -opt_v_dir 'flow_images/v'`

To load the generated hdf5 file and retrieve the data from it, refer to the jupyter notebook: read_hdf5_test.ipynb
