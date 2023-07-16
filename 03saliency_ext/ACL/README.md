Source code for ‘Revisiting Video Saliency: A Large-scale Benchmark and a New Model’ (V0)

Run 'main.py';
The results of our model: ACL (Attentive CNN-LSTM) can be found at video_data/video/saliency
The model is trained on the training sets of UCF-sports, Hollywood-2 and DHF1K dataset.


DHF1K dataset can be downloaded from:

Google disk：https://drive.google.com/open?id=1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ
Baidu pan: https://pan.baidu.com/s/110NIlwRIiEOTyqRwYdDnVg

------

The codes for extracting the saliency and attention tensors are in `saliency_attention_extract.py`; and to save them as jpeg stream is in jpeg_hdf5_saliency_attention_extract.py. Highly suggested using `jpeg_hdf5_saliency_attention_extract.py` for memory storage consideration.

For more information, please refer to the codes.

To run the codes in data61 computer, please enter Lei's myenvclone conda environment.
