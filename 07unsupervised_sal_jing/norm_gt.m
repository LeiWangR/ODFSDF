%% norm gray saliency maps
img_dir = './RBD/'; %% gray saliency map directory
save_dir = './normed/RBD/'; %% bianry saliency map directory
img_list = dir([img_dir '*' '_RBD.png']);
for i = 1:length(img_list)
    i
    img_cur = imread([img_dir img_list(i).name]);
    level = graythresh(img_cur);
    bw_img = im2bw(img_cur,level);
    bw_img =255* uint8(bw_img);
    imwrite(bw_img,[save_dir img_list(i).name]);
end

