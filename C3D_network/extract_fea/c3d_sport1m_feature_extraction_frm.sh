#mkdir -p output/c3d/v_ApplyEyeMakeup_g01_c01
#mkdir -p output/c3d/v_BaseballPitch_g01_c01
#cd input/frm
#tar xvzf v_ApplyEyeMakeup_g01_c01.tar.gz
#tar xvzf v_BaseballPitch_g01_c01.tar.gz
#cd ../..
GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin $1 $2  0 16 161 prototxt/test_lbp_output  prob
