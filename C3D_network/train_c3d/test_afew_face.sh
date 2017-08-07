GLOG_logtostderr=1 ../../build/tools/test_net.bin afew_face_conv3d_sport1m_test.prototxt $1 11 GPU 0 \
    2>&1 | tee 3600_log
