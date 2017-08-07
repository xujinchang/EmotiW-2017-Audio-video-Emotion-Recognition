#!/bin/bash

function test()
{
    for i in `ls $1`
    do
        GLOG_logtostderr=1 ../../build/tools/test_net.bin afew_face_conv3d_sport1m_test.prototxt "${1}/${i}" 1400 GPU 2 \
            2>&1 | tee ./log/${i}.log
    done
    for i in `ls ./log`
    do
        echo $i & tail -1 ./log/$i | awk '{print $7}' | tee -a all_accuarcy
    done
}
test $1
exit 0

        
