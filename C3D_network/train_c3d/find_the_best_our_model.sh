#!/bin/bash

function test()
{
    for i in `ls $1`
    do
        if [[ $i !=  *.solverstate ]];then
            GLOG_logtostderr=1 ../../build/tools/test_net.bin afew_face_conv3d_sport1m_test.prototxt "${1}/${i}" 120 GPU 2 \
                2>&1 | tee ./log/afew_face_log/${i}.log
        fi
    done

    for i in `ls ./log/afew_face_log/`
    do
        echo $i | tee -a afew_0_accuarcy
        tail -1 ./log/afew_face_log/$i | awk '{print $7}' | tee -a all_accuarcy
    done
}
test $1
exit 0


