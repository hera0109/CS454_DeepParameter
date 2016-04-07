#!/bin/bash
wget https://codeload.github.com/Itseez/opencv/zip/master
unzip *.zip
cp CMakeLists.txt opencv-master
cd opencv-master
cmake -DWITH_CUDA:BOOL="0" -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=OFF
