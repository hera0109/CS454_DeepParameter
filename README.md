# DPT-OpenCV

#Run ./setup.bsh to download and install OpenCV, the Positive and Negative testcases
#setup.bsh has yet to be fully run. May be buggy. If you have a problem with then it's your problem

CMake .
make
./classify_images [directory containing images]

#classify_images should output which images have and which images do not contain faces (1 == face, 0 == not face)
