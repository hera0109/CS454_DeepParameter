# DPT-OpenCV

#Run ./setup.bsh to download and install OpenCV, the Positive and Negative testcases

CMake .
make
./classify_images [directory containing images]

#classify_images should output which images have and which images do not contain faces (1 == face, 0 == not face)
