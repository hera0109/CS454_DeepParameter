# DPT-OpenCV

##Basic Setup
Run `./setup.bsh` to download and install OpenCV, the Positive and Negative testcases

Please note that cmake, make, build-essentials are all required for setup.bsh to run successfully ( sudo apt-get install cmake make build-essentials )

To compile and run the application:
`CMake .`
`make`
`./classify_images [directory containing images]`

classify_images should output which images have and which images do not contain faces (1 == face, 0 == not face)

##Profiling the software
the directory "profiling" provides two scripts "setup_profiling.bsh" and "remove_profiling.bsh" which, when executed from the "profiling" directory shall setup the application for profiling and revert this action respectively

"profiling/callgrind.out.11739" was produced using the "valgrind" tool on a subset of the positive and negative testcases (after profiling was setup using "setup_profiling.bsh" :
`valgrind --tool=callgrind ./classify_images [directory containing images]

"profiling/callgraph.png" was generated using "qcachegrind" for Mac OS X (use "kcachegrind" for similar results on Linux)
