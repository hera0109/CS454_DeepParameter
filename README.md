# DPT-OpenCV

##Basic Setup
Run `./setup.bsh` to download and install OpenCV, the Positive and Negative testcases

Please note that cmake, make, build-essentials are all required for setup.bsh to run successfully ( `sudo apt-get install cmake make build-essentials' )

To compile and run the application:

`CMake .`

`make`

`./classify_images [directory containing images]`

classify_images should output which images have and which images do not contain faces (1 == face, 0 == not face)

##Profiling the software
The directory "profiling" provides two scripts "setup_profiling.bsh" and "remove_profiling.bsh" which, when executed from the "profiling" directory shall setup the application for profiling and revert this action respectively

"profiling/callgrind.out.11739" was produced using the "valgrind" tool on a subset of the positive and negative testcases (after profiling was setup using "setup_profiling.bsh" :

`valgrind --tool=callgrind ./classify_images [directory containing images]'

"profiling/callgraph.png" was generated using "qcachegrind" for Mac OS X (use "kcachegrind" for similar results on Linux)

The "modifiable_files" directory contains the files which shall be modified. 

##Exposing the parameters
In the "modifiable_files directory there is an "expose_integers.bsh" script which when run can extract the integer constants from a program. The usage of the script is as follows:

`./expose_integers.bsh [input file] [the output file] [define_file]'

This script utalises the "replace_integer.pl" script

We used this create the "cascadedetect_exposed.cpp" and "cascadedetect_exposed.hpp" files along with the "replaces.hpp" file which contains all of the extracted integer constants. We add this as an include at the top of the file.

These files are copied into the appropriate directory during setup. As with the profiling directory, the scripts and data are mostly for reproducability purposes.

##Filtering the extracted constants
For optimisation we want to remove all the constants that are of low value or of no use at all. Since, for this example, we have over 500 constants we wrote a script which goes through each constant and runs some basic tests to indicate whether this is worthwhile.

The "run_sensativity_filteration.bsh" script works by iterating through each constant. For each one, it first of all increments it by one. If it does not compile or produces an outcome which crashes on the "sensitivity_set" then this is considered too sensative to really be of value. If it passes this test then the integer constant has 50 added. If the program compiles, runs, and completes running in the normal amount of time (within the 95% confidence interval of the original application) for the "training_st" then we determine this constant to not be sensative enough for our requirements (it may not be necessisary at all!). The goal is to find constants which can be modified slightly without crashing but change the running of the software when modified by a small amount. A sensativity "sweet-spot" as it were.

The output of the "run_sensativity_filteration.bsh" script can be interpreted as follows (it is in a CSV format):

<Constant>,<Not_too_sensative?>,<time_training_set>,<sensative_enough?>

We modify only those constants where <Not_too_sensative?> and <sensative_enough> are true. `cat run_sensativity_filteration_output.csv | awk -F "," '($2=="true" && $4=="true"){print $1}' >replaces_selection.dat'
