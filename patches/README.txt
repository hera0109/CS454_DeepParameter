Each of these patches will patch an instance of opencv in '../' directory with a pareto optimal solution. Use '100ind_10gen_run/pareto_optimal_test_set.csv' for reference on which patch is which.

To apply a patch please use `patch -p0 -i [PATCH]`

These patches have only been tested in a specifc revision of opencv. Please execution the following if you have not already executed `setup.bsh`. 

git clone https://github.com/Itseez/opencv.git  && \
cd opencv && \
git checkout `git rev-list -1 --before="Apr 7 2016" master` 
