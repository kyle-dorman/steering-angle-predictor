### Steering wheel angle prediction
This is my (late) submission for Udacity's setter wheel angle prediction challenge. My work is influenced by  the results of the top 5 teams. 

### Step 0
Checkout envrionment file by running `conda env create -f environment.yml`.

### Step 1
On aws, clone this project and run `python get_orig_data.py` to download original data set.

### Step 2
clone [this](https://github.com/kyle-dorman/udacity-driving-reader) project and run `./run-bagdump.sh -i ~/steering-angle-predictor/orig_data/Ch2_002 -o ~/steering-angle-predictor/image_data` to convert data files to csvs and images. Must be an absolute path to this project.

### Step 3
run `python generate_bottleneck_data.py` to run images thorugh VGG16 and save results to S3. Once Data is run through VGG, it should be useable from local machine. 

### Step 4
Profit. Or just train a real model. 

