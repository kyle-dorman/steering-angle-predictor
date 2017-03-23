### Steering wheel angle prediction
THis is my (late) submission for Udacity's setter wheel angle prediction challenge. My work is influenced by  the results of the top 5 teams. 

### Step 1
On aws, clone this project and run `python download.py` to download original data set.

### Step 2
clone [this](https://github.com/kyle-dorman/udacity-driving-reader) project and run `./run-bagdump.sh -i /data -o /output` to convert data files to csvs and images.

### Step 3
run `python generate_bottleneck_data.py` to run images thorugh VGG16 and save results to S3. Once Data is run through VGG, it should be useable from local machine. 

### Step 4
Profit. Or just train a real model. 

