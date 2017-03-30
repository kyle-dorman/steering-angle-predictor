### Steering wheel angle prediction
This is my (late) submission for Udacity's steering wheel angle prediction challenge. My work is influenced by  the results of the top 5 teams specifically the winner [Team Komanda](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/komanda) and [Team Chauffer](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur) who both used LSTMs to take advantage of the video sequence. After deciding to work on this project, I also came across a blog post by a fellow Udacity student, Matt Harvey. The [blog post](https://hackernoon.com/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5#.x39teb3gd) covers Matt's exploration of five ways to process videos for classification. 

### Step 0
Checkout environment file by running `conda env create -f environment.yml`.

### Step 1
On aws, clone this project and run `python get_orig_data.py` to download original data set.

### Step 2
clone [this](https://github.com/kyle-dorman/udacity-driving-reader) project and run `./run-bagdump.sh -i ~/steering-angle-predictor/orig_data/Ch2_002 -o ~/steering-angle-predictor/image_data` to convert data files to CSVs and images. Must be an absolute path to this project. 

Note: I ran into some issues getting docker running on the carnd ami, but was able to get it installed eventually using [this walkthrough](https://docs.docker.com/engine/installation/linux/ubuntu/#os-requirements).

### Step 3
run `python generate_bottleneck_data.py` to run images through VGG16 and save results to S3. Once data is run through VGG, I should in theory be able to train the rest of the model locally. 

TODO: Figure out better way to download the bottleneck data.

### Step 4
Profit. Or just train a real model. 

