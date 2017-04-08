## Steering wheel angle prediction
In this project, I am attempting to use Udacity's data from it's [steering wheel angle prediction challenge](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3). My work is influenced by  the results of the top 5 teams specifically the winner [Team Komanda](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/komanda) and [Team Chauffer](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur) who both used LSTMs to take advantage of the video sequence. After deciding to work on this project, I also came across a blog post by a fellow Udacity student, Matt Harvey. The [blog post](https://hackernoon.com/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5#.x39teb3gd) covers Matt's exploration of five ways to process videos for classification. 

My goal to this project was to use all the data at my disposal to predict the next steering wheel angle based on images from the car and the real time data collected from the vehicle. 

[//]: # (Image References)

[left]: ./readme_images/left.jpg "Left Image"
[right]: ./readme_images/right.jpg "Right Image"
[center]: ./readme_images/center.jpg "Center Image"
[model]: ./readme_images/model.png "Model diagram"

### Data
The [dataset](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2) is over 4 gigs of tar-ed ROSBAG taken from the Udacity car driving around Mountain View. The dataset is broken down into six videos. After extracting the data (see below), we are left with a bunch of csv files as well as left, right, and center images from the car. In the official Udacity challenge, challengers where suppose to predict the steering wheel angle using only the images. While this is an interesting concept, in practice models should have access to more vehicle information than just what the camera captures and all of this information should be used when predicting the next steering wheel angle. In training my model I wanted to favor practicality, so I decided to include other vehicle information besides the images.

Below are three images from the car:

Left | Center | Right |
:---:|:---:|:---:|
![alt text][left] |![alt text][center] |![alt text][right] |

I have included the first 50 entries of the "real time" data in the folder `example_data`. Some of real time data is included with each image. This data includes angle, torque, speed, lat, long, and alt in the file `interpolated.csv`. Other interesting data not in `interpolated.csv` includes brake, gear, and throttle. After looking at these three csv files, I didn't see a lot of variability in the data and decided for simplicity and lack of usefulness, the not use it. I also decided not to use latitude, longitude, and altitude because they don't seem useful for predicting steering angle. 

Based on my short exploration of the data, I decided to train my model using the left, right and center images from the car as well as the speed, steering wheel angle, and torque. 

### Training
The model I built can be broken out into two steps. The first is to utilize a pre-trained image classifier like VGG16 to essentially extract image features. For each time-stamp, I applied VGG16 to the left, right and center image and saved the resulting tensors as pickle files. This uses the simple generator in `steering/orig_generator.py` and is trained on aws using the script `steering/generate_bottleneck_data.py`. 

The second step was to train a recurrent network on the processed images and the real time vehicle data (speed, torque, steering angle). For this I used a single GRU layer with a output dimensionality of 256. When training the model I used the last 50 frames as input to the GRU. I chose 50 frames because one of the winning projects suggested this worked best for them. The video is 20 frames per second so this equates to looking at the last 2.5 seconds of information. To really take advantage of the GRU's memory capabilities, I set it to run "statefully". In Keras, this meant I had to declare a fixed batch_size, I used 32. This all meant that I was passing a 6D tensor (32, 50, 3, bottleneck_data_shape) and a 3D tensor (32, 50, 3) to my model which definitely added a complexity to my generator (`steering/bottleneck_generator.py`). 

Model digram:
![alt text][model]

One concept I struggled with was how to split up my data for testing and validation. I ended up devising a system by which I randomly selected a continuous 20% from the  middle of each video and used that as the validation data. I then use the continuous data from before and after the validation data as the testing data. To ensure clean training, I reset the model state in between each video training sequence. At the beginning of each epoch I randomly reset the validation sequence starting point. 

It made sense to me to include past vehicle information (speed, steering wheel angle ect.) in the model because in a real self driving car this information should be readily available and would help a car determine how it should turn the wheel. For example, a car moving at a faster speed might need to have a larger steering wheel angle delta than a car moving at a slower speed because whatever is in the image will be arriving sooner. I also decided to include steering wheel angle for similar reasons. One "limitation" that I self imposed was that the most recent five steering wheel angles were not included (set to 0 in my case). This made sense to me because there would probably be some delay in the system between when the car system knows the vehicle data and when the system retaining that information for training received it. Skipping the last five values equated to a 250ms delay. 

The model was trained on aws p2-xlarge instance. It took about 15 minutes to train an epoch and using early stopping, I found my best model after the 28th epoch with a validation loss of 0.027. You can view the training results in the csv file `model_logs_32_100_50_0.001_10.csv` or using the tensorboard logs `tensorboard_logs.zip`. 

## Data extraction and model training steps

### Step 1: Setup aws environment
Checkout environment file by running `conda env create -f environment.yml`. On aws, clone this project and run `python ./bin/get_orig_data.py` to download original data set.

### Step 2: Convert ROSBAGs to images and csv files
clone [this](https://github.com/kyle-dorman/udacity-driving-reader) project and run `./run-bagdump.sh -i ~/steering-angle-predictor/orig_data/Ch2_002 -o ~/steering-angle-predictor/image_data` to convert data files to CSVs and images. Must be an absolute path to this project. 

Note: I ran into some issues getting docker running on the carnd ami, but was able to get it installed eventually using [this walk-through](https://docs.docker.com/engine/installation/linux/ubuntu/#os-requirements).

### Step 3: generate processed images
run `python ./bin/generate_bottleneck_data.py` to run images through VGG16 and save results to S3. Model training was broken up into image processing (VGG16) and recurrent training to allow (in theory) for the second half to be trained locally on my machine.

### Step 4: Train recurrent model
run `python ./bin/train.py` to train and save the recurrent model. In my current best model, I trained for a maximum of 100 epochs, with an early stopping min_delta of 0.01, and an early stopping patience of 10. The model ended up stopping after the 38th epoch. 

### Step 5: Create video displaying actual steering angle, predicted steering angle, and absolute error percentage.
run `python ./bin/video.py`. 

## Conclusions
This was a really fun project because it was my first opportunity to work with ROSBAG data, recurrent layers, Keras functional model, and multi inputs to a model. I think I learned a lot about how recurrent networks work and will definitely be able to use this knowledge in the future. Although I have already learned a lot, there is still a ton of work I would to on the project in the future.

I initially split up the training into "bottleneck" training using VGG and the recurrent training because I hoped I would be able to perform the recurrent training on my local GPU, this was not the case in practice and splitting up the training was a huge pain in developing clean generators to work this. Right now I have three generators, one for each step of training and one for generating the video. If I had built my model end to end, I would not have had this problem, although my model may have taken longer to train. 

The current results of this project are a bit disappointing. My goal was to have a video of actual and predicted steering wheel angles to show similar to the videos from the Udacity challenge ([link](https://vimeo.com/196356123)). In my case, when running my model to generate video output, the model is returning the same steering angle every time. Something is clearly wrong but the whole prediction process very complicated and I can't tell right now if the model is flawed or the way I am passing the input data is incorrect. The model also requires me to input a batch of 32 frames at a time which isn't practical in a real self driving car. If I were to go back I would want to simplify the model architecture to one training process and figure out a better way to run my test and validation splits. I would also want to consider training my own model end to end which would give faster calculation speeds and play with more of the hyper parameters.

