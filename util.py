#!/bin/python

import boto3
import os
import sys
import os.path
import zipfile
from urllib.request import urlretrieve
import pickle
import threading
import glob
import tarfile

# Get full path to a resource underneath this project (CarND-Behavioral-Cloning)
def full_path(name):
    base_dir_name = "steering-angle-predictor"
    base_dir_list = os.getcwd().split("/")
    i = base_dir_list.index(base_dir_name)
    return "/".join(base_dir_list[0:i+1]) + "/" + name

# Save and fetch data and saved models from S3. Useful for working between AWS and local machine.

bucket_name = 'kd-carnd'
key_name = 'steering-angle-predictor/'
region_name = 'us-east-2'

def upload_s3(rel_path):
	bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)
	print("Uploading file", rel_path)
	bucket.upload_file(full_path(rel_path), key_name + rel_path, Callback=UploadProgressPercentage(rel_path))
	print("Finished uploading file", rel_path)

def download_s3(rel_path):
	bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)

	print("Downloading file", rel_path)
	bucket.download_file(key_name + rel_path, full_path(rel_path), Callback=DownloadProgressPercentage(rel_path))
	print("Finished downloading file", rel_path)

def download_bottleneck_features():
  for i in [1,2,4,5,6]:
    download_s3("bottleneck_data/HMB_{}.p".format(i))

class UploadProgressPercentage(object):
  def __init__(self, filename):
    self._filename = filename
    self._size = float(os.path.getsize(filename))
    self._seen_so_far = 0
    self._lock = threading.Lock()
  def __call__(self, bytes_amount):
    # To simplify we'll assume this is hooked up
    # to a single filename.
    with self._lock:
      self._seen_so_far += bytes_amount
      percentage = (self._seen_so_far / self._size) * 100
      sys.stdout.write(
        "\r%s  %s / %s  (%.2f%%)" % (
          self._filename, self._seen_so_far, self._size,
          percentage))
      sys.stdout.flush()

class DownloadProgressPercentage(object):
  def __init__(self, filename):
    self._filename = filename
    self._seen_so_far = 0
    self._lock = threading.Lock()
  def __call__(self, bytes_amount):
    # To simplify we'll assume this is hooked up
    # to a single filename.
    with self._lock:
      self._seen_so_far += bytes_amount
      sys.stdout.write(
        "\r%s --> %s bytes transferred" % (
          self._filename, self._seen_so_far))
      sys.stdout.flush()

def download_file(url, file):
  """
  Download file from <url>
  :param url: URL to file
  :param file: Local file path
  """
  if os.path.isfile(file) == False:
    print("Unable to find " + file + ". Downloading now...")
    urlretrieve(url, file)
    print('Download Finished!')
  else:
    print(file + " already downloaded.")

def unzip_data(zip_file_name, location):
  """
  unzip file 
  :param zip_file_name: name of zip file
  :param location: path to unzip location
  """
  with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
    print("Extracting zipfile " + zip_file_name + "...")
    zip_ref.extractall(location)

def untar_data(tar_file_name):
  with tarfile.open(tar_file_name, 'r:gz') as in_file:
    in_file.extractall()

def zipdir(path, ziph):
  # ziph is zipfile handle
  for root, dirs, files in os.walk(path):
    for file in files:
      ziph.write(os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

def open_pickle_file(file_name):
  """
  open a pickled file
  :param file_name: name of file
  """
  print("Unpickling file " + file_name)
  full_file_name = full_path(file_name)
  with open(full_file_name, mode='rb') as f:
    return pickle.load(f)

def open_large_pickle_file(file_name):
  file_path = full_path(file_name)
  max_bytes = 2**31 - 1

  bytes_in = bytearray(0)
  input_size = os.path.getsize(file_path)
  print("Opening {} which is size {}.".format(file_name, input_size))
  with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
      bytes_in += f_in.read(max_bytes)
  data = pickle.loads(bytes_in)

  return data

def save_pickle_file(file, data):
  """
  save an object as a pickled file
  :param file: name of file
  :param data: python object to save
  """
  abs_file = full_path(file)
  pickle.dump(data, open(abs_file, "wb" ) )

def stop_instance():
  ec2 = boto3.resource('ec2', region_name='us-east-1')
  instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
  ids = [i.id for i in instances]
  ec2.instances.filter(InstanceIds=ids).stop() # .terminate()


