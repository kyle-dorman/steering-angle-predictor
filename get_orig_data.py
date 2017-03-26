#!/bin/python 

import os
import glob
import shutil

from util import download_s3
from util import full_path
from util import untar_data

_file = "Ch2_002"
_filegz = _file + ".tar.gz"
_folder = "orig_data"

def download():
	if os.path.isfile(full_path(_filegz)) == False:
		print("Unable to find " + _filegz + ". Downloading now...")
		download_s3(_filegz)
		print('Download Finished!')
	else:
		print(_filegz, "already downloaded.")

def unpack():
	if os.path.isdir(full_path(_folder + "/" + _file)) == False:
		print("unpacking", _filegz)
		untar_data(_filegz)
		shutil.move(full_path(_file), full_path(_folder))
	else:
		print(_file, "already unpacked.")

def per_file_folder():
	for file in glob.glob(_folder + "/" + _file + "/*.bag"):
		folder = file.split(".")[0]

		if not os.path.exists(full_path(folder)):
			os.makedirs(full_path(folder))

		print("Moving", file, "to", folder)
		shutil.move(full_path(file), full_path(folder))

download()
unpack()
per_file_folder()
