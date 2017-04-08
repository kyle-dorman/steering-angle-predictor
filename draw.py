#!/bin/python 

import numpy as np 
import cv2

# taken from https://github.com/commaai/research
def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) #* deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

# taken from https://github.com/commaai/research
def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

# taken from https://github.com/commaai/research
def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
	height = img.shape[0]
	max_line_height = height // 3
	width_midpoint = img.shape[1]/2
	height_offset = np.arange(0., max_line_height, 1.0)
	x_offset, _ = calc_lookahead_offset(speed_ms, angle_steers, height_offset)

	pts = [(width_midpoint + point[0], height - point[1]) for point in zip(x_offset, height_offset)]

	return draw_path(img, pts, color=color)

def draw_path(img, pts, thickness=1, color=(0, 0, 255)):
	for point in pts:
		# (x, y)
		top_left = (int(point[0] - thickness), int(point[1] - thickness))
		bottom_right = (int(point[0] + thickness), int(point[1] + thickness))
		cv2.line(img, top_left, bottom_right,color, thickness)

	return img

def draw_error(img, error, font_size=2, color=(255, 255, 255)):
	text = "Absolute Error(%): {:.3f}".format(error)
	height = img.shape[0]
	cv2.putText(img, text, (50, height - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, color)
	return img
