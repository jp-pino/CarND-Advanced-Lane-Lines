# Common imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# Project imports
from camera.calibration import Calibration
from camera.threshold import Threshold
from camera.perspective import Perspective
from lane.detector import Detector  
from lane.curvature import Curvature
from lane.plotter import Plotter

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# Initialize calibration filter
calibration = Calibration()

# Initialize thresholding filter
threshold = Threshold(s_thresh=(80, 255), r_thresh=(175, 255))

# Initialize warping filter
perspective = Perspective(width_top=0.07, width_base=0.75, top=0.62)

# Initialize lane detector
detector = Detector(minlanepix = 11000, lost_max = 5, nsmooth = 5)

# Initialize curvature calculator
curvature = Curvature()

# Initialize data plotter
plotter = Plotter()


def pipeline(img):
  # Store original image
  original = np.copy(img)
  # Undistort 
  img = calibration.undistort(img)
  # Threshold
  img, color_binary = threshold.process(img) 
  # return np.dstack((img, img, img)) * 255
  # return color_binary
  # Eagle's eye view
  img = perspective.warp(img)  
  # Detect lanes
  img, left_fit, right_fit = detector.find(img)
  # Plot lane
  img = plotter.plot_warped(img, left_fit, right_fit)

  # Restore original perspective
  img = perspective.warp(img, inverse = True)

  ## Overlay
  # Overlay lane data over original image
  img = cv2.addWeighted(original, 1, img, 1, 0)

  ## Annotations
  # Get curvature
  r_left = curvature.calculate(img.shape[0], left_fit)
  r_right = curvature.calculate(img.shape[0], right_fit)
  # Get lane position
  pos_left = curvature.x_pos(img.shape[0], left_fit)
  pos_right = curvature.x_pos(img.shape[0], right_fit)
  # Print curvature
  img = plotter.plot_unwarped(img, r_left, r_right, pos_left, pos_right)
  return img


white_output = 'project_video_out.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False, threads = 1)