import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from collections import deque


# Lane detector class
# - Finds probable starting place for lane using histogram detection
# - Performs sliding window search to find rest of lane
# - Remembers data from previous image to find lane in next frame
class Detector():
  def __init__(self, nwindows = 9, margin = 100, minpix = 50, minlanepix = 8000, lost_max = 5, nsmooth = 10):
    self.nwindows = nwindows
    self.margin = margin
    self.minpix = minpix
    self.left_fit = None
    self.right_fit = None
    self.minlanepix = minlanepix
    self.lost = False
    self.lost_count = 0
    self.lost_max = lost_max
    self.recent_left = deque([None for i in range(nsmooth)])
    self.recent_right = deque([None for i in range(nsmooth)])

  def __sliding_window(self, binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]//1.5):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//self.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(self.nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window+1)*window_height
      win_y_high = binary_warped.shape[0] - window*window_height
      # Find the four below boundaries of the window
      win_xleft_low = leftx_current - self.margin
      win_xleft_high = leftx_current + self.margin
      win_xright_low = rightx_current - self.margin
      win_xright_high = rightx_current + self.margin
      
      # Draw the windows on the visualization image
      cv2.rectangle(out_img,(win_xleft_low,win_y_low),
      (win_xleft_high,win_y_high),(0,255,0), 2) 
      cv2.rectangle(out_img,(win_xright_low,win_y_low),
      (win_xright_high,win_y_high),(0,255,0), 2) 
      
      # Identify the nonzero pixels in x and y within the window 
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
      
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
      
      # If you found > minpix pixels, recenter next window
      # (`right` or `leftx_current`) on their mean position
      if good_left_inds.shape[0] > self.minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if good_right_inds.shape[0] > self.minpix: 
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    self.recent_left.rotate(1)
    self.recent_left[0] = [leftx, lefty]
    self.recent_right.rotate(1)
    self.recent_right[0] = [rightx, righty]

    return out_img

  def __search_around_poly(self, binary_warped):

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values 
    # within the +/- margin of our polynomial function 
    # Hint: consider the window areas for the similarly named variables 
    # in the previous quiz, but change the windows to our new search area 
    left_lane_inds = ((nonzerox > (self.left_fit(nonzeroy) - self.margin)) & 
                      (nonzerox < (self.left_fit(nonzeroy) + self.margin)))
    right_lane_inds = ((nonzerox > (self.right_fit(nonzeroy) - self.margin)) & 
                      (nonzerox < (self.right_fit(nonzeroy) + self.margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    self.recent_left.rotate(1)
    self.recent_left[0] = [leftx, lefty]
    self.recent_right.rotate(1)
    self.recent_right[0] = [rightx, righty]
    
    ## Visualization ##
    # Fit new polynomials
    left_fitx, right_fitx, ploty = self.__fit(binary_warped.shape)
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    ## End visualization steps ##
    
    return out_img

  def __fit(self, img_shape):
    leftx = np.array([])
    lefty = np.array([])
    for data in self.recent_left:
      if data is not None:
        leftx = np.concatenate((leftx, data[0]))
        lefty = np.concatenate((lefty, data[1]))

    rightx = np.zeros(0)
    righty = np.zeros(0)
    for data in self.recent_right:
      if data is not None:
        rightx = np.concatenate((rightx, data[0]))
        righty = np.concatenate((righty, data[1]))
    
    # Fit a second order polynomial to each with np.polyfit()
    try:
      self.left_fit = np.poly1d(np.polyfit(lefty, leftx, 2))
      self.right_fit = np.poly1d(np.polyfit(righty, rightx, 2))
    except:
      pass
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = self.left_fit(ploty)
    right_fitx = self.right_fit(ploty)
    
    return left_fitx, right_fitx, ploty


  def find(self, binary_warped, plot = False):
    if self.left_fit is None or self.right_fit is None: 
      # Find our lane pixels first
      out_img = self.__sliding_window(binary_warped)
    else:
      # Find lane lines using previously detected lane
      out_img = self.__search_around_poly(binary_warped)
      # If number of pixels found is lower than threshold, do sliding window search
      # print("lost left: ", self.recent_left[0][0].shape[0], self.recent_left[0][0].shape[0] < self.minlanepix)
      # print("lost right: ", self.recent_right[0][0].shape[0], self.recent_right[0][0].shape[0] < self.minlanepix)
      if (self.recent_left[0][0].shape[0] < self.minlanepix or self.recent_right[0][0].shape[0] < self.minlanepix):
        if self.lost:
          self.lost_count += 1
        if self.lost and self.lost_count == self.lost_max:
          self.lost_count = 0
          self.lost = False
          self.recent_left = deque([None for i in range(len(self.recent_left))])
          self.recent_right = deque([None for i in range(len(self.recent_right))])
          out_img = self.__sliding_window(binary_warped)
        else:
          self.lost = True
      else:
        self.lost = False

    left_fitx, right_fitx, ploty = self.__fit(out_img.shape)

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[self.recent_left[0][1], self.recent_left[0][0]] = [255, 0, 0]
    out_img[self.recent_right[0][1], self.recent_right[0][0]] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    if plot:
      plt.plot(left_fitx, ploty, color='yellow')
      plt.plot(right_fitx, ploty, color='yellow')

    return out_img, self.left_fit, self.right_fit
