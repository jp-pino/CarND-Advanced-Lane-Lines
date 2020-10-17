import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Camera threshold class
# - Applies RGB, Sobel, S-channel thresholding
class Threshold():
  def __init__(self, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(200, 255)):
    self.s_thresh = s_thresh
    self.sx_thresh = sx_thresh
    self.r_thresh = r_thresh

  def process(self, img):
    # Copy image to preserve original data
    img = np.copy(img)

    # RGB thresholding
    r_channel = img[:,:,0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= self.r_thresh[0]) & (r_channel <= self.r_thresh[1])] = 1
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack((r_binary, sxbinary, s_binary)) * 255
    
    # Combined binary image
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (r_binary == 1)) | (sxbinary == 1)] = 1

    return combined_binary, color_binary


if __name__ == "__main__":
  threshold = Threshold(s_thresh=(120, 255), r_thresh=(220, 255), sx_thresh=(50, 80))
  
  image = mpimg.imread('test_images/test4.jpg')

  image, color = threshold.process(image)

  fig = plt.figure(figsize=(10, 5))

  fig.add_subplot(1, 2, 1)
  plt.imshow(plt.imread('test_images/test4.jpg'))
  plt.title("Original")

  fig.add_subplot(1, 2, 2)
  plt.imshow(image, cmap = 'gray')
  plt.title("Thresholded")

  plt.show(block = True)