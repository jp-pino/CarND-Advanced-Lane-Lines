import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Lane curvature class
# - Calculates real world curvature from an image's lane equations
class Plotter():
  def __init__(self, curvature_text = "Radius: {:.2f}", distance_text = "Distance: {:.2f}", offset_text = "Offset: {:.2f}", xm_per_pix = 3.7/700):
    self.curvature_text = curvature_text
    self.distance_text = distance_text
    self.offset_text = offset_text
    self.count = 0
    self.radius = np.zeros(10)
    self.distance = np.zeros(10)
    self.offset = np.zeros(10)
    self.xm_per_pix = xm_per_pix

  def plot_unwarped(self, img, left_radius, right_radius, left_pos, right_pos): 
    self.distance = np.roll(self.distance, 1)
    self.distance[0] = right_pos - left_pos

    self.radius = np.roll(self.radius, 1)
    self.radius[0] = (left_radius + right_radius) / 2

    self.offset = np.roll(self.offset, 1)
    self.offset[0] = (left_pos + abs(right_pos - left_pos) / 2) - (img.shape[1] / 2) * self.xm_per_pix

    img = cv2.putText(img = img, text = self.curvature_text.format(np.average(self.radius)), org = (50,200), fontFace = 2, fontScale=0.5, color = (255, 255, 255), thickness = 2)
    img = cv2.putText(img = img, text = self.distance_text.format(np.average(self.distance)), org = (50,250), fontFace = 2, fontScale=0.5, color = (255, 255, 255), thickness = 2)
    img = cv2.putText(img = img, text = self.offset_text.format(np.average(self.offset)), org = (50,300), fontFace = 2, fontScale=0.5, color = (255, 255, 255), thickness = 2)

    return img

  def plot_warped(self, img, left_fit, right_fit):
    # Create base image
    warp_zero = np.zeros_like(img)

    # Generate x and y values for plotting
    chan = np.dsplit(warp_zero, warp_zero.shape[-1])[0]
    cols, rows, chans = np.indices(chan.shape)

    # Mask
    chan[(rows > left_fit(cols)) & (rows < right_fit(cols))] = 255

    # Stack RGB
    color = np.dstack((chan, chan, chan * 0))

    img = cv2.addWeighted(img, 1, color, 0.6, 0)

    return img