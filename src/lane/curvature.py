import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Lane curvature class
# - Calculates real world curvature from an image's lane equations
class Curvature():
  def __init__(self, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    self.ym_per_pix = ym_per_pix
    self.xm_per_pix = xm_per_pix

  def calculate(self, y, poly): 
    y = y * self.ym_per_pix
    coeff0 = np.asarray(poly)[0] * (self.xm_per_pix / (self.ym_per_pix ** 2))
    coeff1 = np.asarray(poly)[1] * (self.xm_per_pix / self.ym_per_pix)
    return ((1 + (2 * coeff0 * y + coeff1) ** 2) ** (3/2)) / abs(2 * coeff0)

  def x_pos(self, y, poly):
    y = y * self.ym_per_pix
    coeff0 = np.asarray(poly)[0] * (self.xm_per_pix / (self.ym_per_pix ** 2))
    coeff1 = np.asarray(poly)[1] * (self.xm_per_pix / self.ym_per_pix)
    coeff2 = np.asarray(poly)[2] * self.xm_per_pix
    return coeff0 * (y ** 2) + coeff1 * y + coeff2