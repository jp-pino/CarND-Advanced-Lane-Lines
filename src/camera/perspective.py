import numpy as np
import cv2

# Camera perspective class
# - Applies perspective warping to implement Bird's Eye view
class Perspective():
  def __init__(self, center_base = 0.02, center_top = 0.5, top = 0.62, width_base = 0.75, width_top = 0.08):
    self.center_base = center_base
    self.center_top = center_top
    self.top = top
    self.width_base = width_base
    self.width_top = width_top
  
  def warp(self, img, inverse = False):
    # Get dimensions
    width = img.shape[1]
    height = img.shape[0]
    
    # Set offset
    offset = (1 - self.width_base) * width

    # For source points (similar to ROI vertices)
    src = np.float32([(self.center_top  * width - (self.width_top * width) / 2, height * self.top), 
                      (self.center_top  * width + (self.width_top * width) / 2, height * self.top), 
                      (width / 2 + self.center_base * width + (self.width_base * width) / 2, height),
                      (width / 2 + self.center_base * width - (self.width_base * width) / 2, height)])


    # For destination points
    dst = np.float32([[offset, 0], 
                      [width - offset, 0], 
                      [width - offset, height], 
                      [offset, height]])

    # Given src and dst points, calculate the perspective transform matrix
    if not inverse:
      M = cv2.getPerspectiveTransform(src, dst)
    else:
      M = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(img, M, img.shape[1::-1])


if __name__ == "__main__":
  pers = Perspective()

  cv2.imshow('result', pers.warp(cv2.imread('test_images/test1.jpg')))
  cv2.waitKey(20000)

  # Close preview window
  cv2.destroyAllWindows()
