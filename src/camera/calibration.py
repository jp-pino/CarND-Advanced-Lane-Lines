import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Camera calibration class
# - Find chessboard corners
# - Correct camera distortion
# - Apply to images
class Calibration:
  def __init__(self, route = 'camera_cal/calibration*.jpg', preview = False):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(route)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            if preview:
              cv2.imshow('img',img)
              cv2.waitKey(500)

    # Close preview window
    if preview:
      cv2.destroyAllWindows()

    # Get camera distortion correction parameters
    retval, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
  
  def undistort(self, img):
    return cv2.undistort(img, self.cameraMatrix, self.distCoeffs)



if __name__ == "__main__":
  cal = Calibration()

  fig = plt.figure(figsize=(10, 5))

  fig.add_subplot(1, 2, 1)
  plt.imshow(plt.imread('test_images/test2.jpg'))
  plt.title("Distorted")

  fig.add_subplot(1, 2, 2)
  plt.imshow(cal.undistort(plt.imread('test_images/test2.jpg')))
  plt.title("Undistorted")

  plt.show(block = True)
