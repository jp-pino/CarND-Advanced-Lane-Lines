## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_test.png "Road Transformed"
[image3]: ./output_images/threshold_test.png "Binary Example"
[image4]: ./output_images/warp_test.png "Warp Example"
[image5]: ./output_images/detector1.png "Fit Visual 1"
[image5]: ./output_images/detector2.png "Fit Visual 2"
[image6]: ./output_images/output.png "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Everything in my code is broken up into classes in different libraries to enhance modularity. The idea is that each object can perform a specific task and be tunnable on its own. The camera matrix computation is handled by a `Calibration` object (class in `src/camera/calibration.py`). When this object is constructed, it reads in many calibration images and begins the calibration process.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

At this point the object is finally initialized, which means we can now call its `undistort()` method. This method takes in an image and uses the calculated camera calibration and distortion coefficients to undistort an image. 

Using this method, we obtain results like the following:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As I mentioned before, a `Calibration` object is used to undistort images. The modularity of the code allows us to observe the output of the pipeline at this step: 

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image that identifies where the lane lines are most like to be found. This step is performed by a `Threshold` object (class in `src/camera/threshold.py`). The constructor for this object allows the user to tune the thresholding parameters in order to dial in the expected results. 

The thresholding process uses a combination of red (RGB) and saturation (HSV) channels, and the sobel gradient in the x axis. 

We can observe here an example of the output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As with previous steps in the pipeline, a `Perspective` object (class in `src/camera/perspective.py`) handles the calibration and action for this step. The constructor takes in percentual values to specify the following parameters: 

- The width of the ROI as a percentage of the width of the image (`width_base`)
- The center of the base of the ROI as a percentage of the width of the image (`center_base`)
- The width of the ROI as a percentage of the width of the image (`width_top`)
- The center of the base of the ROI as a percentage of the width of the image (`center_top`)
- The height of the ROI as a percentage of the height of the image (`top`)

The perspective transform takes place in the `warp()` function, which takes as input an image (`img`). The function calculates the source (`src`) and destination (`dst`) points based on the input image and returns the transformed image.  I chose the hardcode the source and destination points in the following manner:

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels identification is performed by a `Detector` object (class in `src/lane/detector.py`). This class implements two methods to perform lane detection `__sliding_window()` and `__search_around_poly()`. These methods are not exposed to the user however, only the `find()` method, which takes a binary warped image and chooses which detection function to use to find the lane pixels. 

If no detection has been made previously or the lost count is met, `__sliding_window()` is chosen. This method takes in the image and performs a histogram search over the lower part. With this as its starting point, the method identifies window boundaries in its vicinity and adds the pixels in this area to its list of identified pixels. The average position of these pixels is then recalculated and the next windows is centered around this new value. 

Once this detection has been made, the `__fit()` method is called internally to fit a polinomial to each group of identified pixels. This calculation also uses the identified pixels from the last `nsmooth` frames to smooth out the results by averaging them. 

Having found the polinomials, the next call to `find()` will perform a more targeted search for pixels around these polinomials. This greatly reduces computation time.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane is calculated by a `Curvature` object (class in `src/lane/curvature.py`). This object stores the pixel-to-meter conversion for the x and y axes in its constructor. It then can use the `calculate()` method (taking the height at which to calculate the curvature and the polinomial to calculate it on)  to find the real world radius of curvature of the lane, by finding a circle that osculates the polinomial at the point which corresponds to the specified height. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I've identified in my pipeline is that the counter for lost frames (when the `Detector` can't find a lane) is the same for both lanes. This forces the pipeline to run `__sliding_windows()` on both lanes even when only one of them is the problem. The pipeline could be made more robust if the lanes were identified on a more independent manner. 