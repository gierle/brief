############  BRIEF  ############
#################################

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the images and convert images to the single channel grayscale images
img1 = cv2.imread("./images/homo_deus/1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/homo_deus/7.jpg", cv2.IMREAD_GRAYSCALE)

# downscale and blur if necessary
#img1 = cv2.pyrDown(img1)
img2 = cv2.pyrDown(img2)

#img2 = cv2.flip(img2, 1)

# Add a gaussian blur
img1_blur = cv2.GaussianBlur(img1, (0,0), 0.1)
img2_blur = cv2.GaussianBlur(img2, (0,0), 1.2)

# Other possible Feature Detectors/Describtors
#sift = cv2.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
#orb = cv2.ORB_create(nfeatures=3000)

# Initiate FAST detector, which was selected in the paper (from CenSurE)
star = cv2.FastFeatureDetector_create() 
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=64,use_orientation = True)

# find the keypoints with STAR (from CenSurE)
kp_img1 = star.detect(img1_blur,None)
kp_img2 = star.detect(img2_blur,None)

# compute the descriptors with BRIEF
kp1, des1 = brief.compute(img1_blur, kp_img1)
kp2, des2 = brief.compute(img1_blur, kp_img2)

# Brute Force Matching with the Hamming Distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)

# Sort the matches by hamming distance (low to high)
matches = sorted(matches, key=lambda x:x.distance)

# Draw keypoints to the images 
img1_with_kp1 = cv2.drawKeypoints(img1_blur, kp1, None)
img2_with_kp2 = cv2.drawKeypoints(img2_blur, kp2, None)

# Display traning image and testing image with keypoints
fx, plots = plt.subplots(1, 2, figsize = (15,10))
plots[0].set_title("First image blurred with keypoints")
plots[0].imshow(img1_with_kp1)

plots[1].set_title("Second image blurred with keypoints")
plots[1].imshow(img2_with_kp2)

# Draw the keypoints from img1_blur and img2_blur and also draw the n best results
n = 100
matching_result = cv2.drawMatches(img1_blur, kp1, img2_blur, kp2, matches[:n], None)

# Display Matches
plt.figure(2, figsize=(15,10))
plt.title('Best Matching Points')
plt.imshow(matching_result)
plt.show()