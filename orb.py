#############  ORB  #############
#################################

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the images and convert images to the single channel grayscale images
img1 = cv2.imread("./images/homo_deus/8.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/homo_deus/7.jpg", cv2.IMREAD_GRAYSCALE)

# Add a gaussian blur
img1_blur = cv2.GaussianBlur(img1, (0,0), 0.5)
img2_blur = cv2.GaussianBlur(img2, (0,0), 0.5)

# Initiate STAR detector
orb = cv2.ORB_create(nfeatures=8000)

# find the keypoints with ORB
img1_kp = orb.detect(img1_blur, None)
img2_kp = orb.detect(img2_blur, None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1_blur, img1_kp)
kp2, des2 = orb.compute(img2_blur, img2_kp)

# Brute Force Matching with the Hamming Distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)

# Show the number of matches in each picture
print("Anzahl der Keypoints in Bild 1: " + str(len(img1_kp)))
print("Anzahl der Keypoints in Bild 2: " + str(len(img2_kp)))
print("Anzahl der Matches: " + str(len(matches)))

# Sort the matches by hamming distance (low to high)
matches = sorted(matches, key=lambda x:x.distance)

# Draw keypoints to the images 
img1_with_kp1 = cv2.drawKeypoints(img1_blur, kp1, None, color=(0,255,0))
img2_with_kp2 = cv2.drawKeypoints(img2_blur, kp2, None, color=(0,255,0))

# Display first image and second image with each keypoints
fx, plots = plt.subplots(1, 2, figsize = (15,10))
plots[0].set_title("First image with Gaussian filter and keypoints")
plots[0].imshow(img1_with_kp1)

plots[1].set_title("Second image with Gaussian filter and keypoints")
plots[1].imshow(img2_with_kp2)

# Draw the top n Matches between the images
n = 100
matching_result = cv2.drawMatches(img1_blur, kp1, img2_blur, kp2, matches[:n], None)

# Display Matches
plt.figure(2, figsize=(15,10))
plt.title('Best ' + str(n) + ' Matches')
plt.imshow(matching_result)
plt.show()