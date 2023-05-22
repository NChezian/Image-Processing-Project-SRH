from skimage.feature import hog
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Loading the image path
os.chdir("C:/Users/Admin/Desktop/allimg")

# Loading the input images
image1 = cv2.imread("Verna.jpg")
image2 = cv2.imread("4M.jpg")

# Grayscale Conversion
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initializing - Feature Detector & Descriptor
sift = cv2.SIFT_create()

# Detect and describe keypoints in both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Initialize the feature matcher
bf = cv2.BFMatcher()

# Match the descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to filter out false matches
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

# Check if there are enough good matches
if len(good_matches) > 10:
    # Extract the keypoint locations from the good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Find the homography between the two images
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Get the dimensions of image 2
    h, w = image2.shape[:2]

    # Define the four corners of image 2 in the original image
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)

    # Transform the corners to the original image using the homography matrix
    dst = cv2.perspectiveTransform(pts, M)

    # Convert the PIL image to numpy array
    i1_np = np.array(image1)

    # Draw a bounding box around the detected object in the original image
    cv2.polylines(i1_np, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)

    # Convert the numpy array back to PIL image
    image1 = Image.fromarray(i1_np)

    cv2.putText(i1_np, 'MATCH FOUND', (40,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 2, cv2.LINE_AA)

# Display the original image with bounding box
cv2.imshow('Result', i1_np)
cv2.waitKey(0)

# Convert PIL image to numpy array before resizing
image1 = np.array(image1)

# Resize the images to the uniform size
size = (100,100)
i1 = cv2.resize(gray1, size)
i2 = cv2.resize(gray2, size)


# Compute HOG features for input images
fd1 = hog(i1, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
fd2 = hog(i2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

np.set_printoptions(threshold=np.inf)

# Display Output
print("HOG feature vector for image1:")
print(fd1)
print("HOG feature vector for image2:")
print(fd2)

# Cosine similarity computation of 2 HOG feature vectors
sim = cosine_similarity(fd1.reshape(1, -1), fd2.reshape(1, -1))[0][0]
print("Cosine similarity of the two HOG feature vectors:", sim)

