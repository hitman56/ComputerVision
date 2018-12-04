
# coding: utf-8

# In[ ]:


from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


# images/ folder is required as all the images are written in it. 

def showImage(imgVar):
	cv2.imshow("image", imgVar)
	while(True):
		k = cv2.waitKey(0)
		if k == -1:  # if no key was pressed, -1 is returned
			continue
		else:
			break
	cv2.destroyWindow('image')
	return

def edgeDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Blur
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	# Find egdes
	edged = cv2.Canny(gray, 75, 200)
 
	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	# cv2.imshow("Image", image)
	cv2.imwrite("images/Edged.png", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return edged

def largestRectContour(image, edged):
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#print(cnts[1])
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	#print(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	print(cnts) 
	# loop over the contours
	for c in cnts:
	    # approximate the contour
	    peri = cv2.arcLength(c, True)
	    #print(peri)
	    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	    #print(approx)

	    # if our approximated contour has four points, then we
	    # can assume that we have found our screen
	    if len(approx) == 4:
	        screenCnt = approx
	        break
 
	
	cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2)
	# cv2.imshow("Outline", image)
	cv2.imwrite("images/outline.png", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return image,screenCnt



def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 

	# just calculating distances

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# bird eye view
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 

	return warped


if __name__=="__main__":


	image = cv2.imread("images/b.jpg")
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)


	# Step-1: edgeDeetction
	edged = edgeDetection(image)
	# Step-2: draw largest rectangular contour (background)
	image, screenCnt = largestRectContour(image, edged)
	# Step-3: Four point transform to fix alignment of image
	# Four point transform followed by thresholding
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255
	cv2.imwrite('images/laststep.png',warped)
	 









