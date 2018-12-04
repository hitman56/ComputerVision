import cv2,sys
import sOcropus as sO
from itertools import chain
import numpy as np
import scipy.ndimage.filters
import scipy.misc

def blurImage(img,k):
	return cv2.blur(img, (k, k))

def methodOne(blurimg,serial):
	# Guassian Sharpening
	furtherBlur=cv2.GaussianBlur(blurimg,(5,5),0)
	sharp=cv2.addWeighted(furtherBlur,1.5,blurimg,-0.5,0)
	cv2.imwrite('images/sharpGBlur{}.png'.format(serial),sharp)

def methodTwo(blurimg,serial):
	# Sharpening using Laplacian Filter
	laplacian=cv2.Laplacian(blurimg,cv2.CV_8UC1)
	blurimg += laplacian
	blurimg -= np.amin(blurimg)
	blurimg *= 255/np.amax(blurimg)
	cv2.imwrite('images/sharpLBlur{}.png'.format(serial),blurimg)

def methodThree(blurimg,serial):
	#Mod Laplacian
	#Here we have implemented Laplacian using Lapl. Kernel (same as cv2.Lap)
	A0 = blurimg
	A0 = (A0 - np.amin(A0))*255.0 /(np.amax(A0)-np.amin(A0))
	kernel      = np.ones((3,3))*(-1)
	kernel[1,1] = 8
	Lap        = scipy.ndimage.filters.convolve(A0, kernel)
	ShF         = 100                   #Sharpening factor!
	Laps        = Lap*ShF/np.amax(Lap) 
	A           = A0 + Laps 
	A = np.clip(A, 0, 255)
	cv2.imwrite('images/sharpMLBlur{}.png'.format(serial),A)

def methodFour(blurimg,serial):
	kernel=np.ones((1,1),np.uint8)
	dilation = cv2.dilate(blurimg,kernel,iterations = 3)
	ret,dilation=cv2.threshold(blurimg,200,255,0)


	cv2.imwrite('images/sharpDBlur{}.png'.format(serial),dilation)



if __name__=="__main__":
	try:
		imgpath=sys.argv[1]
	except:
		print('Provide File Name. Example: python imageSharpen.py images/sample1.png')
	else:	
		img=cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (0,0), fx=1.5, fy=1.5) 

		# Creating Sample Blurr Images.
		blur1=blurImage(img, 1)
		cv2.imwrite('images/blur1.png',blur1)
		blur2=blurImage(img, 3)
		cv2.imwrite('images/blur2.png',blur2)
		blur3=blurImage(img, 5)
		cv2.imwrite('images/blur3.png',blur3)

		#Gaussian Sharping
		methodOne(blur1,1)
		methodOne(blur2,2)
		methodOne(blur3,3)

		#Laplacian Sharpening
		methodTwo(blur3,3)

		#Mod. Laplacian
		methodThree(blur3, 3)

		# Dilation
		methodFour(blur3, 3)


