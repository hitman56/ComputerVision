import os,sys
import cv2
import subprocess as sp
import Preprocess as pp
from itertools import chain
import pytesseract
from PIL import Image, ImageDraw
import PossibleChar
import numpy as np



# This dictionary consists of original text to calculate accuracy. 
oTranscript={}
oTranscript['images/sample1.png']=['O_sample1Transcript.txt','P_sample1Transcript.txt']

def excecuteCommand(command):
	command=command.split()
	process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
	stdout, stderr = process.communicate()
	for l in stdout.split('\n'):
		print(l)
	print(stderr)
	return

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


def binarization(imgpath):
	command='ocropus-nlbin -n {} -o book'.format(imgpath)
	excecuteCommand(command)
	return


def segmentation(imgpath):
	command='ocropus-gpageseg -n --maxcolseps 0 {}'.format(imgpath)
	excecuteCommand(command)
	return

def extractingText(imgpath):
	command='env PYTHONIOENCODING=UTF-8 ocropus-rpred -m en-default.pyrnn.gz {} -n'.format(imgpath)
	excecuteCommand(command)
	return

def aggregatingText(imgpath):
	command='cat book/{}/*.txt > ocr.txt'.format(imgpath)
	excecuteCommand(command)
	return

def writeInFile(fname,message):
	f=open(fname,"w")
	message=message.encode('utf-8')
	f.write(message)
	f.close()
		
def calculateAccuracy(imgpath,recogFile,type):
	
	orignalText=oTranscript.get(imgpath,'')
	if orignalText:
		orignalText = orignalText[1] if type=='P' else orignalText[0]
		print(orignalText)
		f1=open(orignalText, 'r').readlines()
		f2=open(recogFile, 'r').readlines()
		
		f1=[l1 for l1 in f1 if l1.strip()!='']
		f2=[l2 for l2 in f2 if l2.strip()!='']
		f=open("error.psv","w")
		f.write('Original|Predicted\n')
		total, error=0,0
		for l1,l2 in zip(f1,f2):
			temp1=l1.lower().replace(' ','').strip()
			temp2=l2.lower().replace(' ','').strip()
			#print(temp1,temp2)
			for i,j in zip(range(len(temp1)),range(len(temp2))):
				if temp1[i]!=temp2[j]:
					#print(temp1[i],temp2[i])
					f.write('|'.join([temp1[i],temp2[j]])+"\n")
					error+=1
				total+=1
		f.close()
		return (total-error)*100/total
	else:
		return -1

def invertColor(imgMatrix):
	y,x=chain(imgMatrix.shape)
	for i in range(y):
		for j in range(x):
			if imgMatrix[i][j]==0:
				imgMatrix[i][j]=255
			else:
				imgMatrix[i][j]=0
	return imgMatrix


def methodOne():
	binarization(imgpath)
	path='book/0001.bin.png'
	segmentation(path)
	path='book/0001/*.png'
	extractingText(path)
	#aggregatingText('0001')
	os.system('cat book/0001/*.txt > {}'.format(outpath))
	accuracy=calculateAccuracy(imgpath, outpath,type='O')
	print(accuracy)

def methodTwo(imgpath):
	img=cv2.imread(imgpath)
	# gray,thres=pp.preprocess(img)
	# big = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 
	ret,thresh=cv2.threshold(gray,127,255,0)

	cv2.imwrite("book/0002.bin.png",thresh)
	
	path='book/0002.bin.png'
	segmentation(path)
	path='book/0002/*.png'
	extractingText(path)
	#aggregatingText('0001')
	os.system('cat book/0002/*.txt > {}'.format(outpath))
	accuracy=calculateAccuracy(imgpath, outpath,type='O')
	print(accuracy)


def methodThree(imgpath):
	img=cv2.imread(imgpath)
	gray,thres=pp.preprocess(img)
	thres=invertColor(thres)
	thres = cv2.resize(thres, (0,0), fx=1.5, fy=1.5) 
	cv2.imwrite("book/0003.bin.png",thres)
	
	path='book/0003.bin.png'
	segmentation(path)
	path='book/0003/*.png'
	extractingText(path)
	#aggregatingText('0001')
	os.system('cat book/0003/*.txt > {}'.format(outpath))
	accuracy=calculateAccuracy(imgpath, outpath,type='O')
	print(accuracy)

def methodFour(imgpath):
	img=cv2.imread(imgpath)
	gray,thres=pp.preprocess(img)
	gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 
	cv2.imwrite("book/0004.bin.png",gray)
	result=pytesseract.image_to_string(Image.open("book/0004.bin.png"))
	writeInFile(outpath,result)
	accuracy=calculateAccuracy(imgpath, outpath,type='P')
	print(accuracy)

def methodFive(imgpath):
	img=cv2.imread(imgpath)
	gray,thres=pp.preprocess(img)
	thres=invertColor(thres)
	thres = cv2.resize(thres, (0,0), fx=1.5, fy=1.5) 
	cv2.imwrite("book/0005.bin.png",thres)
	result=pytesseract.image_to_string(Image.open("book/0005.bin.png"))
	writeInFile(outpath,result)
	accuracy=calculateAccuracy(imgpath, outpath,type='P')
	print(accuracy)

def methodSix(imgpath):
	img=cv2.imread(imgpath)
	gray,thres=pp.preprocess(img)
	
	gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 
	ret,thresh=cv2.threshold(gray,127,255,0)
	height,width=chain(thresh.shape)

	_, contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	imgContour = np.zeros((height, width, 3), np.uint8)
	cv2.drawContours(thresh, contours, -1, (255.0, 255.0, 255.0))

	
	cv2.imwrite("book/0006.bin.png",thresh)
	result=pytesseract.image_to_string(Image.open("book/0006.bin.png"))
	writeInFile(outpath,result)
	accuracy=calculateAccuracy(imgpath, outpath,type='P')
	print(accuracy)

	return

def methodSeven(imgpath):
	img=cv2.imread(imgpath)
	gray,thres=pp.preprocess(img)
	

	gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 
	ret,thresh=cv2.threshold(gray,127,255,0)
	_, contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	height,width=chain(thresh.shape)
	imgContour = np.zeros((height, width, 3), np.uint8)
	cv2.drawContours(thresh, contours, -1, (255.0, 255.0, 255.0))

	
	cv2.imwrite("book/0007.bin.png",thresh)
	path='book/0007.bin.png'
	segmentation(path)
	path='book/0007/*.png'
	extractingText(path)
	os.system('cat book/0007/*.txt > {}'.format(outpath))
	accuracy=calculateAccuracy(imgpath, outpath,type='O')
	print(accuracy)

	return


def methodEight(imgpath):
	img=cv2.imread(imgpath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (0,0), fx=1.5, fy=1.5) 
	ret,thresh=cv2.threshold(gray,127,255,0)
	height,width=chain(gray.shape)

	_, contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#contours = sorted(contours, key=cv2.contourArea)
	for contour in contours:
		if cv2.contourArea(contour) > 50:
			rectangleInfo=PossibleChar.PossibleChar(contour)
			#h,w=rectangleInfo.intBoundingRectHeight,rectangleInfo.intBoundingRectWidth
			x,y,w,h=cv2.boundingRect(contour)
			mask = np.zeros((height, width), np.uint8)
			cv2.drawContours(mask, [contour], -1, 255.0, -1)
			dst = cv2.bitwise_and(gray, gray, mask=mask)
			crop = dst[y:y+h, x:x+w]
			showImage(crop)

	
	cv2.imwrite("book/0006.bin.png",thresh)
	result=pytesseract.image_to_string(Image.open("book/0006.bin.png"))
	writeInFile(outpath,result)
	accuracy=calculateAccuracy(imgpath, outpath,type='P')
	print(accuracy)

	return


if __name__ =='__main__':

	try:
		sys.argv[1]
		sys.argv[2]
	except:
		print('**** Provide image location and name of output file. Example: python sOcropus.py images/sample1.png out6.txt')

	else:
		imgpath=sys.argv[1]
		outpath=sys.argv[2]
		#methodOne()
		#methodTwo(imgpath)
		#methodThree(imgpath)
		#methodFour(imgpath)
		#methodFive(imgpath)
		methodSix(imgpath)
		#methodSeven(imgpath)
		#methodEight(imgpath)