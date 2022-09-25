import cv2
import glob
for i in range (10,37):
	for filename in glob.glob("Resized/"+str(i)+'/*.png'): #assuming gif#im=Image.open(filename)
		print("file------------------- ", filename)
		#path="ao.JPG"
		#img = cv2.imread("Resized/1/DSC_1456.png",0)
		img=cv2.imread(filename,0)
		#img = cv2.imread('21.png',0)

		# global thresholding
		ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

		# Otsu's thresholding
		ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# Otsu's thresholding after Gaussian filtering
		blur = cv2.GaussianBlur(img,(5,5),0)
		ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# plot all the images and their histograms
		images = [img, 0, th1,
				  img, 0, th2,
				  blur, 0, th3]
		titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
				  'Original Noisy Image','Histogram',"Otsu's Thresholding",
				  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
		write=filename[7:len(filename)]
		#cv2.imshow("im ", th2)
		cv2.imwrite("Otsu"+write, th2)

		#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		print("Otsu"+write)
		#cv2.imwrite("Resized/"+write, img_gray)
		#print("Resized/"+write)