import sys
import tensorflow as tf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
import tkinter as tk
import cv2
#import numpy as np
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import timeit
import time
from keras.models import load_model


class StartingPage(QWidget):
	def __init__(self, parent=None):
		super(StartingPage, self).__init__(parent)

		root = tk.Tk()
		self.width = root.winfo_screenwidth()
		self.height = root.winfo_screenheight()
		self.left = 0
		self.top = 0

		self.title = 'Bangla Sign Language detection'
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		self.setStyleSheet("background-color: rgb(181, 124, 25);")


		self.ID = 1
		self.initUI()

	def predictor(self, fileName):
		import numpy as np
		from keras.preprocessing import image
		#classifier = load_model('F:\NSU\Fall 21\CSE499A\Final\CSE499A-Real Time Bangla Sign Language Detector/newTrain3.h5')
		classifier = load_model('F:/NSU/Fall 21/CSE499A/Final/CSE499A-Real Time Bangla Sign Language Detector/model1.h5')
		test_image=image.load_img(fileName,
		   grayscale=True,
		   target_size=(128,128),
		   interpolation='nearest'
		   )

		test_image = image.img_to_array(test_image)  
		test_image = np.expand_dims(test_image, axis = 0)
		result = classifier.predict(test_image)

		if result[0][0] == 1:
			  #return "1"
			return 'অ---1'
		elif result[0][1] == 1:
			  #return "2"
			return 'আ---2'
		elif result[0][2] == 1:
			  #return "3"
			return 'ই---3'
		elif result[0][3] == 1:
			  #return "4"
			return 'উ---4'
		elif result[0][4] == 1:
			return 'এ---5'
		elif result[0][5] == 1:
			return 'উ---6'
		elif result[0][6] == 1:
			return 'ক---7'
		elif result[0][7] == 1:
			return 'খ---8'
		elif result[0][8] == 1:
			return 'গ---9'
		elif result[0][9] == 1:
			return 'ঘ---10'
		elif result[0][10] == 1:
			return 'চ---11'
		elif result[0][11] == 1:
			return 'ছ---12'
		elif result[0][12] == 1:
			return 'জ---13'
		elif result[0][13] == 1:
			return 'ঝ---14'
		elif result[0][14] == 1:
			return 'ট---15'
		elif result[0][15] == 1:
			return 'ঠ---16'
		elif result[0][16] == 1:
			return 'ড---17'
		elif result[0][17] == 1:
			return 'ঢ---18'
		elif result[0][18] == 1:
			return 'ত---19'
		elif result[0][19] == 1:
			return 'থ---20'
		elif result[0][20] == 1:
			return 'দ---21'
		elif result[0][21] == 1:
			return 'ধ---22'
		elif result[0][22] == 1:
			return 'ন---23'
		elif result[0][23] == 1:
			return 'প---24'
		elif result[0][24] == 1:
			return 'ফ---25'
		elif result[0][25] == 1:
			return 'ব---26'
		elif result[0][26] == 1:
			return 'ভ---27'
		elif result[0][27] == 1:
			return 'ম---28'
		elif result[0][28] == 1:
			return 'য়---29'
		elif result[0][29] == 1:
			return 'র---30'
		elif result[0][30] == 1:
			return 'ল---31'
		elif result[0][31] == 1:
			return 'স---32'
		elif result[0][32] == 1:
			return 'হ---33'
		elif result[0][33] == 1:
			return 'ড়---34'
		elif result[0][34] == 1:
			return 'ং---35'
		elif result[0][35] == 1:
			return 'ঃ---36'


	def initUI(self):

		horUnit = int(self.width / 12)
		verUnit = int(self.height / 12)

		
		self.btnE = QPushButton("Capture Image from Camera", self)
		self.btnE.setGeometry(6.5*horUnit, 3*verUnit, 2.6*horUnit, 0.4*verUnit)
		self.btnE.setStyleSheet("background-color: white; font-weight: ; font-size: 18px; color: black;")
		self.btnE.clicked.connect(self.btn_E_clicked)
		

		self.label2 = QLabel(self)
		self.label2.setText("বাংলা Sign Language Detection\n")
		self.label2.setGeometry(1*horUnit, 0.5*verUnit, 10*horUnit, 1.5*verUnit)
		self.label2.setStyleSheet("color: black; font-size: 70px; font-weight: ;")
		self.label2.setAlignment(Qt.AlignCenter)
		
		self.btn4 = QPushButton("Select Image from computer", self)
		self.btn4.setGeometry(2*horUnit, 3*verUnit, 2.6*horUnit, 0.4*verUnit)
		self.btn4.setStyleSheet("background-color: white; font-weight: ; font-size: 18px; color: black;")
		self.btn4.clicked.connect(self.btn_4_clicked)



	def paintEvent(self, event):
		horUnit = int(self.width / 12)
		verUnit = int(self.height / 12)

		painter = QPainter()
		painter.begin(self)
		painter.setRenderHint(QPainter.Antialiasing)
		painter.setPen(QPen(Qt.white, 4, Qt.SolidLine))
		painter.setBrush(QtCore.Qt.white)
		#painter.drawLine(3.6*horUnit, 0.5*verUnit, 3.6*horUnit, 11*verUnit)
		


	
	
	def btn_E_clicked(self): 
		image_x, image_y = 64,64
		cam = cv2.VideoCapture(0)

		img_text = ''
		start = timeit.default_timer()                  
		while True:
			ret, frame = cam.read()
			frame = cv2.flip(frame,1)

			img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

			#lower_blue = np.array([l_h, l_s, l_v])
			#upper_blue = np.array([u_h, u_s, u_v])
			imcrop = img[102:298, 427:623]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
			#mask = cv2.inRange(hsv, 0, 0)
			  
			cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
			cv2.imshow("test", frame)
			#cv2.imshow("mask", hsv)
			stop = timeit.default_timer()
			diff= stop-start

			k = cv2.waitKey(1)

			if k%256 == 27:
				# ESC pressed

			  print("Escape hit, closing...")

			  #print('Time: ', stop - start)  
			  break

			if diff>=5:
			  #print("diss --------- ", diff)
			  diff=0
			  start=start+5
			  img_name = "499AB.png"
			  
			  save_img = cv2.resize(hsv, (image_x, image_y))
			  cv2.imwrite(img_name, save_img)
			  imgo = cv2.imread(img_name,0)

			  # global thresholding
			  ret1,th1 = cv2.threshold(imgo,127,255,cv2.THRESH_BINARY)

			  # Otsu's thresholding
			  ret2,th2 = cv2.threshold(imgo,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			  # Otsu's thresholding after Gaussian filtering
			  blur = cv2.GaussianBlur(imgo,(5,5),0)
			  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			  # plot all the images and their histograms
			  images = [imgo, 0, th1,
					  imgo, 0, th2,
					  blur, 0, th3]
			 
			  #cv2.imshow("im ", th2)
			  cv2.imwrite("res.png", th2)
			  print("{} written!".format(img_name))
			  #cannyEdgeDetector().myMain(img_name)
			  img_text = self.predictor("res.png")


			  img2 = np.zeros((250,500,3),np.uint8)
			  b,g,r,a = 153,255,153,0
			#text = time.strftime("%Y/%m/%d %H:%M:%S %Z", time.localtime()) 
			#cv2.putText(img2,  text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 15, (b,g,r), 1, cv2.LINE_AA)
			  fontpath = "F:/NSU/Fall 21/CSE499A/Final/CSE499A-Real Time Bangla Sign Language Detector/HS_SemiBold.ttf"
			  font = ImageFont.truetype(fontpath, 120)
			  img_pil = Image.fromarray(img2)
			  draw = ImageDraw.Draw(img_pil)
			  if img_text:
			  	draw.text((70, 50),  img_text, font = font, fill = (b, g, r, a))
			  else:
			  	draw.text((70, 50),  "could not find", font = font, fill = (b, g, r, a))
			  img2 = np.array(img_pil)	
			  winname="res"
			  cv2.namedWindow(winname)        # Create a named window
			  cv2.moveWindow(winname, 650,500) 
			  cv2.imshow(winname, img2);cv2.waitKey();cv2.destroyAllWindows()		
		cam.release()
		cv2.destroyAllWindows()

			

	def btn_4_clicked(self):
		img2 = np.zeros((250,500,3),np.uint8)
		b,g,r,a = 153,255,153,0
		fileName=""
		fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "F:/NSU/Fall 21/CSE499A/Final/CSE499A-Real Time Bangla Sign Language Detector/Otsu_again/test_set", "Image Files (*.png *.jpg *.jpeg *.bmp)")
		print(fileName)
		conName="out"
		outut=self.predictor(fileName)
		cv2.namedWindow(conName)
		testIm=cv2.imread(fileName,0)
		cv2.moveWindow(conName,750,400)
		cv2.imshow(conName, testIm)
		#text = time.strftime("%Y/%m/%d %H:%M:%S %Z", time.localtime()) 
		#cv2.putText(img2,  text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 15, (b,g,r), 1, cv2.LINE_AA)
		fontpath = "F:/NSU/Fall 21/CSE499A/Final/CSE499A-Real Time Bangla Sign Language Detector/HS_SemiBold.ttf"
		font = ImageFont.truetype(fontpath, 120)
		img_pil = Image.fromarray(img2)
		draw = ImageDraw.Draw(img_pil)

		if outut:
			draw.text((70, 50),  outut, font = font, fill = (b, g, r, a))
		else:
			draw.text((70, 50),  "could not find", font = font, fill = (b, g, r, a))
		img2 = np.array(img_pil)
		winname="res"
		cv2.namedWindow(winname)        # Create a named window
		cv2.moveWindow(winname, 650,500) 
		cv2.imshow(winname, img2);cv2.waitKey();cv2.destroyAllWindows()




def main():
	app = QApplication(sys.argv)
	obj = StartingPage()
	obj.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
