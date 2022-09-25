import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import timeit
import time
import glob
import numpy as np
from keras.preprocessing import image
#from cannyEdgeDetector import cannyEdgeDetector
def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('/home/anika/Desktop/artificialIntelligence/Simple-Sign-Language-Detector-master/model1.h5')

def predictor():
       

       if result[0][0] == 1:
              #return "1"
              return '1'
       elif result[0][1] == 1:
              #return "2"
              return '2'
       elif result[0][2] == 1:
              #return "3"
              return '3'
       elif result[0][3] == 1:
              #return "4"
              return '4'
       elif result[0][4] == 1:

              return '5'
       elif result[0][5] == 1:
              return '6'
       elif result[0][6] == 1:
              return '7'
       elif result[0][7] == 1:
              return '8'
       elif result[0][8] == 1:
              return '9'
       elif result[0][9] == 1:
              return '10'
       elif result[0][10] == 1:
              return '11'
       elif result[0][11] == 1:
              return '12'
       elif result[0][12] == 1:
              return '13'
       elif result[0][13] == 1:
              return '14'
       elif result[0][14] == 1:
              return '15'
       elif result[0][15] == 1:
              return '16'
       elif result[0][16] == 1:
              return '17'
       elif result[0][17] == 1:
              return '18'
       elif result[0][18] == 1:
              return '19'
       elif result[0][19] == 1:
              return '20'
       elif result[0][20] == 1:
              return '21'
       elif result[0][21] == 1:
              return '22'
       elif result[0][22] == 1:
              return '23'
       elif result[0][23] == 1:
              return '24'
       elif result[0][24] == 1:
              return '25'
       elif result[0][25] == 1:
              return '26'
       elif result[0][26] == 1:
              return '27'
       elif result[0][27] == 1:
              return '28'
       elif result[0][28] == 1:
              return '29'
       elif result[0][29] == 1:
              return '30'
       elif result[0][30] == 1:
              return '31'
       elif result[0][31] == 1:
              return '32'
       elif result[0][32] == 1:
              return '33'
       elif result[0][33] == 1:
              return '34'
       elif result[0][34] == 1:
              return '35'
       elif result[0][35] == 1:
              return '36'

for i in range(1,37):
    f= open("model1/"+str(i)+"write.txt","w+")
    for filename in glob.glob("train_set/"+str(i)+"/"+'*.png'):
        print(filename)
        f.write(filename)
        f.write("---------- ")
        test_image=image.load_img(filename,
        grayscale=True,
        target_size=(128,128),
        interpolation='nearest'
        )
        test_image = image.img_to_array(test_image)  
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        print("result---",i,"----- ",result[0][i-1])       
        #print(predictor())
        st=predictor()
        print(str(i)," gets ",st)
        if st:
            if st==str(i):
                f.write(st)
            else:
                f.write("wrong////////////// "+st)
        f.write("\n")