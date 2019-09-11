import numpy as np
import pywt
import cv2

class WaveletTransform:
	def detect(self, image, name):
		features = []
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		c1, c2, c3 = cv2.split(image)
		#print(len(c1))
		
		c1 = pywt.dwt2(c1, 'db1')
		c2 = pywt.dwt2(c2, 'db1')
		c3 = pywt.dwt2(c3, 'db1')
		
		#coeffs2 = pywt.dwt2(coeffs2[0], 'db1')
		#cA, (cH, cV, cD) = coeffs2
		
		if name == "jpg45\\100000.jpg":
			print(c1[0])
			print("========")
			#print(c2)
			print("========")
			#print(c3)
		
		features.extend(np.ravel(c1[0][0]))
		features.extend(np.ravel(c2[0][0]))
		features.extend(np.ravel(c3[0][0]))
		#print(len(features))
		return features