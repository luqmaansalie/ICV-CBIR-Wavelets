import numpy as np
import csv
import cv2

class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
	
	def stdFilter(self, img, queryImg):
		beta = 0.5

		std1 = np.std(np.array(img)[:64])
		queryStd1 = np.std(np.array(queryImg)[:64])
		mintheta1 = std1 * beta
		maxtheta1 = std1 / beta
		
		std2 = np.std(np.array(img)[64:128])
		queryStd2 = np.std(np.array(queryImg)[64:128])
		mintheta2 = std2 * beta
		maxtheta2 = std2 / beta

		std3 = np.std(np.array(img)[128:])
		queryStd3 = np.std(np.array(queryImg)[128:])
		mintheta3 = std3 * beta
		maxtheta3 = std3 / beta

		#print(str(queryStd1) + "... std1:" + str(std1) + " ... maxtheta1: " + str(maxtheta1) + " ... mintheta1: " + str(mintheta1))
		#print(str(queryStd2) + "... std2:" + str(std2) + " ... maxtheta2: " + str(maxtheta2) + " ... mintheta2: " + str(mintheta2))
		#print(str(queryStd3) + "... std3:" + str(std3) + " ... maxtheta3: " + str(maxtheta3) + " ... mintheta3: " + str(mintheta3))

		if (mintheta1 < queryStd1 < maxtheta1) or (mintheta2 < queryStd2 < maxtheta2 and mintheta3 < queryStd3 < maxtheta3):
			return True
		return False
	
	def euclideanDistance(self, img, queryImg):
		dist = 0.4 * np.linalg.norm(np.array(queryImg[:64])-np.array(img[:64]))
		dist+= 0.3 * np.linalg.norm(np.array(queryImg[64:128])-np.array(img[64:128]))
		dist+= 0.3 * np.linalg.norm(np.array(queryImg[128:])-np.array(img[128:]))
		
		#print(dist)
		
		if dist < 1000:
			return dist
		return -1

	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}
		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			for row in reader:
				features = [float(x) for x in row[1:]]
				
				d = -1
				if self.stdFilter(features, queryFeatures):
					d = self.euclideanDistance(features, queryFeatures)
				
				results[row[0]] = d

			f.close()

		results = sorted([(v, k) for (k, v) in results.items()])

		distFilter = {}

		for r in results:
			#print(r[0])
			#if -1 < r[0] < 502 and len(distFilter) < limit:
			if -1 < r[0] and len(distFilter) < limit:
				distFilter[r[1]] = r[0]
		
		distFilter = sorted([(v, k) for (k, v) in distFilter.items()])
		#print(distFilter)
		#print(results[0])
		
		return distFilter