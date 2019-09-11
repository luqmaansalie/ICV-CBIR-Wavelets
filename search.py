from transform import WaveletTransform
from featuredetector import SegmentFeatures
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import csv

# example
# python search.py --images jpg --query 100000.jpg --type s --limit 10

def show_images(images, titles = None):
	#print(titles)
	
	plt.switch_backend('TkAgg')
	fig=plt.figure(figsize=(25, 25))

	l = len(images)

	rows = np.ceil(l/5)
	
	n = 0
	for img in images:
		a = fig.add_subplot(rows, np.ceil(l/rows), n + 1)
		a.set_title(titles[n])
		plt.axis('off')
		plt.imshow(img)
		n += 1
	
	mng = plt.get_current_fig_manager()
	mng.window.state('zoomed')
	plt.show()

def stdFilter(img, queryImg):
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

def euclideanDistance(img, queryImg):
	dist = 0.4 * np.linalg.norm(np.array(queryImg[:64])-np.array(img[:64]))
	dist+= 0.3 * np.linalg.norm(np.array(queryImg[64:128])-np.array(img[64:128]))
	dist+= 0.3 * np.linalg.norm(np.array(queryImg[128:])-np.array(img[128:]))
	
	#print(dist)
	
	if dist < 1000:
		return dist
	return -1

def search(queryFeatures, limit, indexPath):
	results = {}
	with open(indexPath) as f:
		reader = csv.reader(f)
		for row in reader:
			features = [float(x) for x in row[1:]]
			
			d = -1
			if stdFilter(features, queryFeatures):
				d = euclideanDistance(features, queryFeatures)
			
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

# =========================================================================
start_time = time.time()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
	help = "Folder database with saved features")
ap.add_argument("-q", "--query", required = True,
	help = "Name of query image")
ap.add_argument("-t", "--type", required = True,
	help = "Type of feature detection. w = avelet, s=segmentation")
ap.add_argument("-l", "--limit", required = True,
	help = "Number of results to return")
args = vars(ap.parse_args())

imgdir = args["images"]
imgname = imgdir + "/" + args["query"]
fdtype = args["type"]

f = "data/" + imgdir + "_" + fdtype + ".csv"

# initialize the image descriptor
cd = SegmentFeatures((8, 12, 3))
if fdtype == "w":
	cd = WaveletTransform()

query = cv2.imread(imgname)
query512 = cv2.resize(query, (512, 512), interpolation = cv2.INTER_LINEAR)
query = cv2.resize(query, (128, 128), interpolation = cv2.INTER_LINEAR)

if fdtype == "w":
	features = cd.detect(query, imgdir)
else:
	features = cd.describe(query)

# perform the search
results = search(features, int(args["limit"]), f)

print("Found matching images in %.2fs" % (time.time() - start_time))

imgResults = []
imgNames = []

imgResults.append(query512)
imgNames.append("Query: " + imgname[imgname.rfind("/") + 1:])

# loop over the results
for (score, resultID) in results:
	#print(resultID + " " +  str(score))
	
	result = cv2.imread(resultID)
	#result = cv2.resize(result, (512, 512), interpolation = cv2.INTER_LINEAR)
	
	imgResults.append(result)
	imgNames.append(resultID[resultID.rfind("\\") + 1:])

show_images(imgResults, imgNames)
