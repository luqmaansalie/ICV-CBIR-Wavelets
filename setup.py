from transform import WaveletTransform
from featuredetector import SegmentFeatures
import argparse
import glob
import cv2
import time

# exmaple
# python index.py --images s51 --type w

def processImage(imgpath):
	#print(imgpath)
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_LINEAR)

	# describe the image
	if fdtype == "w":
		features = cd.detect(image, imageID)
	else:
		features = cd.describe(image)
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))

# =========================================================================
start_time = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--images", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--type", required = True,
	help = "Type of feature detection. w = avelet, s=segmentation")
args = vars(ap.parse_args())

imgdir = args["images"]
fdtype = args["type"]
f = "data/" + imgdir + "_" + fdtype + ".csv"

# initialize the color descriptor
cd = SegmentFeatures((8, 12, 3))
if fdtype == "w":
	cd = WaveletTransform()

output = open(f, "w")

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(imgdir + "/*.jpg"):
	processImage(imagePath)
for imagePath in glob.glob(imgdir + "/*.png"):
	processImage(imagePath)
for imagePath in glob.glob(imgdir + "/*.tif"):
	processImage(imagePath)

# close the index file
output.close()

print("Feature database computed in %.2fs" % (time.time() - start_time))
