from src import face_alignment
from src import makeup_utils
from src import xdog

import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imutils import opencv2matplotlib

class digital_makeup:
	def __init__(self, subject_image_path, example_image_path, show_intermediary=False) -> None:
		self.subject_image = cv.imread(subject_image_path)
		self.example_image = cv.imread(example_image_path)
		self.show_intermediary = show_intermediary

		# pyrUp pyrDown fix, requires shape of image
		x, y, _ = self.subject_image.shape
		x = (x//2)*2
		y = (y//2)*2
		self.subject_image = self.subject_image[:x,:y,:]
		x, y, _ = self.example_image.shape
		x = (x//2)*2
		y = (y//2)*2
		self.example_image = self.example_image[:x,:y,:]

		self.xdog = xdog.xdog_thresholding(self.subject_image)
		# Properties of self that will be set by other functions
		# Face landmark points for subject image - 2d array - inner dimension is coord of point
		# self.subject_face_landmarks = None

	def process(self):
		face_alignment.extract_face_triangles(self)
		face_alignment.warp_example(self)
		face_alignment.make_masks(self)

		makeup_utils.layer_decomposition(self)
		makeup_utils.color_transfer(self)
		makeup_utils.skin_detail_transfer(self)
		makeup_utils.highlight_shading_transfer(self)
		makeup_utils.lip_makeup(self)

		self.subject_makeup_mask_lab = np.zeros_like(self.subject_image)

		self.makeup_mask = self.entire_face_mask - self.eyes_mask - self.inner_mouth_mask

		self.subject_makeup_mask_lab[:,:,0] = self.face_structure_resultant + self.skin_detail_resultant
		self.subject_makeup_mask_lab[:,:,1:] = np.where(cv.merge([self.lip_mask, self.lip_mask]) == 255, self.subject_lip_makeup[:,:,1:], cv.merge([self.rc_a, self.rc_b]))
		self.subject_makeup_mask_lab = cv.bitwise_and(self.subject_makeup_mask_lab, self.subject_makeup_mask_lab, mask=self.makeup_mask)

		self.subject_makeup_mask_bgr = cv.cvtColor(self.subject_makeup_mask_lab, cv.COLOR_LAB2BGR)

		self.subject_makeup = np.where(cv.merge([self.makeup_mask, self.makeup_mask, self.makeup_mask]) == 255, self.subject_makeup_mask_bgr, self.subject_image)

		plt.axis('off')
		plt.subplot(2, 3, 1)
		plt.imshow(opencv2matplotlib(self.subject_image))
		plt.subplot(2, 3, 2)
		plt.imshow(opencv2matplotlib(self.xdog))
		plt.subplot(2, 3, 3)
		plt.imshow(opencv2matplotlib(self.subject_makeup))
		plt.subplot(2, 3, 5)
		plt.imshow(opencv2matplotlib(self.example_image))
		plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Color Transfer")
	parser.add_argument("subject_image", type=str, help="Path to subject image i.e. the image to put makeup on")
	parser.add_argument("example_image", type=str, help="Path to example image i.e. the image from which make makeup is copied")
	parser.add_argument("-si", "--show-intermediary", action="store_true", help="Display intermediary steps")
	args = parser.parse_args()

	self = digital_makeup(args.subject_image, args.example_image, args.show_intermediary)
	self.process()