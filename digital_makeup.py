from src import face_alignment
from src import makeup_utils
from src import xdog

import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imutils import opencv2matplotlib

class digital_makeup:
	def __init__(self, subject_image_path, example_image_path, gamma, delta_subject, show_intermediary=False) -> None:
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

		self.gamma = gamma
		self.delta_subject = delta_subject

	def process(self):
		# Face shape related
		face_alignment.extract_face_triangles(self)
		face_alignment.warp_example(self)
		face_alignment.make_masks(self)

		# Makeup
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
		
		# Blur boundaries
		eroded_mask = cv.erode(self.entire_face_mask, np.ones((1, 1), np.uint8))
		contours, _ = cv.findContours(eroded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		self.blurred_face_contour = np.zeros_like(eroded_mask)
		cv.drawContours(self.blurred_face_contour, contours, -1, 255, 3)

		subject_makeup_blurred = cv.GaussianBlur(self.subject_image, (9, 9), 0)
		blurred_face_contour_boolean = (self.blurred_face_contour == 255)

		self.subject_makeup[blurred_face_contour_boolean] = subject_makeup_blurred[blurred_face_contour_boolean]
		
		# self.subject_makeup = cv.seamlessClone(self.subject_makeup_mask_bgr, self.subject_image, self.makeup_mask, (self.subject_image.shape[1]//2, self.subject_image.shape[0]//2), cv.MIXED_CLONE)

		self.xdog_makeup_lab = cv.cvtColor(cv.cvtColor(self.xdog, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2LAB)
		self.xdog_makeup_lab[:,:,1:] = np.where(cv.merge([self.makeup_mask, self.makeup_mask]) == 255, self.subject_makeup_mask_lab[:,:,1:], self.xdog_makeup_lab[:,:,1:])
		self.xdog_makeup_lab[:,:,0] = np.where(self.skin_mask == 255, 0.25*self.face_structure_resultant + 0.75*self.xdog_makeup_lab[:,:,0], self.xdog_makeup_lab[:,:,0])
		self.xdog_makeup_lab[:,:,0] = np.where(self.lip_mask == 255, 0.75*self.face_structure_resultant + 0.25*self.xdog_makeup_lab[:,:,0], self.xdog_makeup_lab[:,:,0])
		# self.xdog_makeup_lab[:,:,1:] = self.gamma*self.xdog_makeup_lab[:,:,1:]
		self.xdog_makeup = cv.cvtColor(self.xdog_makeup_lab, cv.COLOR_LAB2BGR)

		# Showcase
		plt.subplot(2, 3, 1)
		plt.imshow(opencv2matplotlib(self.subject_image))
		plt.subplot(2, 3, 2)
		plt.imshow(opencv2matplotlib(self.xdog_makeup))
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
	parser.add_argument("-lf", "--light-foundation", action="store_true", help="Apply light foundation (normal otherwise)")
	parser.add_argument("-lc", "--light-color", action="store_true", help="Apply light makeup color (normal otherwise)")
	args = parser.parse_args()

	delta_subject = 1 if args.light_foundation else 0
	gamma = 0.5 if args.light_color else 0.8
	self = digital_makeup(args.subject_image, args.example_image, gamma, delta_subject, args.show_intermediary)
	self.process()