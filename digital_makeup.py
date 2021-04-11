from src import face_alignment
from src import makeup_utils

import argparse
import numpy as np
import cv2 as cv

class digital_makeup:
	def __init__(self, subject_image_path, example_image_path) -> None:
		self.subject_image = cv.imread(subject_image_path)
		self.example_image = cv.imread(example_image_path)

		# pyrUp pyrDown fix, requires shape of image
		x, y, _ = self.subject_image.shape
		x = (x//2)*2
		y = (y//2)*2
		self.subject_image = self.subject_image[:x,:y,:]
		x, y, _ = self.example_image.shape
		x = (x//2)*2
		y = (y//2)*2
		self.example_image = self.example_image[:x,:y,:]

		# Properties of dm that will be set by other functions\
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

		self.makeup_mask = dm.entire_face_mask - dm.eyes_mask - dm.inner_mouth_mask

		self.subject_makeup_mask_lab[:,:,0] = dm.face_structure_resultant + dm.skin_detail_resultant
		self.subject_makeup_mask_lab[:,:,0] = cv.bitwise_and(self.subject_makeup_mask_lab[:,:,0], self.subject_makeup_mask_lab[:,:,0], mask=self.makeup_mask)
		self.subject_makeup_mask_lab[:,:,1:] = cv.bitwise_or(np.array([dm.rc_a, dm.rc_b]),dm.subject_with_lip_makeup, mask=self.makeup_mask)

		self.subject_makeup_mask_bgr = cv.cvtColor(self.subject_makeup_mask_lab, cv.COLOR_LAB2BGR)

		cv.imshow("a", self.subject_makeup_mask_bgr)
		cv.waitKey(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Color Transfer")
	parser.add_argument("subject_image", type=str, help="Path to subject image i.e. the image to put makeup on")
	parser.add_argument("example_image", type=str, help="Path to example image i.e. the image from which make makeup is copied")
	args = parser.parse_args()

	dm = digital_makeup(args.subject_image, args.example_image)
	dm.process()