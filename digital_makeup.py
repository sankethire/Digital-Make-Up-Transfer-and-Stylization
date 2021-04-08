
import argparse
import cv2 as cv

class digital_makeup:
	def __init__(self, subject_image_path, example_image_path) -> None:
		self.subject_image = cv.imread(subject_image_path)
		self.example_image = cv.imread(example_image_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Color Transfer")
	parser.add_argument("subject_image", type=str, help="Path to subject image i.e. the image to put makeup on")
	parser.add_argument("example_image", type=str, help="Path to example image i.e. the image from which make makeup is copied")
	args = parser.parse_args()

