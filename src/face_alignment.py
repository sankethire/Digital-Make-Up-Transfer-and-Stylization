import numpy as np
import cv2 as cv
import dlib
from imutils import face_utils
from imutils import opencv2matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from . import skin_detector

import cv2

hair_luminosity_thresh = 20
forehead_search_headstart = 20

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/shape_predictor_68_face_landmarks.dat')

classes_landmark = {"jaw": (0,16), "right eyebrow": (17, 21), "left eyebrow": (22, 26), "nose": (27, 35), "right eye": (36, 41), "left eye": (42, 47), "outer mouth": (48, 59), "inner mouth": (60, 67), "forehead": (68, 82)}

def part_way_points(start, end, n):
	points = []
	start_x, start_y = start
	end_x, end_y = end
	for i in range(1, n+1):
		points.append(np.array([(i*start_x + (n+1-i)*end_x) // (n+1), (i*start_y + (n+1-i)*end_y) // (n+1)]))
	return np.array(points)

def classify_landmark(index):
	global classes_landmark
	for l_class in classes_landmark:
		end_points = classes_landmark[l_class]
		if end_points[0] <= index <= end_points[1]:
			return l_class
	return None

def select_regions(luminosity_image, message):
	fig, ax = plt.subplots()
	ax.set_title("Click and drag to draw a rectangle. Right click to finalize.\n" + message)
	ax.imshow(luminosity_image, cmap='gray', vmin=0, vmax=255)

	def line_select_callback(eclick, erelease):
		if eclick.button == 3:
			plt.close()
	
	def toggle_selector(event):
		pass
	
	toggle_selector.RS = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=True, button=[1, 3],  minspanx=1, minspany=1, spancoords='pixels', interactive=True)

	fig.canvas.mpl_connect('key_press_event', toggle_selector)
	plt.show()

	return [round(coord) for coord in toggle_selector.RS.extents] # [xmin, xmax, ymin, ymax]

def auto_adjust_forehead_landmarks(landmarks, luminosity_image, forehead_region, hair_region, skin_mask):
	global classes_landmark, hair_luminosity_thresh, forehead_search_headstart
	forehead_range = classes_landmark["forehead"]

	# f_x_min, f_x_max, f_y_min, f_y_max = forehead_region
	# h_x_min, h_x_max, h_y_min, h_y_max = hair_region
	# forehead_image = luminosity_image[f_x_min:f_x_max, f_y_min:f_y_max]
	# hair_image = luminosity_image[h_x_min:h_x_max, h_y_min:h_y_max]
	# f_mean = np.mean(forehead_image)
	# h_mean = np.mean(hair_image)
	# f_std = np.std(forehead_image)
	# h_std = np.std(hair_image)
	# range_factor = 1.5
	# f_h_ratio = f_std/h_std

	# def classify_f_h(luminosity_val):
	# 	# binary to decimal of two flags indicating whether luminosity lies in range of each class
	# 	class_output = 0
	# 	f_dist = np.abs(luminosity_val - f_mean)
	# 	h_dist = np.abs(luminosity_val - h_mean)

	# 	if f_dist <= range_factor * f_std:
	# 		class_output = 1
	# 	if h_dist <= range_factor * h_std:
	# 		if class_output == 0:
	# 			class_output = 2
	# 		else:
	# 			class_output = 3

	# 	if class_output == 0:
	# 		return "f"
	# 	elif class_output == 1:
	# 		return "f"
	# 	elif class_output == 2:
	# 		return "h"
	# 	# Tie breaker
	# 	elif class_output == 3:
	# 		dist_ratio = f_dist / h_dist
	# 		if dist_ratio > f_h_ratio:
	# 			return "h"
	# 		return "f"
		

	for i in range(forehead_range[0], forehead_range[1]+1):
		x, y = landmarks[i]
		y -= forehead_search_headstart
		
		while y > 0:
			# l1 = int(luminosity_image[x, y])
			# l2 = int(luminosity_image[x, y-1])
			# c1 = classify_f_h(l1)
			# c2 = classify_f_h(l2)
			# if l2 - l1 >= hair_luminosity_thresh and c1 != c2:
			# if c1 == "f" and c2 == "h":
			if skin_mask[y, x] == 0:
				break
			y -= 1
		
		landmarks[i] = np.array([x, y])

class interactive_forehead_selector:
	def __init__(self, luminosity_image, landmarks) -> None:
		global classes_landmark
		self.state = 0
		self.luminosity_image = luminosity_image
		self.forehead_range = classes_landmark["forehead"]
		self.landmarks = landmarks
		self.forehead_landmarks = landmarks[self.forehead_range[0]:self.forehead_range[1]+1, :]
		self.selected_index = None
	
	def draw_landmarks(self):
		self.ax.clear()
		self.ax.imshow(self.luminosity_image, cmap='gray', vmin=0, vmax=255)
		x = self.forehead_landmarks[:, 0]
		y = self.forehead_landmarks[:, 1]
		for i in range(len(x)-1):
			self.ax.plot([x[i], x[i+1]], [y[i], y[i+1]], "g-")
		self.ax.scatter(x, y, marker="o", c="b")
		if self.selected_index is not None:
			self.ax.scatter([self.forehead_landmarks[self.selected_index, 0]], [self.forehead_landmarks[self.selected_index, 1]], marker="o", c="r")
		
		self.fig.canvas.draw()
	
	def select_nearest_point(self, x, y):
		best_index = 0
		min_dist = np.inf
		current_index = 0
		for point in self.forehead_landmarks:
			current_dist = np.linalg.norm(np.array((x, y))-np.array(point))
			if current_dist < min_dist:
				min_dist = current_dist
				best_index = current_index
			current_index += 1

		self.selected_index = best_index

	def click_transition(self, event):
		if event.button == 3:
			self.state = 3
			for i in range(len(self.forehead_landmarks)):
				self.landmarks[self.forehead_range[0]+i] = self.forehead_landmarks[i]
			plt.close()
		elif event.button == 1:
			x, y = event.xdata, event.ydata
			if x is None or y is None:
				return
			x = 0 if x < 0 else self.luminosity_image.shape[0] if x > self.luminosity_image.shape[0] else x
			y = 0 if y < 0 else self.luminosity_image.shape[1] if y > self.luminosity_image.shape[1] else y
			if self.state == 1:
				self.select_nearest_point(x, y)
				self.state = 2
			elif self.state == 2: 
				self.forehead_landmarks[self.selected_index, 1] = y
				self.selected_index = None
				self.state = 1
		self.draw_landmarks()
	
	def process(self):
		self.fig, self.ax = plt.subplots()
		self.fig.canvas.mpl_connect("button_press_event", self.click_transition)
		self.state = 1
		self.draw_landmarks()
		plt.show()


def face_points(image):
	global detector, predictor, classes_landmark
	gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	face = detector(gray_image)[0]

	landmarks_shape = predictor(gray_image, face)
	landmarks = face_utils.shape_to_np(landmarks_shape)
	jaw_range = classes_landmark["jaw"]
	l_eb_range = classes_landmark["left eyebrow"]
	r_eb_range = classes_landmark["right eyebrow"]

	# Right Left from perspective of person in the image, could be irrelevant
	rightmost_jaw_point = landmarks[jaw_range[0]]
	leftmost_jaw_point = landmarks[jaw_range[1]]
	leftmost_right_eyebrow_point = landmarks[r_eb_range[-1]]
	rightmost_right_eyebrow_point = landmarks[r_eb_range[0]]
	leftmost_left_eyebrow_point = landmarks[l_eb_range[-1]]
	rightmost_left_eyebrow_point = landmarks[l_eb_range[0]]

	landmarks = np.concatenate([landmarks, part_way_points(rightmost_right_eyebrow_point, rightmost_jaw_point, 2), landmarks[r_eb_range[0]:r_eb_range[1]+1], part_way_points(leftmost_right_eyebrow_point, rightmost_left_eyebrow_point, 1), landmarks[l_eb_range[0]:l_eb_range[1]+1], part_way_points(leftmost_jaw_point, leftmost_left_eyebrow_point, 2)])
	
	forehead_region, hair_region = None, None
	# forehead_region = select_regions(gray_image, "Select a rectangle only containing forehead.")
	# hair_region = select_regions(gray_image, "Select a rectangle only containing hair.")
	skin_mask = skin_detector.process(image)
	# cv.imshow("skin_mask", skin_mask)
	# cv.waitKey(0)
	auto_adjust_forehead_landmarks(landmarks, gray_image, forehead_region, hair_region, skin_mask)
	interactive_forehead_selector(gray_image, landmarks).process()

	return landmarks

# dlib can predict face landmark points out side the image too, fix for this
def landmark_range_fix(landmarks, img_shape):
	for i, (x, y) in enumerate(landmarks):
		changed = False

		if x > img_shape[1]-1:
			x = img_shape[1]-1
			changed = True
		elif x < 0:
			x = 0
			changed = True

		if y > img_shape[0]-1:
			y = img_shape[0]-1
			changed = True
		elif y < 0:
			y = 0
			changed = True
		
		if changed:
			landmarks[i] = np.array([x, y])

def extract_face_triangles(dm):
	dm.subject_face_landmarks = face_points(dm.subject_image)
	landmark_range_fix(dm.subject_face_landmarks, dm.subject_image.shape)
	dm.example_face_landmarks = face_points(dm.example_image)
	landmark_range_fix(dm.example_face_landmarks, dm.example_image.shape)

	subject_face_hull = cv.convexHull(dm.subject_face_landmarks)
	example_face_hull = cv.convexHull(dm.example_face_landmarks)

	subject_face_bounding_box = cv.boundingRect(subject_face_hull)
	example_face_bounding_box = cv.boundingRect(example_face_hull)

	subject_subdiv = cv.Subdiv2D(subject_face_bounding_box)
	for p_x, p_y in dm.subject_face_landmarks:
		subject_subdiv.insert((p_x, p_y))
	dm.subject_triangles = subject_subdiv.getTriangleList()

	example_subdiv = cv.Subdiv2D(example_face_bounding_box)
	for p_x, p_y in dm.example_face_landmarks:
		example_subdiv.insert((p_x, p_y))
	dm.example_triangles = example_subdiv.getTriangleList() # x1 y1 x2 y2 x3 y3

def warp_example(dm):
	dm.example_image_warped = np.zeros_like(dm.subject_image)

	def find_index_of_point(point, point_array):
		for i, current_point in enumerate(point_array):
			if np.all(current_point == point):
				return i
		return -1
	
	# Search index
	for subject_triangle in dm.subject_triangles:
		subject_point1 = subject_triangle[:2]
		subject_point2 = subject_triangle[2:4]
		subject_point3 = subject_triangle[4:6]
		index1 = find_index_of_point(subject_point1, dm.subject_face_landmarks)
		index2 = find_index_of_point(subject_point2, dm.subject_face_landmarks)
		index3 = find_index_of_point(subject_point3, dm.subject_face_landmarks)

		example_point1 = dm.example_face_landmarks[index1]
		example_point2 = dm.example_face_landmarks[index2]
		example_point3 = dm.example_face_landmarks[index3]

		subject_cropped_triangle_coord = np.array([subject_point1, subject_point2, subject_point3], np.int32)
		example_cropped_triangle_coord = np.array([example_point1, example_point2, example_point3], np.int32)

		# Triangulate example
		example_crop_bounding_rect = cv.boundingRect(example_cropped_triangle_coord)
		x, y, w, h = example_crop_bounding_rect
		cropped_example_triangle = dm.example_image[y: y+h, x:x+w]
		# cropped_example_triangle_mask = np.zeros((h, w), np.uint8)

		cropped_example_triangle_coord_relative = example_cropped_triangle_coord - np.tile(np.array([x, y]), (3, 1))

		# cv.fillConvexPoly(cropped_example_triangle_mask, cropped_example_triangle_coord_relative, 255)

		# Triangulate subject
		subject_crop_bounding_rect = cv.boundingRect(subject_cropped_triangle_coord)
		x, y, w, h = subject_crop_bounding_rect
		# cropped_subject_triangle = dm.subject_image[y: y+h, x:x+w]
		cropped_subject_triangle_mask = np.zeros((h, w), np.uint8)

		cropped_subject_triangle_coord_relative = subject_cropped_triangle_coord - np.tile(np.array([x, y]), (3, 1))

		cv.fillConvexPoly(cropped_subject_triangle_mask, cropped_subject_triangle_coord_relative, 255)

		# Transform
		M = cv.getAffineTransform(cropped_example_triangle_coord_relative.astype(np.float32), cropped_subject_triangle_coord_relative.astype(np.float32))
		warped_example_triangle = cv.warpAffine(cropped_example_triangle, M, (w, h))
		warped_example_triangle = cv.bitwise_and(warped_example_triangle, warped_example_triangle, mask=cropped_subject_triangle_mask)

		# Reconstruct
		# Fix to remove white lines present when adding triangles to places
		result_face_area = dm.example_image_warped[y: y+h, x: x+w]
		result_face_area_gray = cv.cvtColor(result_face_area, cv.COLOR_BGR2GRAY)
		_, triangle_fix_mask = cv.threshold(result_face_area_gray, 1, 255, cv.THRESH_BINARY_INV)
		warped_example_triangle = cv.bitwise_and(warped_example_triangle, warped_example_triangle, mask=triangle_fix_mask)

		result_face_area = cv.add(result_face_area, warped_example_triangle)
		dm.example_image_warped[y: y+h, x: x+w] = result_face_area
		# cropped_subject_triangle_mask_3_channel = np.zeros((cropped_subject_triangle_mask.shape[0], cropped_subject_triangle_mask.shape[1], 3), np.uint8)
		# cropped_subject_triangle_mask_3_channel[:,:,0] = np.array([cropped_subject_triangle_mask])
		# cropped_subject_triangle_mask_3_channel[:,:,1] = np.array([cropped_subject_triangle_mask])
		# cropped_subject_triangle_mask_3_channel[:,:,2] = np.array([cropped_subject_triangle_mask])
		# a = cv.bitwise_and(cropped_subject_triangle_mask_3_channel, warped_example_triangle)
		# dm.example_image_warped[y: y+h, x: x+w] = cv.bitwise_or(dm.example_image_warped[y: y+h, x: x+w], a)
		# e = dm.example_image_warped
		# cv.imshow("e", e)
		# cv.waitKey(100)
	
	if dm.show_intermediary:
		plt.subplot(1, 2, 1)
		plt.imshow(opencv2matplotlib(dm.subject_image))
		plt.subplot(1, 2, 2)
		plt.imshow(opencv2matplotlib(dm.example_image_warped))
		plt.show()

def make_masks(dm):
	# C2
	dm.lip_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	# C3
	dm.eyes_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	# C1
	dm.skin_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	# Other used for compositions
	dm.nose_outline_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	dm.entire_face_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	dm.inner_mouth_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)
	dm.outer_mouth_mask = np.zeros((dm.subject_image.shape[0], dm.subject_image.shape[1]), np.uint8)

	jaw_points = [landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "jaw"]
	forehead_points = [landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "forehead"]
	forehead_points.reverse()
	entire_face_points = np.array([jaw_points + forehead_points])

	cv.fillPoly(dm.entire_face_mask, entire_face_points, 255)

	right_eye_points = [landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "right eye"]
	left_eye_points = [landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "left eye"]
	eyes_points = np.array([right_eye_points, left_eye_points])

	cv.fillPoly(dm.eyes_mask, eyes_points, 255)

	outer_mouth_points = np.array([[landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "outer mouth"]])

	cv.fillPoly(dm.outer_mouth_mask, outer_mouth_points, 255)

	inner_mouth_points = np.array([[landmark for index, landmark in enumerate(dm.subject_face_landmarks) if classify_landmark(index) == "inner mouth"]])

	cv.fillPoly(dm.inner_mouth_mask, inner_mouth_points, 255)

	nose_outline_points = np.array([[landmark for index, landmark in enumerate(dm.subject_face_landmarks) if 31 <= index <= 35 or index == 27]])

	cv.polylines(dm.nose_outline_mask, nose_outline_points, True, 255, 2)

	dm.lip_mask = dm.outer_mouth_mask - dm.inner_mouth_mask
	dm.skin_mask = dm.entire_face_mask - dm.eyes_mask - dm.outer_mouth_mask

	if dm.show_intermediary:
		plt.subplot(2, 2, 1)
		plt.imshow(dm.lip_mask, cmap='gray', vmin=0, vmax=255)
		plt.subplot(2, 2, 2)
		plt.imshow(dm.eyes_mask, cmap='gray', vmin=0, vmax=255)
		plt.subplot(2, 2, 3)
		plt.imshow(dm.skin_mask, cmap='gray', vmin=0, vmax=255)
		plt.subplot(2, 2, 4)
		plt.imshow(dm.nose_outline_mask, cmap='gray', vmin=0, vmax=255)
		plt.show()
