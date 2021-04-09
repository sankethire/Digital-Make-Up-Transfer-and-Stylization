import numpy as np
import cv2 as cv
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

hair_luminosity_thresh = 20
forehead_search_headstart = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/shape_predictor_68_face_landmarks.dat')

classes_landmark = {"jaw": (0,16), "right eyebrow": (17, 21), "left eyebrow": (22, 26), "nose": (27, 35), "right eye": (36, 41), "left eye": (42, 47), "lip": (48, 59), "mouth": (60, 67), "forehead": (68, 82)}

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

def auto_adjust_forehead_landmarks(landmarks, luminosity_image, forehead_region, hair_region):
	global classes_landmark, hair_luminosity_thresh, forehead_search_headstart
	forehead_range = classes_landmark["forehead"]

	f_x_min, f_x_max, f_y_min, f_y_max = forehead_region
	h_x_min, h_x_max, h_y_min, h_y_max = hair_region
	forehead_image = luminosity_image[f_x_min:f_x_max, f_y_min:f_y_max]
	hair_image = luminosity_image[h_x_min:h_x_max, h_y_min:h_y_max]
	f_mean = np.mean(forehead_image)
	h_mean = np.mean(hair_image)
	f_std = np.std(forehead_image)
	h_std = np.std(hair_image)
	range_factor = 1.5
	f_h_ratio = f_std/h_std

	def classify_f_h(luminosity_val):
		# binary to decimal of two flags indicating whether luminosity lies in range of each class
		class_output = 0
		f_dist = np.abs(luminosity_val - f_mean)
		h_dist = np.abs(luminosity_val - h_mean)

		if f_dist <= range_factor * f_std:
			class_output = 1
		if h_dist <= range_factor * h_std:
			if class_output == 0:
				class_output = 2
			else:
				class_output = 3

		if class_output == 0:
			return "f"
		elif class_output == 1:
			return "f"
		elif class_output == 2:
			return "h"
		# Tie breaker
		elif class_output == 3:
			dist_ratio = f_dist / h_dist
			if dist_ratio > f_h_ratio:
				return "h"
			return "f"
		

	for i in range(forehead_range[0], forehead_range[1]+1):
		x, y = landmarks[i]
		y -= forehead_search_headstart
		
		while y > 0:
			l1 = int(luminosity_image[x, y])
			l2 = int(luminosity_image[x, y-1])
			c1 = classify_f_h(l1)
			c2 = classify_f_h(l2)
			# if l2 - l1 >= hair_luminosity_thresh and c1 != c2:
			if c1 == "f" and c2 == "h":
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
	
	forehead_region = select_regions(gray_image, "Select a rectangle only containing forehead.")
	hair_region = select_regions(gray_image, "Select a rectangle only containing hair.")
	auto_adjust_forehead_landmarks(landmarks, gray_image, forehead_region, hair_region)
	interactive_forehead_selector(gray_image, landmarks).process()

	return landmarks

def warp_example(dm):
	dm.subject_face_points = face_points(dm.subject_image)
	dm.example_face_points = face_points(dm.example_image)
