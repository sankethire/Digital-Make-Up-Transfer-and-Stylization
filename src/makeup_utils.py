import cv2 as cv
import numpy as np
from random import shuffle

def layer_decomposition(dm):
	# decompose subject and warped image into color and ligtness layers
	dm.subject_l, dm.subject_a, dm.subject_b = cv.split(cv.cvtColor(dm.subject_image, cv.COLOR_BGR2LAB))
	dm.example_image_warped_l, dm.example_image_warped_a, dm.example_image_warped_b = cv.split(cv.cvtColor(dm.example_image_warped, cv.COLOR_BGR2LAB))

	# cv.bilateralFilter(src, d, sigmaColor, sigmaSpace)
	# d − A variable of the type integer representing the diameter of the pixel neighborhood.
	# sigmaColor − A variable of the type integer representing the filter sigma in the color space.
	# sigmaSpace − A variable of the type integer representing the filter sigma in the coordinate space.
	bilateral_filter_subject = cv.bilateralFilter(dm.subject_l,9, 75,75)
	dm.face_structure_subject = bilateral_filter_subject
	dm.skin_detail_subject = dm.subject_l - bilateral_filter_subject

	bilateral_filter_example_image_warped = cv.bilateralFilter(dm.example_image_warped_l,9, 75,75)
	dm.face_structure_example_image_warped = bilateral_filter_example_image_warped
	dm.skin_detail_example_image_warped = dm.example_image_warped_l - bilateral_filter_example_image_warped

def color_transfer(dm):
	# subject = dm.subject_image
	# warped_target = dm.example_image_warped
	gamma = 0.8

	# dm.skin_mask
	rc_a = (1-gamma)*dm.subject_a + gamma*dm.example_image_warped_a
	rc_b = (1-gamma)*dm.subject_b + gamma*dm.example_image_warped_b

	dm.rc_a = cv.bitwise_and(rc_a, rc_a, mask=dm.skin_mask)    
	dm.rc_b = cv.bitwise_and(rc_b, rc_b, mask=dm.skin_mask)
	
def skin_detail_transfer(dm):
	# setting delta_subject = 0 to conceal/hide skin details of subject image
	delta_subject = 0
	# setting delta_example = 1 to transfer skin details in example image to resultant skin detail layer (skin_detail_resultant) image
	delta_example = 1

	dm.skin_detail_resultant = delta_subject * dm.skin_detail_subject + delta_example * dm.skin_detail_example_image_warped

def highlight_shading_transfer(dm):
	subject_gaussian = cv.pyrDown(dm.face_structure_subject)
	subject_gaussian_upscale = cv.pyrUp(subject_gaussian)
	# subject_laplacian = dm.face_structure_subject - subject_gaussian_upscale

	example_gaussian = cv.pyrDown(dm.face_structure_example_image_warped)
	example_gaussian_upscale = cv.pyrUp(example_gaussian)
	example_laplacian = dm.face_structure_example_image_warped - example_gaussian_upscale

	special_mask = dm.eyes_mask + dm.nose_outline_mask + dm.outer_mouth_mask
	special_mask = cv.dilate(special_mask, np.ones((5,5),np.uint8), iterations=3)
	special_mask = dm.entire_face_mask - special_mask

	dm.face_structure_resultant = np.where(special_mask, dm.face_structure_subject, example_laplacian + subject_gaussian_upscale)

def lip_makeup(dm):
	# luminance remapping of example image wrt subject image
	# lip_luminance_subject = []
	# lip_luminance_example = []
	# lip_points = []
	# for x in range(dm.lip_mask.shape[0]):
	#     for y in range(dm.lip_mask.shape[1]):
	#         if dm.lip_mask[x,y] == 255:
	#             lip_luminance_subject.append(dm.subject_l)
	#             lip_luminance_example.append(dm.example_image_warped_l)
	#             lip_points.append([x,y])

	# lip_luminance_subject = np.array(lip_luminance_subject)
	# lip_luminance_example = np.array(lip_luminance_example)
	
	# lip_luminance_subject_mean = np.mean(lip_luminance_subject)
	# lip_luminance_subject_std = np.std(lip_luminance_subject)

	# lip_luminance_example_mean = np.mean(lip_luminance_subject)
	# lip_luminance_example_std = np.std(lip_luminance_subject)

	# lip_luminance_remapping_example = (lip_luminance_example - lip_luminance_example_mean) * (lip_luminance_subject_std/lip_luminance_example_std) + lip_luminance_subject_mean
	
	lip_mask_boolean = (dm.lip_mask == 255)

	lip_luminance_subject_mean = np.mean(dm.subject_l, where=lip_mask_boolean)
	lip_luminance_subject_std = np.std(dm.subject_l, where=lip_mask_boolean)

	lip_luminance_example_mean = np.mean(dm.example_image_warped_l, where=lip_mask_boolean)
	lip_luminance_example_std = np.std(dm.example_image_warped_l, where=lip_mask_boolean)

	lip_luminance_remapping_example = ((dm.example_image_warped_l - lip_luminance_example_mean) * (lip_luminance_subject_std/lip_luminance_example_std)) + lip_luminance_subject_mean

	Gaussian = lambda t : np.e**(-0.5*float(t))

	# random_sample = lip_points.copy()
	# shuffle(random_sample)

	# iterations = len(lip_points)//50

	M = np.array([dm.subject_l, dm.subject_a, dm.subject_b])
	example_image_warped_LAB  = np.array([dm.example_image_warped_l, dm.example_image_warped_a, dm.example_image_warped_b])
	# example_image_warped_LAB = cv.cvtColor(dm.example_image_warped, cv.COLOR_BGR2LAB)

	for p in lip_points:
		q_tilda = 0
		argmax_q_tilda = -np.inf
		for i in range(iterations):
			q = random_sample[i]
			current_q_tilda = Gaussian(((p[0]-q[0])**2+(p[1]-q[1])**2)/5) * Gaussian((np.fabs(lip_luminance_remapping_example[q[0],q[1]] - dm.subject_l[p[0],p[1]])/255.0)**2)
			if argmax_q_tilda < current_q_tilda:
				argmax_q_tilda = current_q_tilda
				q_tilda = q
				if argmax_q_tilda >= 0.9:
					break
		M[p[0], p[1]] = example_image_warped_LAB[q_tilda[0], q_tilda[1]]

	dm.subject_with_lip_makeup = cv.cvtColor(dm.subject_image.copy(), cv.COLOR_BGR2LAB)

	for p in lip_points:
		dm.subject_with_lip_makeup[p[0],p[1]][1] = M[p[0],p[1]][1]
		dm.subject_with_lip_makeup[p[0],p[1]][2] = M[p[0],p[1]][2]

	# dm.subject_with_lip_makeup = cv.cvtColor(dm.subject_with_lip_makeup, cv.COLOR_LAB2BGR)
