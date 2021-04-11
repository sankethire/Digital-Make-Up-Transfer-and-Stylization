import cv2 as cv
import numpy as np

def layer_decomposition(dm):

    # decompose subject and warped image into color and ligtness layers
    dm.subject_l, dm.subject_a, dm.subject_b = cv.split(cv.cvtColor(dm.subject_image.COLOR_BGR2LAB))
    dm.example_image_warped_l, dm.example_image_warped_a, dm.example_image_warped_b = cv.split(cv.cvtColor(dm.example_image_warped.COLOR_BGR2LAB))

    # why these parameters in bilateral filter
    bilateral_filter_subject = cv.bilateralFilter(dm.subject_l,9, 75,75)
    dm.face_structure_subject = bilateral_filter_subject
    dm.skin_detail_subject = dm.subject_l - bilateral_filter_subject

    # cv.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    # d − A variable of the type integer representing the diameter of the pixel neighborhood.
    # sigmaColor − A variable of the type integer representing the filter sigma in the color space.
    # sigmaSpace − A variable of the type integer representing the filter sigma in the coordinate space.

    bilateral_filter_example_image_warped = cv.bilateralFilter(dm.example_image_warped_l,9, 75,75)
    dm.face_structure_example_image_warped = bilateral_filter_example_image_warped
    dm.skin_detail_example_image_warped = dm.example_image_warped - bilateral_filter_example_image_warped

    

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

    skin_detail_resultant = delta_subject * dm.skin_detail_subject + delta_example * dm.skin_detail_example_image_warped


def lip_makeup(dm):
    # dm.lip_mask

    # luminance remapping of example image wrt subject image
    lip_luminance_subject = []
    lip_luminance_example = []
    for x in range(dm.lip_mask.shape[0]):
        for y in range(dm.lip_mask.shape[1]):
            if dm.lip_mask[x,y] == 255:
                lip_luminance_subject.append(dm.subject_l)
                lip_luminance_example.append(dm.example_image_warped_l)

    lip_luminance_subject = np.array(lip_luminance_subject)
    lip_luminance_example = np.array(lip_luminance_example)
    
    lip_luminance_subject_mean = np.mean(lip_luminance_subject)
    lip_luminance_subject_std = np.std(lip_luminance_subject)

    lip_luminance_example_mean = np.mean(lip_luminance_subject)
    lip_luminance_example_std = np.std(lip_luminance_subject)

    lip_luminance_remapping_example = (lip_luminance_example - lip_luminance_example_mean) * (lip_luminance_subject_std/lip_luminance_example_std) + lip_luminance_subject_mean

    