import cv2 as cv
import math
import numpy as np

# To show image output using opencv function
def showimage(imglabel, img):
	cv.imshow(imglabel, img)
	while True:
		k = cv.waitKey(0) & 0xFF
		if k == 27:         # wait for ESC key to exit
			cv.destroyAllWindows()
			break

def xdog_thresholding(image):

    gamma = 0.99
    phi = 200
    epsilon = -0.1
    k = 1.6
    sigma = 0.8

    input_image = image
    # showimage("input_img", input_image)

    # sigma_s = determines amount of smoothing (sigma spatial) -> size of neighbourhood is directly proportional to sigma s 
    # sigma_r = controls how the dissimilar colors within the neighbourhood will be averaged -> larger sigma_r larger regions of constant color
    input_image = cv.edgePreservingFilter(input_image, flags=1, sigma_s=100, sigma_r=0.05)
    # showimage("edge_preserving_input_img", input_image)

    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    # showimage("gray_input_img", input_image)


    input_image = cv.normalize(input_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    # showimage("inp_img_float", input_image)

    gaussian_filter_image1 = cv.GaussianBlur(input_image, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
    gaussian_filter_image2 = cv.GaussianBlur(input_image, ksize=(0, 0), sigmaX=sigma*k, borderType=cv.BORDER_REPLICATE)

    diff_of_gaussian = gaussian_filter_image1 - (gamma*gaussian_filter_image2)
    # showimage("dog", diff_of_gaussian)

    x,y = diff_of_gaussian.shape

    for i in range(x):
        for j in range(y):
            if diff_of_gaussian[i][j] < epsilon:
                diff_of_gaussian[i][j] = 1
            else:
                diff_of_gaussian[i][j] = 1 + math.tanh(phi*(diff_of_gaussian[i][j] - epsilon))


    xdog_image = diff_of_gaussian
    # showimage("xdog", xdog_image)


    otsu_xdog = cv.threshold(xdog_image.astype(np.uint8), 0, 255, cv.THRESH_OTSU)
    otsu_xdog = otsu_xdog[1]

    # showimage("otsu", otsu_xdog)

    mean_val = np.mean(xdog_image)

    for i in range(x):
        for j in range(y):
            if xdog_image[i][j] <= mean_val:
                xdog_image[i][j] = 0.0
            else:
                xdog_image[i][j] = 1.0
    # showimage("xdog_thresholding", xdog_image)

    return otsu_xdog

if __name__ == "__main__":
    img = cv.imread("../input/xdog_subject.png")
    img = xdog_thresholding(img)
    showimage("img", img)
   