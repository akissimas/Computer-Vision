# required packages
import cv2
import numpy as np
import skimage.measure
import os
from random import choice
from xlwt import Workbook, easyxf
from pathlib import Path


def calculation_of_scores(img_BGR, blur):

    # 1. compute the Structural Similarity Index (SSIM) between the two images
    (ssim_score, _) = skimage.measure.compare_ssim(img_BGR, blur, multichannel=True, full=True)

    # 2. compute the mean-squared error (mse) between two images.
    mse_score = skimage.measure.compare_mse(img_BGR, blur)

    # 3. compute the absolute difference (absdiff) between the pixels of the two image arrays
    result = cv2.absdiff(img_BGR, blur)  # take the absolute difference of the images
    result = result.astype(np.uint8)  # convert the result to integer type
    absdiff_result = (np.count_nonzero(result) * 100) / result.size  # find percentage difference based on number of pixels that are not zero

    return ssim_score, mse_score, absdiff_result


def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv2.imshow(title, image)
    cv2.resizeWindow(title, 480, 360)


# create folders if they don't exist
Path("./question 1").mkdir(parents=True, exist_ok=True)

# creating excel
wb = Workbook()  # create Workbook
sheet1 = wb.add_sheet('Sheet 1')  # add_sheet is used to create sheet.
style = easyxf('font: bold 1')  # make the labels bold

# save labels to excel
sheet1.write(1, 0, 'Averaging Filter', style)
sheet1.write(2, 0, 'Gaussian Filtering', style)
sheet1.write(3, 0, 'Median Filtering', style)
sheet1.write(4, 0, 'Bilateral Filtering', style)
sheet1.write(0, 1, 'SSIM score', style)
sheet1.write(0, 2, 'MSE score', style)
sheet1.write(0, 3, 'absdiff score', style)


# load random image from folder "images to use"
files = os.listdir("./images to use")
random_image = choice(files)
img_BGR = cv2.imread('./images to use/%s' % random_image)

# and illustrate the results
show_image("Original Image", img_BGR)

cv2.waitKey()


# a. Averaging filter
kernel = (5, 5)  # blurring kernel size
blur = cv2.blur(img_BGR, kernel)

# and illustrate the results
show_image("Averaging Filter", blur)

cv2.imwrite('./question 1/averaging_filter.jpg', blur)  # save image

# calculate the similarity between the images
print("Averaging blur filter")

ssim_score, mse_score, absdiff_result = calculation_of_scores(img_BGR, blur)  # function call

print("\tSSIM score: {:.4f}".format(ssim_score))
sheet1.write(1, 1, ssim_score)  # save score to excel

print("\tMSE score: {:.4f}".format(mse_score))
sheet1.write(1, 2, mse_score)  # save score to excel

print("\tabsdiff score: {:.4f}".format(absdiff_result))
sheet1.write(1, 3, absdiff_result)  # save score to excel

cv2.waitKey()


# b. Gaussian Filtering
blur = cv2.GaussianBlur(img_BGR, (5, 5), 0)

# illustrate the results
show_image("Gauss Filter", blur)

cv2.imwrite('./question 1/gauss_filter.jpg', blur)  # save image

# calculate the similarity between the images
print("Gauss blur Filter")

ssim_score, mse_score, absdiff_result = calculation_of_scores(img_BGR, blur)  # function call

print("\tSSIM score: {:.4f}".format(ssim_score))
sheet1.write(2, 1, ssim_score)  # save score to excel

print("\tMSE score: {:.4f}".format(mse_score))
sheet1.write(2, 2, mse_score)  # save score to excel

print("\tabsdiff score: {:.4f}".format(absdiff_result))
sheet1.write(2, 3, absdiff_result)  # save score to excel

cv2.waitKey()


# c. Median Filtering (highly effective against salt-and-pepper noise)
blur = cv2.medianBlur(img_BGR, 5)

# illustrate the results
show_image("Median Filter", blur)

cv2.imwrite('./question 1/median_filter.jpg', blur)  # save image

# calculate the similarity between the images
print("Median Filter")

ssim_score, mse_score, absdiff_result = calculation_of_scores(img_BGR, blur)  # function call

print("\tSSIM score: {:.4f}".format(ssim_score))
sheet1.write(3, 1, ssim_score)  # save score to excel

print("\tMSE score: {:.4f}".format(mse_score))
sheet1.write(3, 2, mse_score)  # save score to excel

print("\tabsdiff score: {:.4f}".format(absdiff_result))
sheet1.write(3, 3, absdiff_result)  # save score to excel

cv2.waitKey()


# d. Bilateral Filtering
blur = cv2.bilateralFilter(img_BGR, 9, 5, 5)

# illustrate the results
show_image("Bilateral Filter", blur)
cv2.imwrite('./question 1/bilateral_filter.jpg', blur)  # save image

# calculate the similarity between the images
print("Bilateral Filter")

ssim_score, mse_score, absdiff_result = calculation_of_scores(img_BGR, blur)  # function call

print("\tSSIM score: {:.4f}".format(ssim_score))
sheet1.write(4, 1, ssim_score)  # save score to excel

print("\tMSE score: {:.4f}".format(mse_score))
sheet1.write(4, 2, mse_score)  # save score to excel

print("\tabsdiff score: {:.4f}".format(absdiff_result))
sheet1.write(4, 3, absdiff_result)  # save score to excel

cv2.waitKey()
cv2.destroyAllWindows()

wb.save("./question 1/result.xls")  # save excel
print("\n\'result.xls\' created")
