# required packages
import cv2
import numpy as np
import skimage.measure
import os
import random
from xlwt import Workbook
from pathlib import Path


# define function to create some noise to an image
def sp_noise(image, prob):
    """
    Add salt and pepper noise to image. Replaces random pixels with 0 or 1.
    prob: Probability of the noise
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def speckle_noise(image):
    """
    Multiplicative noise using speckle = image + image * gauss, where
    gauss is Gaussian noise
    """
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    speckle = image + image * gauss
    return speckle


def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv2.imshow(title, image)
    cv2.resizeWindow(title, 480, 360)


def filter_application(noise_img):
    # a. Averaging filter
    kernel = (5, 5)  # blurring kernel size
    avg_filter = cv2.blur(noise_img, kernel)

    # b. Gaussian Filtering
    gaussian_filter = cv2.GaussianBlur(noise_img, (5, 5), 0)

    # c. Median Filtering (highly effective against salt-and-pepper noise)
    median_filter = cv2.medianBlur(noise_img, 5)

    # d. Bilateral Filtering
    bilateral_filter = cv2.bilateralFilter(noise_img, 9, 5, 5)

    return avg_filter, gaussian_filter, median_filter, bilateral_filter


def calculation_of_scores(img_BGR, blur):

    # 1. compute the Structural Similarity Index (SSIM) between the two images
    (ssim_score, _) = skimage.measure.compare_ssim(img_BGR, blur, multichannel=True, full=True)

    # 2. compute the mean-squared error (mse) between two images.
    mse_score = skimage.measure.compare_mse(img_BGR, blur)

    return ssim_score, mse_score


def write_in_excel(image_id, noise_type, filter_name, score1, score2):
    sheet1.write(count_line, 0, image_id)
    sheet1.write(count_line, 1, noise_type)
    sheet1.write(count_line, 2, filter_name)
    sheet1.write(count_line, 3, score1)
    sheet1.write(count_line, 4, score2)


# creating excel
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')


# load images from folder "images to use"
photos = os.listdir("./images to use")

count_line = 0

for choose_image in photos:
    img_BGR = cv2.imread('./images to use/%s' % choose_image)

    # illustrate the results
    show_image("%s" % choose_image.split(".jpg")[0], img_BGR)

    # create folders if they don't exist
    Path("./question 2/%s/s&p" % choose_image.split(".jpg")[0]).mkdir(parents=True, exist_ok=True)
    Path("./question 2/%s/speckle" % choose_image.split(".jpg")[0]).mkdir(parents=True, exist_ok=True)

    '''
    salt and pepper
    '''
    # 1. create the new noisy image using salt and pepper
    prob = round(random.uniform(0, 0.4), 2)  # random float number between range 0 to 0.4 with 2 decimal places
    noise_img = sp_noise(img_BGR, prob)

    # and illustrate the results
    show_image("salt and pepper Image", noise_img)
    cv2.imwrite('./question 2/%s/s&p/s&p_noise.jpg' % choose_image.split(".jpg")[0], noise_img)  # save image

    print("\n%s" % choose_image.split(".jpg")[0])
    print("\\\\=====Salt and pepper noise=====\\\\")
    print("Compare original image with salt and pepper image")

    ssim_score, mse_score = calculation_of_scores(img_BGR, noise_img)

    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))



    # Filter application (a. Averaging filter, b. Gaussian Filtering, c. Median Filtering, d. Bilateral Filtering)
    avg_filter, gaussian_filter, median_filter, bilateral_filter = filter_application(noise_img)

    # a. illustrate the results of the Averaging filter
    show_image("Averaging Filter for s&p", avg_filter)
    cv2.imwrite('./question 2/%s/s&p/averaging_filter_s&p.jpg' % choose_image.split(".jpg")[0], avg_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, avg_filter)

    print("Compare original image with Averaging Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'salt and pepper', 'Averaging Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # b. illustrate the results of the Gaussian Filtering
    show_image("Gaussian Filter for s&p", gaussian_filter)
    cv2.imwrite('./question 2/%s/s&p/gaussian_filter_s&p.jpg' % choose_image.split(".jpg")[0], gaussian_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, gaussian_filter)

    print("Compare original image with Gaussian Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'salt and pepper', 'Gaussian Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # c. illustrate the results of the Median Filtering
    show_image("Median Filter for s&p", median_filter)
    cv2.imwrite('./question 2/%s/s&p/median_filter_s&p.jpg' % choose_image.split(".jpg")[0], median_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, median_filter)

    print("Compare original image with Median Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'salt and pepper', 'Median Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # d. illustrate the results of the Bilateral Filtering
    show_image("Bilateral Filter for s&p", bilateral_filter)
    cv2.imwrite('./question 2/%s/s&p/bilateral_filter_s&p.jpg' % choose_image.split(".jpg")[0], bilateral_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, bilateral_filter)

    print("Compare original image with Bilateral Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'salt and pepper', 'Bilateral Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1

    cv2.waitKey()
    cv2.destroyAllWindows()


    """
    speckle
    """
    # 2. create the new noisy image using speckle
    noise_img = speckle_noise(img_BGR)

    # show original image for comparison
    show_image("%s" % choose_image.split(".jpg")[0], img_BGR)


    # illustrate the results
    show_image("speckle Image", noise_img)
    cv2.imwrite('./question 2/%s/speckle/speckle_noise.jpg' % choose_image.split(".jpg")[0], noise_img)  # save image

    print("\n\\\\=====speckle noise=====\\\\")
    print("Compare original image with speckle image")

    ssim_score, mse_score = calculation_of_scores(img_BGR, noise_img)

    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))



    # Filter application (a. Averaging filter, b. Gaussian Filtering, c. Median Filtering, d. Bilateral Filtering)
    avg_filter, gaussian_filter, median_filter, bilateral_filter = filter_application(noise_img)


    # a. illustrate the results of the Averaging filter
    show_image("Averaging Filter for speckle", avg_filter)
    cv2.imwrite('./question 2/%s/speckle/avg_filter_speckle.jpg' % choose_image.split(".jpg")[0], avg_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, avg_filter)

    print("Compare original image with Averaging Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'speckle', 'Averaging Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # b. illustrate the results of the Gaussian Filtering
    show_image("Gaussian Filter for speckle", gaussian_filter)
    cv2.imwrite('./question 2/%s/speckle/gaussian_filter_speckle.jpg' % choose_image.split(".jpg")[0], gaussian_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, gaussian_filter)

    print("Compare original image with Gaussian Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'speckle', 'Gaussian Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # c. illustrate the results of the Median Filtering
    show_image("Median Filter for speckle", median_filter)
    cv2.imwrite('./question 2/%s/speckle/median_filter_speckle.jpg' % choose_image.split(".jpg")[0], median_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, median_filter)

    print("Compare original image with Median Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'speckle', 'Median Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1


    # d. illustrate the results of the Bilateral Filtering
    show_image("Bilateral Filter for speckle", bilateral_filter)
    cv2.imwrite('./question 2/%s/speckle/bilateral_filter_speckle.jpg' % choose_image.split(".jpg")[0], bilateral_filter)  # save image

    ssim_score, mse_score = calculation_of_scores(img_BGR, bilateral_filter)

    print("Compare original image with Bilateral Filter")
    print("\tSSIM score: {:.4f}".format(ssim_score))
    print("\tMSE score: {:.4f}".format(mse_score))

    write_in_excel(choose_image.split(".jpg")[0], 'speckle', 'Bilateral Filter', "{:.4f}".format(ssim_score),
                   "{:.4f}".format(mse_score))

    count_line = count_line + 1

    cv2.waitKey()
    cv2.destroyAllWindows()


wb.save('./question 2/results.xls')  # save excel
print("\n\'results.xls\' created")
