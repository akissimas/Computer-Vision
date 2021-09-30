# required packages
import os
import cv2
import random
import numpy as np
import skimage.measure
from skimage.transform import radon
from keras.datasets import mnist
from xlwt import Workbook
from pathlib import Path


def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv2.imshow(title, image)
    cv2.resizeWindow(title, 480, 360)


def concat_tile(im_list_2d):
    """"
    A function that concatenates images of the same size with a 2D list (array)
    """
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def number_selection(X_train, Y_train, X_test, Y_test, number):
    """""
    Y_train and Y_test give you the labels of images. With numpy.where
    we can choose the numbers we want
    """""
    train_filter = np.where((Y_train == number))
    test_filter = np.where((Y_test == number))

    # using these filters to get the numbers we want
    X_train, Y_train = X_train[train_filter], Y_train[train_filter]
    X_test, Y_test = X_test[test_filter], Y_test[test_filter]


    # plot the sample of the number
    for i in range(5):
        image = X_train[random.randint(0, 5000)]   # using random to take random images of each number

        cv2.imwrite('./question 3/original images/{0}_{1}.png'.format(number, i), image)  # save image


def fourier_transform(images_to_use):
    counter = 0

    for image_selection in images_to_use:
        img_GRAY = cv2.imread('./question 3/original images/{}'.format(image_selection), 0)

        # now do the fourier stuff
        f = np.fft.fft2(img_GRAY)  # find Fourier Transform
        fshift = np.fft.fftshift(f)  # move zero frequency component (DC component) from top left to center

        # and calculate the magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        magnitude_spectrum_img = np.round(magnitude_spectrum).astype('uint8')

        cv2.imwrite('./question 3/fourier transform/{}'.format(image_selection), magnitude_spectrum_img)  # save image

        if not (counter % 5):
            show_image("%s" % image_selection.split(".png")[0], img_GRAY)
            show_image("Magnitude spectrum for {} image".format(image_selection.split(".png")[0]),
                       magnitude_spectrum_img)

        counter += 1


def calculate_similarity_matrices(images_to_use, type_of_image="original_images"):
    """
    :param images_to_use: list of images we have read from folder
    :param type_of_image: "original_images" -> calculate SIIM score between original images
                          "fourier_transform" -> calculate SIIM score between fourier transform images
    """

    if type_of_image == "fourier_transform":
        path = "./question 3/fourier transform"
    else:
        path = "./question 3/original images"

    # creating excel
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    # write the labels
    counter = 1
    for image_selection in images_to_use:
        sheet1.write(counter, 0, image_selection.split(".png")[0])
        sheet1.write(0, counter, image_selection.split(".png")[0])
        counter += 1

    row = 1
    for image_selection in images_to_use:
        img = cv2.imread('{}/{}'.format(path, image_selection) , 0)  # original image

        column = 1
        for image_selection_2 in images_to_use:
            img_2 = cv2.imread('{}/{}'.format(path, image_selection_2), 0)
            (score, _) = skimage.measure.compare_ssim(img, img_2, full=True)
            sheet1.write(row, column, "{:.4f}".format(score))
            print("{} - {} .. SSIM score: {:.4f}".format(image_selection, image_selection_2, score))

            column += 1

        row += 1

    wb.save('{} comparison.xls'.format(type_of_image))  # save excel
    print("\n\'{}.xls\' created".format(type_of_image))


# create folders if they don't exist
Path("./question 3/original images").mkdir(parents=True, exist_ok=True)
Path("./question 3/fourier transform").mkdir(parents=True, exist_ok=True)
Path("./question 3/radon transform").mkdir(parents=True, exist_ok=True)

list_of_numbers = [3, 5, 8, 9]

# load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# call function 'number_selection' for each number
for i in list_of_numbers:
    number_selection(X_train, Y_train, X_test, Y_test, i)


# load images
images = os.listdir("./question 3/original images")

list_of_all_images = []
counter = 1

for image_selection in images:
    img_BGR = cv2.imread("./question 3/original images/{}".format(image_selection))

    # illustrate the results
    show_image("{}".format(image_selection.split(".png")[0]), img_BGR)

    list_of_all_images.append(img_BGR)

    if not(counter % 5):
        # display each time the 5 pictures of each number
        cv2.waitKey()
        cv2.destroyAllWindows()

    counter = counter + 1

# combine all images to one
all_in_one = concat_tile([[list_of_all_images[0], list_of_all_images[1], list_of_all_images[2], list_of_all_images[3], list_of_all_images[4]],
                         [list_of_all_images[5], list_of_all_images[6], list_of_all_images[7], list_of_all_images[8], list_of_all_images[9]],
                         [list_of_all_images[10], list_of_all_images[11], list_of_all_images[12], list_of_all_images[13], list_of_all_images[14]],
                         [list_of_all_images[15], list_of_all_images[16], list_of_all_images[17], list_of_all_images[18], list_of_all_images[19]]])

cv2.imwrite('./question 3/all_in_one.jpg', all_in_one)

"""""
fourier transform
"""
fourier_transform(images)  # function call

cv2.waitKey()
cv2.destroyAllWindows()


""""
calculating similarity matrices
"""
input("\nPress Enter to calculate SIIM score between original images...")
calculate_similarity_matrices(images, "original_images")    # function call

input("\nPress Enter to calculate SIIM score between fourier transform images...")
calculate_similarity_matrices(images, "fourier_transform")  # function call


"""
radon Transform
"""
for image_selection in images:
    img_GRAY = cv2.imread("./question 3/original images/{}".format(image_selection), 0)
    theta = np.linspace(0., 180., max(img_GRAY.shape), endpoint=False)
    radon_img = radon(img_GRAY, theta=theta)

    normalized_img = np.zeros((800, 800))
    normalized_img = cv2.normalize(radon_img, normalized_img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    show_image("{}".format(image_selection.split(".png")[0]), normalized_img)

    cv2.imwrite('./question 3/radon transform/{}'.format(image_selection), normalized_img)  # save image

cv2.waitKey()
cv2.destroyAllWindows()


"""
calculate similarity matrix
"""
# load images
images = os.listdir("./question 3/radon transform")

# creating excel
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

# write the labels
counter = 1
for image_selection in images:
    sheet1.write(counter, 0, image_selection.split(".png")[0])
    sheet1.write(0, counter, image_selection.split(".png")[0])
    counter += 1

row = 1
for image_selection in images:
    img = cv2.imread('./question 3/radon transform/{}'.format(image_selection), 0)  # radon Transform images

    column = 1
    for image_selection_2 in images:
        img_2 = cv2.imread('./question 3/radon transform/{}'.format(image_selection_2), 0)
        (score, _) = skimage.measure.compare_ssim(img, img_2, full=True)
        sheet1.write(row, column, "{:.4f}".format(score))
        print("{} - {} .. SSIM score: {:.4f}".format(image_selection, image_selection_2, score))

        column += 1

    row += 1

wb.save('radon transform comparison.xls')  # save excel
print("\n\'radon transform comparison.xls\' created")
