import cv2
import random
import numpy as np
from pathlib import Path
from skimage.color import label2rgb
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans


def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    cv2.imshow(title, image)
    cv2.resizeWindow(title, 480, 360)
    cv2.waitKey()


def convert_to_binary(image):
    # initialize image
    new_img = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)

    blue = 0
    red = 0
    # check the background color
    for height in range(50):
        for width in range(50):
            if image[height][width][0] == 255:  # blue background
                blue += 1
            elif image[height][width][2] == 255:    # red background
                red += 1
    if blue > red:
        background_color = 0
    else:
        background_color = 2

    # convert the background color to black and the colour of our interest to white
    if background_color == 0:
        for height in range(image.shape[0]):
            for width in range(image.shape[1]):
                if image[height][width][0] == 255 and image[height][width][1] == 0 and image[height][width][2] == 0:
                    new_img[height][width][0] = 0
                else:
                    new_img[height][width][0] = 255
                    new_img[height][width][1] = 255
                    new_img[height][width][2] = 255

    elif background_color == 2:
        for height in range(image.shape[0]):
            for width in range(image.shape[1]):
                if image[height][width][2] == 255 and image[height][width][1] == 0 and image[height][width][0] == 0:
                    new_img[height][width][2] = 0
                else:
                    new_img[height][width][0] = 255
                    new_img[height][width][1] = 255
                    new_img[height][width][2] = 255

    return new_img


# create folders if they don't exist
Path("./question 3").mkdir(parents=True, exist_ok=True)


# Loading original image in BGR
originImg = cv2.imread("bird.jpg")

# Loading binary annotated image
true_img = cv2.imread("binary_annotated_bird.jpg")


# create noisy image that increases by 5% every time
noise_values = [0, 0.05, 0.1, 0.15, 0.2]

for noise_percentage in noise_values:
    noisy_image = np.zeros([originImg.shape[0], originImg.shape[1], 3], dtype=np.uint8)

    # create noise random noise
    for i in range(originImg.shape[0]):
        for j in range(originImg.shape[1]):
            noisy_image[i][j] = originImg[i][j] + noise_percentage * random.gauss(0, 1) * originImg[i][j] - noise_percentage * random.gauss(0, 1) * originImg[i][j]

    # display the original image
    cv2.imwrite('./question 3/Original image (noise {}%).jpg'.format(int(100 * noise_percentage)), noisy_image)
    show_image("Original image (noise: {}%)".format(noise_percentage*100), noisy_image)

    # Shape of original image
    originShape = noisy_image.shape

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities (or the 3 channels that I currently have)
    flatImg = np.reshape(noisy_image, [-1, 3])



    """
    Meanshift
    """
    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Performing meanshift on flatImg
    print('Using MeanShift algorithm, it takes time...')
    ms.fit(flatImg)
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels = ms.labels_

    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image
    ms_segmentedImg = np.reshape(labels, originShape[:2])
    ms_segmentedImg = label2rgb(ms_segmentedImg) * 255  # need this to work with cv2. imshow
    ms_segmentedImg = ms_segmentedImg.astype(np.uint8)

    show_image("MeanShiftSegments(noise: {}%)".format(noise_percentage*100), ms_segmentedImg)
    cv2.imwrite('./question 3/MeanShiftSegments(noise {}%).jpg'.format(int(100 * noise_percentage)), ms_segmentedImg)


    """
    Kmeans
    """
    # now go for the kmeans
    print('Using kmeans algorithm...')
    km = MiniBatchKMeans(n_clusters=2)
    km.fit(flatImg)
    labels = km.labels_

    # Displaying segmented image
    km_segmentedImg = np.reshape(labels, originShape[:2])
    km_segmentedImg = label2rgb(km_segmentedImg) * 255  # need this to work with cv2.imshow
    km_segmentedImg = km_segmentedImg.astype(np.uint8)

    show_image("kmeans Segments(noise: {}%)".format(noise_percentage*100), km_segmentedImg)
    cv2.imwrite('./question 3/kmeans Segments(noise {}%).jpg'.format(int(100 * noise_percentage)), km_segmentedImg)


    """
    convert segmentImg to binary annotated image
    """
    # meanshift
    ms_binary_annotated_img = convert_to_binary(ms_segmentedImg)

    show_image("meanshift binary annotated image(noise: {}%)".format(noise_percentage*100), ms_binary_annotated_img)
    cv2.imwrite('./question 3/meanshift binary annotated image(noise {}%).jpg'.format(int(100 * noise_percentage)), ms_binary_annotated_img)


    # Kmeans
    km_binary_annotated_img = convert_to_binary(km_segmentedImg)

    show_image("Kmeans binary annotated image(noise: {}%)".format(noise_percentage*100), km_binary_annotated_img)
    cv2.imwrite('./question 3/Kmeans binary annotated image(noise {}%).jpg'.format(int(100 * noise_percentage)), km_binary_annotated_img)


    """
    calculate score
    """
    true_img = true_img.reshape(-1)

    #meanshift
    pred_img = ms_binary_annotated_img.reshape(-1)
    print("\nmeanshift(noise: {}%)".format(noise_percentage*100))
    print("\tf1 score(macro): {:.4f}".format(f1_score(true_img, pred_img, average="macro", labels=np.unique(pred_img))))
    print("\tAccuracy: {:.4f}\n".format(accuracy_score(true_img, pred_img)))


    #kmeans
    pred_img = km_binary_annotated_img.reshape(-1)
    print("Kmeans(noise: {}%)".format(noise_percentage*100))
    print("\tf1 score(macro): {:.4f}".format(f1_score(true_img, pred_img, average="macro", labels=np.unique(pred_img))))
    print("\tAccuracy score: {:.4f}\n".format(accuracy_score(true_img, pred_img)))



    cv2.waitKey()
    cv2.destroyAllWindows()

