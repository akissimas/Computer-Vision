import cv2
import random
import numpy as np
from pathlib import Path


# create folders if they don't exist
Path("./question 1").mkdir(parents=True, exist_ok=True)
Path("./question 1/A").mkdir(parents=True, exist_ok=True)
Path("./question 1/B").mkdir(parents=True, exist_ok=True)

# create noisy image that increases by 5% every time
noise_values = [0, 0.05, 0.1, 0.15, 0.2]

for noise_percentage in noise_values:
    # load the image to check
    img_rgb = cv2.imread('motherboard.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    noisy_image = img_gray

    # create noise random noise
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            noisy_image[i][j] = img_gray[i][j] + noise_percentage * random.gauss(0, 1) * img_gray[i][j] - noise_percentage * random.gauss(0, 1) * img_gray[i][j]


    cv2.imwrite('./question 1/A/noise {}%.jpg'.format(int(100 * noise_percentage)), noisy_image)
    cv2.imshow('{}% noise'.format(int(100 * noise_percentage)), noisy_image)


    # load the template image we look for
    template = cv2.imread('motherboard_template.jpg', 0)

    w, h = template.shape[::-1]

    # run the template matching
    res = cv2.matchTemplate(noisy_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where(res >= threshold)


    # mark the corresponding location(s)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    cv2.imwrite('./question 1/A/detected {}%.jpg'.format(int(100 * noise_percentage)), img_rgb)
    cv2.imshow('Detected', img_rgb)


    cv2.waitKey()
    cv2.destroyAllWindows()

    # b. Gaussian Filtering
    filtered_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    cv2.imwrite('./question 1/B/filtered_image {}%.jpg'.format(int(100 * noise_percentage)), filtered_image)
    cv2.imshow('{}% filtered_image'.format(int(100 * noise_percentage)), filtered_image)

    # load the template image we look for
    template = cv2.imread('motherboard_template.jpg', 0)

    w, h = template.shape[::-1]

    # run the template matching
    res = cv2.matchTemplate(filtered_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    loc = np.where(res >= threshold)

    # mark the corresponding location(s)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    cv2.imwrite('./question 1/B/detected {}%.jpg'.format(int(100 * noise_percentage)), img_rgb)
    cv2.imshow('Detected', img_rgb)

    cv2.waitKey()
    cv2.destroyAllWindows()
