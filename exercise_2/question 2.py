import cv2
from pathlib import Path

# create folders if they don't exist
Path("./question 2").mkdir(parents=True, exist_ok=True)

# load the image to check (2 times for cv2.rectangle())
img_rgb = cv2.imread('motherboard.jpg')
img_rgb_2 = cv2.imread('motherboard.jpg')

# load the template image we look for
template = cv2.imread('motherboard_template.jpg')

Height = img_rgb.shape[0]
Width = img_rgb.shape[1]

height = template.shape[0]
width = template.shape[1]

# 1. convert the original images to HSV format
hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

# 2. calculate histogram of template
template_hist = cv2.calcHist([hsv_template], [0], None, [256], [0, 256])

for i in range(int(width/2), Width - int(width/2)):
    for j in range(int(height/2), Height - int(height/2)):
        image_patch = img_rgb[j - int(height/2):j + int(height/2), i - int(width/2):i + int(height/2)]

        # 1. convert the image_patch to HSV format
        hsv_image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)

        # 2. calculate histogram of image_patch
        image_patch_hist = cv2.calcHist([hsv_image_patch], [0], None, [256], [0, 256])

        # compare the 2 histograms with Correlation metric
        correlation_result = cv2.compareHist(template_hist, image_patch_hist, cv2.HISTCMP_CORREL)

        # bounding boxes
        if correlation_result >= 0.815:
            correlation_image = cv2.rectangle(img_rgb, (i-int(width/2), j-int(height/2)), (i+int(width/2), j+int(height/2)), (255, 0, 0), 2)


        # compare the 2 histograms with Intersection metric
        intersection_result = cv2.compareHist(template_hist, image_patch_hist, cv2.HISTCMP_INTERSECT)

        # bounding boxes
        if intersection_result >= 237:
            intersection_image = cv2.rectangle(img_rgb_2, (i-int(width/2), j-int(height/2)), (i+int(width/2), j+int(height/2)), (255, 0, 0), 2)



cv2.imshow("Correlation", correlation_image)
cv2.imshow("Intersection", intersection_image)

cv2.imwrite('./question 2/{}.jpg'.format("Correlation"), correlation_image)
cv2.imwrite('./question 2/{}.jpg'.format("Intersection"), intersection_image)

cv2.waitKey()
cv2.destroyAllWindows()
