import cv2
import sys

cap = cv2.VideoCapture('people_walking.mp4/')

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video")
    sys.exit()

fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

# set kernel for morphology
Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()  # remember it's BGR format!
    if not ret:
        break

    # convert frame to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # additional bluring to handle real life noise
    blur_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)

    fgmask = fgbg.apply(blur_frame)

    # open morphology to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, Kernel, iterations=2)
    # close morphology to close small holes inside the foreground objects
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, Kernel)

    # find contours
    _, contours, Hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw rectangle
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # threshold
        if cv2.contourArea(contour) < 230:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show foreground and tracking video
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
