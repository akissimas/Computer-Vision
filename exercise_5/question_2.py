from __future__ import print_function
import sys
import cv2
from math import hypot
from xlwt import Workbook
from copy import deepcopy
from random import randint
from time import process_time

trackerTypes = ['BOOSTING', 'MIL', 'KCF']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# create excel
wb = Workbook()
# create sheet
sheet = wb.add_sheet("results", cell_overwrite_ok=True)
# initialize columns
column_list = ["technique name", "Frame number", "processing time", "centroid change box1", "centroid change box2"]

# save in excel
for i in range(len(column_list)):
    sheet.write(0, i, column_list[i])

# Create a video capture object to read videos
cap = cv2.VideoCapture("oldman_walking.mp4/")
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)


# Select boxes
bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:  # q is pressed
        break

excel_line_cnt = 1
for tracker_type in trackerTypes:
    oldCentroids = []
    newCentroids = []
    frameCount = 0

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(tracker_type), frame, bbox)
        # find the centroid of the bbox
        oldCentroids.append((bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2)))

    while cap.isOpened():
        frameCount += 1
        success, frame = cap.read()
        start = process_time()
        if not success:
            break


        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            # find the new centroid of the bbox
            newCentroids.append((newbox[0] + (newbox[2] / 2), newbox[1] + (newbox[3] / 2)))

        # find the translocation of centroid
        translocation = []
        for i in range(len(oldCentroids)):
            translocation.append(hypot(newCentroids[i][0] - oldCentroids[i][0], newCentroids[i][1] - oldCentroids[i][1]))

        oldCentroids = deepcopy(newCentroids)
        newCentroids = []

        # show frame
        cv2.imshow(tracker_type, frame)
        timeTaken = process_time() - start


        # save in excel
        sheet.write(excel_line_cnt, 0, tracker_type)
        sheet.write(excel_line_cnt, 1, frameCount)
        sheet.write(excel_line_cnt, 2, timeTaken)
        sheet.write(excel_line_cnt, 3, translocation[0])
        sheet.write(excel_line_cnt, 4, translocation[1])
        excel_line_cnt += 1
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    # Create a video capture object to read videos
    cap = cv2.VideoCapture("oldman_walking.mp4/")
    # Read first frame
    success, frame = cap.read()





cap.release()
cv2.destroyAllWindows()
# close excel
wb.save("results.xls")
