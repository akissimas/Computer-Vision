import cv2
import sys
import time

# change the frame size
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Set up tracker.
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

'''
using Tracker
'''
# Read video
video = cv2.VideoCapture("man_walking.mp4/")


# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# rescale frame
frame = rescale_frame(frame, percent=65)

# bounding box for man walking
bbox = (26, 354, 42, 113)

total_time = 0
frame_counter = 0

start_time = time.time()
ok = tracker.init(frame, bbox)  # Initialize tracker with first frame and bounding box
total_time += (time.time() - start_time)

frame_counter += 1
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # rescale frame
    frame = rescale_frame(frame, percent=65)

    # timer for fps display
    fps_timer = cv2.getTickCount()

    start_time = time.time()    # timer to calculate average time
    ok, bbox = tracker.update(frame)    # Update tracker
    total_time += (time.time() - start_time)

    frame_counter += 1  # counter to calculate average time


    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\n==={} Tracker===".format(tracker_type))
average_time = total_time / frame_counter
print("\tAverage Time: {:.3}".format(average_time))

'''
using HAAR cascade classifier
'''
# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Read video
video = cv2.VideoCapture("man_walking.mp4/")

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame2 = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# rescale frame
frame2 = rescale_frame(frame2, percent=65)

total_time = 0
frame_counter = 0
while True:
    # Read a new frame
    ok, frame2 = video.read()
    if not ok:
        break

    # rescale frame
    frame2 = rescale_frame(frame2, percent=65)

    # timer for fps display
    fps_timer = cv2.getTickCount()

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    start_time = time.time()    # timer to calculate average time
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)  # Pass frame to our body classifier
    total_time += (time.time() - start_time)

    frame_counter += 1  # counter to calculate average time

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_timer)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame2, "HAAR cascade classifier", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame2, "FPS : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow('HAAR', frame2)


    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break


print("\n===HAAR cascade classifier===")
average_time = total_time / frame_counter
print("\tAverage Time: {:.3}".format(average_time))

video.release()
cv2.destroyAllWindows()
