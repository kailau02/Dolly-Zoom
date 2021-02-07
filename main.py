import numpy as np
from Frame import *

##############################################################################################
#       SETTINGS        # What worked best for me:
##############################################################################################
ZOOM = 0.25             # Medium = 0.2 to 0.3,    Close = 0.35 to 0.5
SHOW_BOX = False         # Show face detection box around the largest detected face
SCALE_FACTOR = 1.22     # Medium = 1.2,     Close = 1.14
MIN_NEIGHBORS = 8       # 8
MINSIZE = (60, 60)    # Medium = (60, 60),    Close = (120, 120)
##############################################################################################
##############################################################################################

CASC_PATH = "haarcascade_frontalface_default.xml"

# Create cascade
faceCascade = cv2.CascadeClassifier(CASC_PATH)
# Capture from camera
cap = cv2.VideoCapture(0)

# Create global detection box for steady screen transformation
box = BoundingBox(-1, -1, -1, -1)


print("Press \'ESC\' to exit")
while True:
    # Read image
    _, img = cap.read(0)

    # Approximate phone camera resolution testing
    # widOff = int(img.shape[1] / 3.0)
    # img = img[0:img.shape[0], widOff:img.shape[1] - widOff]

    # Grayscale image for facial box detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MINSIZE,
    )
    boxes = np.array(boxes)

    # Linear interpolate bounding box to dimensions of largest detected box
    if boxes.size > 0:
        boxLrg = largestBox(boxes)
        if box.dim[0] == -1:
            box = boxLrg
        else:
            box.lerpShape(boxLrg)

    # Setup frame properties and perform filter
    frame = Frame(img, box)
    frame.boxIsVisible = SHOW_BOX
    frame.setZoom(ZOOM)
    frame.filter()
    box = frame.box

    # Display filtered image as an OpenCV window
    frame.show()

    # Key press events
    k = cv2.waitKey(30)

    if k == 27:     # Escape to exit
        break
    if k == 49:     # '1' to toggle box
        SHOW_BOX = not SHOW_BOX
    if k == 50:     # '2' to decrease zoom
        ZOOM = max(ZOOM - 0.05, 0.01)
        print(ZOOM)
    if k == 51:     # '3' to increase zoom
        ZOOM = min(ZOOM + 0.05, 0.99)
        print(ZOOM)

# Release the VideoCapture object
cap.release()
