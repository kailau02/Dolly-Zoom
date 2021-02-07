import cv2


def lerp(a, b, c):
    return int((c * a) + ((1 - c) * b))


def largestBox(boxes):
    lrg_width = 0
    lrg_box = None
    for box in boxes:
        if box[2] > lrg_width:
            lrg_box = BoundingBox(box[0], box[1], box[2], box[3])
            lrg_width = box[2]
    return lrg_box


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.dim = [x, y, w, h]

    def lerpShape(self, newBox):
        for i in range(2):
            self.dim[i] = lerp(self.dim[i], newBox.dim[i], 0.4)
        for i in range(2):
            j = i + 2
            self.dim[j] = lerp(self.dim[j], newBox.dim[j], 0.7)


class Frame:
    boxIsVisible = False

    def __init__(self, img, box):
        self.zoom = 0.4
        self.img = img
        self.box = box
        x, y, w, h = box.dim
        self.postFilterBox = BoundingBox(x, y, w, h)

    def setZoom(self, amount):
        self.zoom = min(max(amount, 0.01), 0.99)

    def filter(self):

        # Declare basic variables
        screenHeight = self.img.shape[0]
        screenWidth = self.img.shape[1]
        screenRatio = float(screenWidth) / screenHeight

        (boxX, boxY, boxW, boxH) = self.box.dim
        distX1 = boxX
        distY1 = boxY                               # dist refers to the distances in front of and
        distX2 = screenWidth - distX1 - boxW        # behind the face detection box
        distY2 = screenHeight - distY1 - boxH       # EX: |---distX1----[ :) ]--distX2--|

        # Equalize x's and y's to shortest length
        if distX1 > distX2:
            distX1 = distX2
        if distY1 > distY2:
            distY1 = distY2

        distX = distX1      # Set to an equal distance value
        distY = distY1

        # Trim sides to match original aspect ratio
        centerX = distX + (boxW / 2.0)
        centerY = distY + (boxH / 2.0)
        distsRatio = centerX / centerY

        if screenRatio < distsRatio:
            offset = centerX - (centerY * screenRatio)
            distX -= offset
        elif screenRatio > distsRatio:
            offset = centerY - (centerX / screenRatio)
            distY -= offset

        # Make screen to box ratio constant
        # (constant can be changed as ZOOM in main.py)
        if screenWidth > screenHeight:
            distX = min(0.5 * ((boxW / self.zoom) - boxW), distX)
            distY = min(((1.0 / screenRatio) * (distX + (boxW / 2.0))) - (boxH / 2.0), distY)
        else:
            distY = min(0.5 * ((boxH / self.zoom) - boxH), distY)
            distX = min((screenRatio * (distY + (boxH / 2.0))) - (boxW / 2.0), distX)

        # Crop image to match distance values
        newX = int(boxX - distX)
        newY = int(boxY - distY)
        newW = int(2 * distX + boxW)
        newH = int(2 * distY + boxH)
        self.crop([newX, newY, newW, newH])

        # Resize image to fit original resolution
        resizePercentage = float(screenWidth) / newW
        self.img = cv2.resize(self.img, (screenWidth, screenHeight))
        for i in range(4):
            self.postFilterBox.dim[i] = int(self.postFilterBox.dim[i] * resizePercentage)

        # Flip Filtered image on y-axis
        self.img = cv2.flip(self.img, 2)

    def drawBox(self):
        (x, y, w, h) = self.postFilterBox.dim
        if x > 0:
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def crop(self, dim):
        x, y, w, h = dim
        self.img = self.img[y:y + h, x:x + w]
        self.postFilterBox.dim[0] -= x
        self.postFilterBox.dim[1] -= y

    def show(self):
        if self.boxIsVisible:
            self.drawBox()
        cv2.imshow("Dolly Zoom", self.img)
