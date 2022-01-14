#https://stackoverflow.com/questions/53021082/recognizing-the-bricks-in-a-brick-wall
import cv2
import random

im_path = "c://users//giovanni.milan//documents//crack_detection_CNN_masonry//dataset//crack_detection_224_images//a_22_4.png"

#thier example
im_path = "C://Users//Giovanni.Milan//Documents//crack_detection_CNN_masonry//dataset//size_rec//wall.jpg"


img = cv2.imread(im_path)
cv2.imshow("Image", img)
cv2.waitKey(0)
# To hsv
hsv =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Get the Saturation out
S=hsv[:,:,1]

# Threshold it
(ret,T)=cv2.threshold(S,42,255,cv2.THRESH_BINARY)

# Show intermediate result
cv2.imshow('win',T)
cv2.waitKey(0)

# Find contours
contours,h = cv2.findContours(T, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img2 = img.copy()


for c in contours:
    area = cv2.contourArea(c)
    # Only if the area is not miniscule (arbitrary)
    if area > 100:
        (x, y, w, h) = cv2.boundingRect(c)

        # Uncomment if you want to draw the conours
        #cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

        # Get random color for each brick
        tpl = tuple([random.randint(0, 255) for _ in range(3)])
        cv2.rectangle(img, (x, y), (x + w, y + h), tpl, -1)

cv2.imshow("bricks", img)
cv2.waitKey(0)