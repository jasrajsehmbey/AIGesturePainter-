import cv2
import numpy as np
import HandTrackerModule as htm
import pytesseract

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)
detector = htm.handDetector(maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
     #Import image
     success, img = cap.read()
     img = cv2.flip(img, 1)

     #Find Hand Landmarks
     img = detector.findHands(img)
     lmList = detector.findPosition(img, draw=False) #finds position of points of fingers

     if len(lmList) != 0:   #when hand is detected
          # tip of index and middle fingers
          x1, y1 = lmList[8][1:] #id 8 is index finger [1:] as we want x and y value which is 1 and 2 in list
          x2, y2 = lmList[12][1:] #if 12 is middle finger

          #Check which fingers are up
          fingers = detector.fingersUp()
          if fingers[1] and fingers[2] and fingers[3] == False and fingers[4] == False: # pause mode
               xp, yp = 0, 0
          if fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4] == False:  #Drawing Mode - if only Index finger is up
               cv2.circle(img, (x1, y1), 15, (255,255,0), cv2.FILLED)
               if xp == 0 and yp == 0:
                    xp, yp = x1, y1

               cv2.line(img, (xp, yp), (x1, y1), (255,255,0), 8)  # draws line
               cv2.line(imgCanvas, (xp, yp), (x1, y1), (255, 255, 0), 8)
               xp, yp = x1, y1

          # Clear Canvas when 4 fingers are up
          if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
             imgCanvas = np.zeros((480, 640, 3), np.uint8)

     imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
     _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
     _, imgInv2 = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
     imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
     if len(lmList)!=0:
          fingers = detector.fingersUp()
          if fingers[1] and fingers[2] and fingers[3] and fingers[4]==False: #detects when 3 fingers are up
               pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
               imgchar = pytesseract.image_to_string(imgInv2, lang='eng')
               if imgchar:
                  print(imgchar)
     #cv2.imshow("Inv", imgInv2)
     img = cv2.bitwise_and(img, imgInv)
     img = cv2.bitwise_or(img, imgCanvas)
     cv2.imshow("Image", img)

     if cv2.waitKey(1) & 0xFF == ord('q'):  # waitkey(1) is frame rate 1 fpms, when pressed q program exits
          break
cap.release()
cv2.destroyAllWindows()