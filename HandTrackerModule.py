import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils #draws points in hands
        self.tipIds = [4, 8, 12, 16, 20] #thumb , index , middle , ring , pinki

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #Processes the frame
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: #if hand is detected
            for handLms in self.results.multi_hand_landmarks: #landmarks of multi hands when detected
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                    self.mpHands.HAND_CONNECTIONS) #draws hands connection

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark): #working on points(finger) on each detected hand
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  #getting in lm pixels
                # print(id, cx, cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]: # -1 as thumb is big 4 and 3 are compared and [1] as when we close thumb we put it on side i.e x coordinate changes.
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]: #ex index finger. it checks the y coordinate of 8 and 6. if y of 8 is more then finger is up
                fingers.append(1) #returns true if finger is open ex when only index is up then list is [0,1,0,0,0]
            else:
                fingers.append(0) #returns false when not ex if only thumb is open then [1,0,0,0,0]

        return fingers

def main():

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()