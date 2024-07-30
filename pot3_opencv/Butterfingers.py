import cv2
import mediapipe as mp
import pyautogui
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands



cap=cv2.VideoCapture(0)

with mp_hands.Hands( static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,) as hands:
    
    while True:
        ret,frame=cap.read()
        a=0
        if ret == False:
            break
        h,w=frame.shape[:2]
        frame=cv2.flip(frame,1)
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)
        disd=0
        disu=0
        disl=0
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*w)
                y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*h)
                x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w)
                y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h)
                x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*w)
                y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*h)
                x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x*w)
                y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y*h)
                disd=((x2-x1)**2+(y2-y1)**2)**(1/2)
                disu=((x3-x1)**2+(y3-y1)**2)**(1/2)
                disl=((x4-x1)**2+(y4-y1)**2)**(1/2)
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
        if disd>0 and disd<30:
            pyautogui.scroll(-100)
        elif disu>0 and disu<30:
            pyautogui.scroll(100)
        elif disl>0 and disl<30:
            pyautogui.doubleClick()
        cv2.putText(frame,f"Distance:{disd:.2f}",(20,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(frame,f"Distance:{disu:.2f}",(40,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(frame,f"Distance:{disl:.2f}",(60,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1)& 0xFF==27:
            break
cap.release()
cv2.destroyAllWindows()