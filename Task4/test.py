import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка!")
else:
    print("Камера работает!")
    cap.release()