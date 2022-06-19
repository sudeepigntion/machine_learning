import cv2
import datetime

cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3, 1208)
cap.set(4, 720)

print(cap.get(3))
print(cap.get(4))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        text = 'Width :'+ str(cap.get(3)) + 'px, Height: '+str(cap.get(4))+' px'

        datet = str(datetime.datetime.now())

        print(datet)

        frame = cv2.putText(frame, datet, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()