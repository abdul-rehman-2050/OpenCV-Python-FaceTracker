#=======================================================
#   Licence @creative common 3.0
#
#   This code is provided with as is form. 
#
#   
#=======================================================



import numpy as np
import cv2
import sys



video_capture = cv2.VideoCapture(0)
#video_capture.set(3,320)
#video_capture.set(4,240)
face_cascade = cv2.CascadeClassifier('haar_frontalface_alt2.xml')
tracker = cv2.TrackerMIL_create()
isTracking = False


if __name__ == "__main__":
    try:
        while(True):
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frame = cv2.flip(frame,2)
            if ret==True:
                #-----------------------------------------------------------------------------
                if not isTracking:
                    print('not tracking')
                    #just to remove mirror effect in camera
                    
                    # Our operations on the frame come here
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50),
                        
                    )

                    # Draw a rectangle around the faces
                    for (x, y, w, h) in faces:
                        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=cv2.cv.CV_FILLED)
                        cv2.rectangle(frame, (x, y), (x+w+3, y+h+3), (0, 0, 0), 2)
                        face_gray_roi = gray[y:y+h, x:x+w]
                        # Initialize tracker with first frame and bounding box
                        tracker = cv2.TrackerMIL_create()
                        isTracking = tracker.init(frame, (x, y, w, h))
                        corners = cv2.goodFeaturesToTrack(face_gray_roi,500,0.01,10)
                        corners = np.int0(corners)
                        for i in corners:
                            x2,y2 = i.ravel()
                            cv2.circle(frame,(x+x2,y+y2),3,255,-1)
                        break

                   #----------------------------------------------------------------------
                else:
                    # Update tracker
                    #print('tracker')
                    isTracking, bbox = tracker.update(frame)
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                
                # Display the resulting frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    except KeyboardInterrupt as k:
        sys.stderr.write("program will exit\nBye!\n")
        
    #except Exception, e:
    #    sys.stderr.write(str(e) + "\n")


# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()        
sys.exit(0)
