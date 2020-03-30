import cv2
import numpy as np
import sys
import os
import PIL

# import mxnet as mx 

import insightface



def draw_results(img, faces):

    or_image = img.copy()

    color_landmark = (0,255,255)
    color_gender = 255
    size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    face_boxes = []

    for idx, face in enumerate(faces):
        if len(face.bbox) > 0:
            x,y,w,h = face.bbox.astype(np.int)
            
            age = face.age

            gender = 'Male'
            if face.gender==0:
                gender = 'Female'

            landmarks = face.landmark.astype(np.int)

            p1, p2, p3, p4, p5 = landmarks[0], landmarks[1], \
                                landmarks[2], landmarks[3], \
                                landmarks[4] 

            cv2.putText(img, str(gender) , (x-35, y+5), font, 0.5,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(img, str(age) , (x-35, y+20), font, 0.5,(0,0,255),1,cv2.LINE_AA)

            cv2.circle(img, tuple(p1), 1, color_landmark)
            cv2.circle(img, tuple(p2), 1, color_landmark)
            cv2.circle(img, tuple(p3), 1, color_landmark)
            cv2.circle(img, tuple(p4), 1, color_landmark)
            cv2.circle(img, tuple(p5), 1, color_landmark)

            cv2.rectangle(img, (x,y), (w,h),(255,255,0),2)

            my_face_box = or_image[y:h,x:w]

            face_boxes.append(my_face_box)

    return img, face_boxes

def main():

    model = insightface.app.FaceAnalysis()
    ctx_id = 0  # -1 if use CPU

    model.prepare(ctx_id=ctx_id,  nms=0.4)
    # model.prepare(ctx_id = ctx_id, nms=0.4)

    cap = cv2.VideoCapture(0)


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict

        faces = model.get(frame)

        img, face_boxes = draw_results(frame, faces)

        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
