#!/usr/bin/env python

# In[4]:


import numpy as np
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# taken from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


cap = cv2.VideoCapture(0)
## choosing the frame size
# x = 640
# y = 480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

ret, image = cap.read()
size = image.shape
print('Frame size is:', size)


# 3D model points.
# this needs to be adjusted depending on the camera frame size
model_points = (np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])*0.7111111)
# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    size = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
        # ignoring the face if it's smaller then 1/3 of the frame
        if rect.height()/size[0] < 1/3:
            continue

        # determine the facial landmarks for the face region, then
        shape = predictor(gray, rect)
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_to_np(shape)
    
        image_points = np.array([
                                    (shape[33][0], shape[33][1]),     # Nose tip
                                    (shape[8][0], shape[8][1]),       # Chin
                                    (shape[36][0], shape[36][1]),     # Left eye left corner
                                    (shape[45][0], shape[45][1]),     # Right eye right corne
                                    (shape[48][0], shape[48][1]),     # Left Mouth corner
                                    (shape[54][0], shape[54][1])      # Right mouth corner
                                ], dtype="int")

    
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        # Good flags that worked for me
        # cv2.cv2.SOLVEPNP_ITERATIVE
        # cv2.CV_ITERATIVE
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points.astype('double'), camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        # Drawing dots on the landmarks and line on the direction
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255,0,0), 2)
        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    # Display image
    cv2.imshow("Output", image)
    # Break when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
