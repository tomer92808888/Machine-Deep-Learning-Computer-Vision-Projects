# Face Recgnition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# Defining a function that will do the detections
def detect(gray, frame): # Take as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0) , 2) # Paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # Get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # Apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # Paint a rectangle around the eyes, but inside the referential of the face.
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # Apply the detectMultiScale method to locate one or several smiles(wide mouthes) in the image.
        for (sx, sy, sw, sh) in smiles: # For each detected smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2) # Paint a rectangle around the smile, but inside the referential of the face.
    return frame # Return the image with the detector rectangles.

video_capture = cv2.VideoCapture(1) # Turn the webcam on.

while True: # Repeat infinitely (Until break):
    _, frame = video_capture.read() # Get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Colour transformations.
    canvas = detect(gray, frame) # Get the output of our detect function.
    cv2.imshow('Video', canvas) # Display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If the person press the 'q' key on the keyboard:
        break # Stop the loop.
    
    
video_capture.release() # Turn the webcam off.
cv2.destroyAllWindows() # Destroy all the windows inside which the images were displayed.
