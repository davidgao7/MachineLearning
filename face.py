import cv2 #opencv library

# load pre-trained data on face frontals from oepncv (haar cascade algorithm: http://www.willberger.org/cascade-haar-explained/)
trainedFaceData = cv2.CascadeClassifier('faceData.xml')

print("data reading complete")

#choose input sources to detect face(webcam, image etc)
davidFace = cv2.imread('Tengjun_face.jpeg') #2d array

print('image reading complete')

# convert to image to grayscale
gray_davidFace = cv2.cvtColor(davidFace, cv2.COLOR_BGR2GRAY)

# detect faces
faceVector = trainedFaceData.detectMultiScale(gray_davidFace) #detect all faces/objects of different size input image (small/big)

print(faceVector) # bonding rectangle of the face(upper left of face & width & height of the rectangle) 

(x,y,w,h) = faceVector[0] #one vector, one person
print("getting one person...")
# draw rectangle of face
cv2.rectangle(davidFace, (x,y),(x+w,y+h), (0,255,0),10)# top left points<647,1028>  and bottom right point<950,950>; color choose of the rectangle<green>; thickness of the rectangle
print("drawing rectangle complete...")
cv2.imshow('face detector',davidFace)
cv2.waitKey(0)