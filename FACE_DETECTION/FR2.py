# Write a python cript that capture images from your wencam video stream
# Extract all faces from the images frame(using haarcascadedes)
# Stores the Faces information into numpy arrays

# To do list:
# 1.Read and show video stream,capture images
# 2.Detect Faces And Show bounding box(using haarcascade)
# 3.Flatten The largest face image And save in a numpy array
# 4.Repeat the above for multiple people to generate training data

import cv2
import numpy as np

#Init camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0

face_data = []
dataset_path = 'project1'

file_name = input("Enter the name of person : ")
while True:
	ret,frame = cap.read()

	if ret==False:
		continue
	gray_frame = cv2.cvtcolor(frame,cv2.COLOR_BGR2GRAY)	
	

	#cv2.imshow("Frame",frame)
	
	faces=face_cascade.detectMultiscale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	#print(faces)

	#pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces:
		x,y,w,h = face 
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255.255),2)

		#Extract (Crop out the required face ) : Region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

	

	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	'''#store the 10th face
	if(skip%10==0):
	    #store the 10th Face later on
	    pass
    '''
	key_pressed = cv2.waitkey(1) & 0xFF
	if key_pressed == ord('q'):
	    break

#Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data suceessfully save at "+dataset_path+file_name+'.npy')

cap.release()
cap.destroyAllWindows()









