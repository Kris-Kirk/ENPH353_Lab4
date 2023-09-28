#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 1
		self._cam_fps = 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

		self.feature_match_constant = 0.6
		self.num_Good_Matches = 10 # Number of good matches to be found for homography

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		homography = []

		ret, frame = self._camera_device.read() #Frame is the full screen
		#TODO run SIFT on the captured frame
		#Create a SIFT Object
		sift = cv2.xfeatures2d.SIFT_create()

		#Load the image that is selected
		selected_Image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)

		#Detect the key points and compute the descriptors for the query image
		kp_image, desc_image = sift.detectAndCompute(selected_Image, None)
		
		#Convert the frame to gray scale
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Train image

		#KP = Key Points, Desc = Descriptors
		kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
		
		#Draw the key points on the original frame
		# annotated_image = cv2.drawKeypoints(frame, kp_grayframe, frame) 

		#Draw keypoints on gray frame
		# grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)
		
		# Feature Matching
		flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5),dict())

		#Draw matches between the selected photo and the image taken from the webcam
		matches = flann.knnMatch(desc_image, desc_grayframe, k = 2) # Need to get descriptors of the chosen image
		
		#Select only good points
		good_points = []
		for m,n in matches:
			if m.distance < self.feature_match_constant*n.distance:
				good_points.append(m)
		
		# Find the matches with the good points and draw them

		# Assuming selected_Image and frame are your two images
		# Assuming kp_image and kp_grayframe are the keypoints

		# Calculate the width of the output image
		output_width = selected_Image.shape[1] + frame.shape[1]

		# Create a blank output image with the required width and height
		output_image = np.zeros((max(selected_Image.shape[0], frame.shape[0]), output_width, 3), dtype=np.uint8)

		# Draw the first image on the output image
		output_image[:selected_Image.shape[0], :selected_Image.shape[1]] = cv2.cvtColor(selected_Image, cv2.COLOR_GRAY2BGR)

		# Draw the second image next to the first image
		output_image[:frame.shape[0], selected_Image.shape[1]:] = frame

		output_image = cv2.drawMatches(selected_Image, kp_image, frame, kp_grayframe, good_points, frame) #Changed grayframe to frame (Last argument)
		# frame = cv2.drawMatches(selected_Image, kp_image, grayframe, kp_grayframe, good_points, frame, flags=2) #Changed grayframe to frame (Last argument)

		# Now Homography to highlight the template in the frame
		# queryIDX gives the index of the matched points in the query image
		# Reshape makes it an array of arrays
		if len(good_points) > self.num_Good_Matches:
			query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
			train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

			matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
			#Need to extract mask points to a list
			matches_mask = mask.ravel().tolist()

			#Perspective transform
			height, width = selected_Image.shape
			pts = np.float32([[0,0], [0,height], [width,height], [width,0]]).reshape(-1,1,2)
			#Consider perspective in the new image
			dst = cv2.perspectiveTransform(pts, matrix)

			#Draw the lines on the detected image
			CloseLines = True
			LineColor = (255,0,0)
			LineThickness = 3
			#Add the homography outline to the frame
			homography  = cv2.polylines(frame, [np.int32(dst)], CloseLines, LineColor, LineThickness)

		if len(homography) != 0:
			pixmap = self.convert_cv_to_pixmap(homography)
		else:			
			pixmap = self.convert_cv_to_pixmap(output_image)
		self.live_image_label.setPixmap(pixmap)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
