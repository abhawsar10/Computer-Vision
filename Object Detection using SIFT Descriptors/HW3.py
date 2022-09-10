import numpy as np
import cv2
import matplotlib.pyplot as plt


def hw3():



	img_obj=[]
	img=[]
	
	img_obj.append(cv2.imread("HW3_Data/src_0.jpg"))	#load object images
	img_obj.append(cv2.imread("HW3_Data/src_1.jpg"))
	img_obj.append(cv2.imread("HW3_Data/src_2.jpg"))

	img.append(cv2.imread("HW3_Data/dst_0.jpg"))	#load target images
	img.append(cv2.imread("HW3_Data/dst_1.jpg"))
	

	#----------------------------------DETECTING SIFT FEATURES-------------------------------------------------

	print("-----------------------------------------------------------------------")

	sift = cv2.SIFT_create()	#create object of SIFT detector

	k1=[]		#variables for storing keypoints of object images
	d1=[]		#variable for storing descriptors of object images
	for i in range(len(img_obj)):

		k,d= sift.detectAndCompute(img_obj[i], None)	#detect keypoints in image
		k1.append(k)
		d1.append(d)

		gray = cv2.cvtColor(img_obj[i], cv2.COLOR_BGR2GRAY)		#create grayscale version of image
		sift_image = cv2.drawKeypoints(gray, k1[i], img_obj[i],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)	#draw keypoints (in color) on image with direction and magnitude of features
		cv2.imwrite("Report/SIFT_src_"+str(i)+".jpg", sift_image)		#save image
		print("Features found in image src_"+str(i)+".jpg= ",len(k))	#print number of feature points found



	k2=[]		#variable for storing keypoints of target images
	d2=[]		#variable for storing descriptors of target images
	for i in range(len(img)):

		k,d= sift.detectAndCompute(img[i], None)	#detect keypoints in image
		k2.append(k)
		d2.append(d)

		gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)		#create grayscale version of image
		sift_image = cv2.drawKeypoints(gray, k2[i], img[i],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)	#draw keypoints (in color) on image with direction and magnitude of features
		cv2.imwrite("Report/SIFT_dst_"+str(i)+".jpg", sift_image)		#save image
		print("Features found in image dst_"+str(i)+".jpg= ",len(k))	#print number of feature points found

	print()

	#----------------------------------FIND MATCHES IN SIFT FEATURES-------------------------------------------------

	bf = cv2.BFMatcher()	#create object of brute force matcher to find matches between keypoints of two images

	#for i in range(1):
	for i in range(len(img_obj)):	#for every object image
		#for j in range(1):
		for j in range(len(img)):	#for every target image
			
			print("------------------------------Target Image: dst_"+str(j)+".jpg | Object Image: src_"+str(i)+".jpg------------------------------")
			
			img_obj[i] = cv2.imread("HW3_Data/src_"+str(i)+".jpg")		#load images again so previous matches are not shown again.
			img[j] = cv2.imread("HW3_Data/dst_"+str(j)+".jpg")
			

			matches=bf.knnMatch(d1[i],d2[j],k=2)	#find top 2 matches in target image, for every descriptor in object image


			good = []								#store good matches
			for m,n in matches:

			    if m.distance < 0.8*n.distance:		#perform ratio test, save good matches
			        good.append([m])		

			#print("Total Matches found =",len(good))		#print number of good matches
			
			good = sorted(good, key=lambda x:x[0].distance)

			img_w_matches = cv2.drawMatchesKnn(img_obj[i],k1[i],img[j],k2[j],good[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)	
			#show all good matches between object image descriptors and  target image descriptors

			cv2.imwrite("Report/img"+str(j)+"_obj"+str(i)+" Top20Matches.jpg", img_w_matches)	#save image



			#----------------------------------FIND HOMOGRAPHY MATRIX-------------------------------------------------


			if len(good)>10:	#run RANSAC only if enough matches are found

				src_pts = np.float32([ k1[i][m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)		#find locations of matched keypoints in object images
				dst_pts = np.float32([ k2[j][m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)		#find locations of matched keypoints in target images

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)		#find homography matrix based on RANSAC method


				"""				#to draw boundary boxs
				h,w,d = img_obj[i].shape
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)	#find corner points of object image so that we can plot object in target image
				dst = cv2.perspectiveTransform(pts,M)	#find locations of corner points in target image
				imgx = cv2.polylines(img[j],[np.int32(dst)],True,(255,255,255),7, cv2.LINE_AA)	#draw white box showing object in target image
				"""


				print("Inliner matches found = ",np.count_nonzero(mask == 1))		#print number of inliner matches

				draw_params = dict(matchColor = (0,255,0), 
						singlePointColor = None,
					matchesMask = mask, 
						flags = 2)		#draw lines between correct matches in green

				img_w_obj = cv2.drawMatchesKnn(img_obj[i],k1[i],img[j],k2[j],good,None,**draw_params) #draw lines between matches
				cv2.imwrite("Report/img"+str(j)+"_obj"+str(i)+" InlinerMatches.jpg", img_w_obj)	#Save image

				print("Homography Matrix=")
				print(M)
				print()

				#----------------------------------FIND TOP 10 MATCHES-------------------------------------------------			


				pred_pts = cv2.perspectiveTransform(src_pts,M)	#find projections of all good points

				diff = []
				for k in range(len(pred_pts)):		#for all predicted points
					diff.append([k, ( (pred_pts[k][0][0] - dst_pts[k][0][0])**2 + (pred_pts[k][0][1] - dst_pts[k][0][1])**2 )**(1/2)])
					#store L2 distances between projected points and the actual destination points in diff array

				diff = np.array(diff)
				diff = diff[np.argsort(diff[:,1])]		#sort diff array by the distances
				
				mask = np.zeros((pred_pts.shape[0], 1))	#create new mask
				for k in range(10):				
					mask[ int(diff[k][0]) ][0] = 1 		#set mask of top 10 low error points to 1 
						
				draw_params = dict(matchColor = (255,255,0), 
						singlePointColor = None,
					matchesMask = mask, 		#display only top 10 low error matchings
						flags = 2)
				
				img_t10 = cv2.drawMatchesKnn(img_obj[i],k1[i],img[j],k2[j],good,None,**draw_params) #draw lines between matches
				cv2.imwrite("Report/img"+str(j)+"_obj"+str(i)+" Top10Predictions.jpg", img_t10)	#Save image


			else:
				
				print( "Not enough matches are found - {}/{}".format(len(good), 10) )


	print("-----------------------------------------------------------------------")

		

hw3()