

import cv2
from matplotlib import pyplot as plt
import glob







max_val = 8
max_pt = -1
max_kp = 0


obj =  cv2.xfeatures2d.SURF_create()
# obj is an alternative to Surf


test_img = cv2.imread('test/test_5i.jpg')




# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = obj.detectAndCompute(test_img, None)

#make a list of refernce images

training_set = glob.glob('files/*.jpg')



for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = obj.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.78
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.78 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected denomination: Rs. ', note)

	

	plt.imshow(img3), plt.show()


else:
	print('No Matches')
	
	

