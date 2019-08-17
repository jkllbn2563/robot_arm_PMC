#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image2
import time
import numpy as np

img = Image2()
import matplotlib.pyplot as plt
from PIL import Image
i=0

def rgb_callback(image):
	for i in range(20):
		bridge = CvBridge()
		time.sleep(5)

		print("bird")
		#object_name=req.obj_name
		image_np = bridge.imgmsg_to_cv2(image, "rgb8")
		image_np=Image.fromarray(image_np,"RGB")
		plt.imshow(image_np)
		plt.axis('off')
		plt.savefig('train_'+str(i)+".png")
		print(i)
		i=i+1
		print("apple",i)
if __name__=='__main__':
	rospy.init_node('camera_node',anonymous=True)
	print("banana")
	#rospy.Subscriber("/c1/camera/rgb/image_raw",Image,rgb_callback)
	rospy.Subscriber("/camera/rgb/image_raw",Image2,rgb_callback)
	#time.sleep(50)
	print("apple")

	rate=rospy.Rate(1)
	rospy.spin()