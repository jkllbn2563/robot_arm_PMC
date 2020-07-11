#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image2
import time
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse
img = Image2()
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
i=1




def take(req):
	global img,i
	bridge = CvBridge()
	#time.sleep(5)
	#rate.sleep()
		

	#object_name=req.obj_name
	image_np = bridge.imgmsg_to_cv2(img, "rgb8")
	image_np=Image.fromarray(image_np,"RGB")
	plt.imshow(image_np)
	plt.axis('off')
	plt.savefig('train_'+str(i)+".png")
	print(i)
	#i=i+1
	print("picture",i)
	i=i+1
	#print("fuck")

	return TriggerResponse(success=True,message="the camera picture"+str(i-1)+" is save" )

def rgb_callback(image):
		global i,img
		img=image

if __name__=='__main__':
	rospy.init_node('camera_node',anonymous=True)
	#rospy.Subscriber("/c1/camera/rgb/image_raw",Image,rgb_callback)
	rospy.Subscriber("/camera/color/image_raw",Image2,rgb_callback,queue_size=1,buff_size=2**26)
	#time.sleep(50)
	rospy.Service('/photo',Trigger,take)
	print("waiting for trigger")
	

	#rate=rospy.Rate(0.5)
	rospy.spin()
