#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image2
import time
import numpy as np

img = Image2()
import matplotlib.pyplot as plt
from PIL import Image
from std_srvs.srv import Trigger, TriggerResponse
picture = None
i = 0
def took(req):
		global picture, i
		bridge = CvBridge()
		print("bird")
		#object_name=req.obj_name
		image_np = bridge.imgmsg_to_cv2(picture, "rgb8")
		image_np=Image.fromarray(image_np,"RGB")
		plt.imshow(image_np)
		plt.axis('off')
		plt.show()
		plt.savefig('train_'+str(i)+".png")
		i=i+1
		return TriggerResponse(
		success=True,
		message="picture "+str(i)+" is tooken!"
		)
def rgb_callback(image):
		global picture
		picture=image



if __name__=='__main__':
	rospy.init_node('camera_node',anonymous=True)
	print("banana")
	#sub = rospy.Subscriber("/scorpio/mmp0/camera/color/image_raw",Image2,rgb_callback)
	sub = rospy.Subscriber("/camera/color/image_raw",Image2,rgb_callback)
	rospy.Service('/photo',Trigger, took)
	#time.sleep(50)
	print("apple")

	rospy.spin()