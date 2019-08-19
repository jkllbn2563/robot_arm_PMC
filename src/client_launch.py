#!/usr/bin/env python
import rospy
from robot_arm_PMC.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
import time
img = Image()
point_cloud = PointCloud2()
point1 = Point()
point2 = Point()

receive_first_image=False
receive_first_point_cloud=False
read_point_cloud_prohibit = False

def rgb_callback(image):
	global img,receive_first_image

	img=image
	receive_first_image=True

def handle_function_willie(req):
	global point_cloud, object_name,img
	bridge = CvBridge()
	bbox_scale=[]
	image_np1 = bridge.imgmsg_to_cv2(img, "rgb8")
	from PIL import Image
	width,high= Image.fromarray(image_np1, 'RGB').size
	#width,high=image_.size
	print("image size is",high,width)
	bbox_data=bbox_calculation()
	#point1.y = bbox_data.data[0]*high
	#point1.x = bbox_data.data[1]*width
	#point2.y = bbox_data.data[2]*high
	#point2.x = bbox_data.data[3]*width
	#print (point1,point2)
	for i,v in enumerate(bbox_data.data):
		if i % 2 == 0:
			v=v*high
		else:
			v=v*width
		bbox_scale.append(v)
	read_point_cloud_prohibit = False
	#return point_cloud, point1, point2
	print("scaling: ",bbox_scale)
	return detection_PMC_halfResponse(bbox_scale)

def bbox_calculation():
	global img, read_point_cloud_prohibit,object_name
	try:
		bridge = CvBridge()

		read_point_cloud_prohibit = True
		bbox=detection_client(img)
		#print(bbox.data)
		if bbox.data==(0.0,0.0,0.0,0.0):
			print("There is no %s on the table"%object_name)
		return bbox

	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)

if __name__=='__main__':
	rospy.init_node('client_node',anonymous=True)
	rospy.Subscriber("/scorpio/mmp0/camera/color/image_raw",Image,rgb_callback)
	#rospy.Subscriber("/camera/color/image_raw",Image,rgb_callback)

	#time.sleep(5)
	rospy.wait_for_service('object_detection')
	detection_client=rospy.ServiceProxy('object_detection',detection_PMC)

	#s=rospy.Service('object_detection_willie',ObjDetectionWithName,handle_function_willie)
	s=rospy.Service('object_detection_willie',detection_PMC_half,handle_function_willie)
	detection_client_willie=rospy.ServiceProxy('object_detection_willie',detection_PMC_half)
	rospy.wait_for_service('object_detection_willie')
	#print("the object is at: ",detection_result.data)


	rospy.loginfo('Ready to caculate the bbox')

	print ('waiting service: object_detection_willie')
	#rospy.wait_for_service('/haf_grasping/PointCloudWithBBox')
	#grasp_srv_client=rospy.ServiceProxy('/haf_grasping/PointCloudWithBBox',PointCloudWithBBox)


	rate=rospy.Rate(1)
	rospy.spin()