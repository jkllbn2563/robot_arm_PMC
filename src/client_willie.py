#!/usr/bin/env python
import rospy
from robot_arm.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_arm.msg import coordinate
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point



if __name__=='__main__':


	rospy.init_node('client_willie_node',anonymous=True)
	
	
	
	detection_client_willie=rospy.ServiceProxy('object_detection_willie',ObjDetectionWithName)

	rospy.wait_for_service('object_detection_willie')
	rospy.loginfo('Ready to caculate the bbox')


    

	object_name=raw_input("please key in : ")

	detection_result = detection_client_willie(object_name)
	
	
	print("the object is at: ",detection_result.bbox_corner1,detection_result.bbox_corner2)
	print ('type(detection_result.input_pc): ',type(detection_result.input_pc))

	
	pub = rospy.Publisher('test_pc2', PointCloud2, queue_size=1)
	#pub.publish(detection_result.input_pc)
	#rate = rospy.Rate(1)
	
	#while not rospy.is_shutdown():
	#	pub.publish(detection_result.input_pc)
	
	