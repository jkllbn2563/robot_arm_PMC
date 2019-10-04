#!/usr/bin/env python
import rospy
from robot_arm_PMC.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_arm_PMC.msg import coordinate
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
import time
img = Image()
point_cloud = PointCloud2()
point1 = Point()
point2 = Point()
task_complete=0
task_complete_score=0
receive_first_image=False
receive_first_point_cloud=False
read_point_cloud_prohibit = False
import math
#object_name = ''
counter=0
bbox_scale_list=[]
#pub = rospy.Publisher('test_pc2', PointCloud2, queue_size=1)
def distance(p1,p2):
	length = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

	print(length)
	return length

def rgb_callback(image):
	global img,receive_first_image

	img=image
	#detection_client_willie()
	receive_first_image=True

def point_cloud_callback(image):
	global point_cloud, receive_first_point_cloud, read_point_cloud_prohibit
	
	#print (image)
	#print ('point_cloud_callback')
	if read_point_cloud_prohibit == False:
		point_cloud=image
		receive_first_point_cloud = True


def handle_function_willie(req):

	global point_cloud, object_name,img,counter,task_complete,task_complete_score
	bridge = CvBridge()
	bbox_scale=[]
	#object_name=req.obj_name
	image_np1 = bridge.imgmsg_to_cv2(img, "rgb8")
	from PIL import Image 
	width,high= Image.fromarray(image_np1, 'RGB').size
	#width,high=image_.size
	#print("image size is",high,width)
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
	if bbox_scale == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
		print("Not detect anything")
		counter=0
	if bbox_scale != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
		#print(bbox_scale)
		print("detect object!!")
		print(counter)
	if counter==4:
		counter=0
	if len(bbox_scale_list)<2:
		bbox_scale_list.append(bbox_scale)

	if (len(bbox_scale_list)==2) & (bbox_scale != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  :
		
		print("compare",bbox_scale_list[0],bbox_scale_list[1])
		if bbox_scale_list[0][0:4]!=[0.0, 0.0, 0.0, 0.0]:
			print(bbox_scale_list[0][0:4])
			print("camera")
			print("center_camera",(bbox_scale_list[0][0]+bbox_scale_list[0][2])/2,(bbox_scale_list[0][1]+bbox_scale_list[0][3])/2,(bbox_scale_list[1][0]+bbox_scale_list[1][2])/2,(bbox_scale_list[1][1]+bbox_scale_list[1][3])/2)
			camera_distance=distance([bbox_scale_list[0][0]+bbox_scale_list[0][2]/2,(bbox_scale_list[0][1]+bbox_scale_list[0][3])/2],[(bbox_scale_list[1][0]+bbox_scale_list[1][2])/2,(bbox_scale_list[1][1]+bbox_scale_list[1][3])/2])
			print(camera_distance)
			if camera_distance<120:
				task_complete_score+=10
		if bbox_scale_list[0][4:8] !=[0.0, 0.0, 0.0, 0.0]:
			print("tsripod")
			print("tripod_center",(bbox_scale_list[0][4]+bbox_scale_list[0][6])/2,(bbox_scale_list[0][5]+bbox_scale_list[0][7])/2,(bbox_scale_list[1][4]+bbox_scale_list[1][6])/2,(bbox_scale_list[1][5]+bbox_scale_list[1][7])/2)
			tripod_distance=distance([bbox_scale_list[0][4]+bbox_scale_list[0][6]/2,(bbox_scale_list[0][5]+bbox_scale_list[0][7])/2],[(bbox_scale_list[1][4]+bbox_scale_list[1][6])/2,(bbox_scale_list[1][5]+bbox_scale_list[1][7])/2])
			if tripod_distance<120:
				task_complete_score+=10

		if bbox_scale_list[0][8:12] !=[0.0, 0.0, 0.0, 0.0]:
			print("USB")
			print("USB_center",(bbox_scale_list[0][8]+bbox_scale_list[0][10])/2,(bbox_scale_list[0][9]+bbox_scale_list[0][11])/2,(bbox_scale_list[1][8]+bbox_scale_list[1][10])/2,(bbox_scale_list[1][9]+bbox_scale_list[1][11])/2)
			USB_distance=distance([bbox_scale_list[0][8]+bbox_scale_list[0][10]/2,(bbox_scale_list[0][9]+bbox_scale_list[0][11])/2],[(bbox_scale_list[1][8]+bbox_scale_list[1][10])/2,(bbox_scale_list[1][9]+bbox_scale_list[1][11])/2])
			if USB_distance<120:
				task_complete_score+=10
		#print(bbox_scale_list[0][12:16])

		if bbox_scale_list[0][12:16] !=[0.0, 0.0, 0.0, 0.0]:
			print("box")
			print(bbox_scale_list[0][12:16])
			
			print("box_center",(bbox_scale_list[0][12]+bbox_scale_list[0][14])/2,(bbox_scale_list[0][13]+bbox_scale_list[0][15])/2,(bbox_scale_list[1][12]+bbox_scale_list[1][14])/2,(bbox_scale_list[1][13]+bbox_scale_list[1][15])/2)
			box_distance=distance([bbox_scale_list[0][12]+bbox_scale_list[0][14]/2,(bbox_scale_list[0][13]+bbox_scale_list[0][15])/2],[(bbox_scale_list[1][12]+bbox_scale_list[1][14])/2,(bbox_scale_list[1][13]+bbox_scale_list[1][15])/2])
			if box_distance<120:
				task_complete_score+=10
			

		
		print(task_complete_score)		
		if task_complete_score>9:
			task_complete=1
			task_complete_score=0
			counter+=1
		else:
			task_complete_score=0

			print("It is not sure for static")
		bbox_scale_list.pop(0)

	if (counter==3)& (task_complete==1):
		print("the object is static !!")
		counter=0
		task_complete=0
		

		print("scaling: ",bbox_scale)
		return detection_PMC_halfResponse(bbox_scale)
	else:
		return detection_PMC_halfResponse([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])






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
	#rospy.Subscriber("/c1/camera/rgb/image_raw",Image,rgb_callback)
	rospy.Subscriber("/camera/rgb/image_raw",Image,rgb_callback,queue_size=1,buff_size=2**26)
	#rospy.Subscriber("/c1/camera/depth/points",PointCloud2,point_cloud_callback)
	time.sleep(5)
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
	
"""
	#while not rospy.is_shutdown():
	while not (receive_first_image and receive_first_point_cloud):
		print ("doesn't receive the first image or the first point cloud")
		if rospy.is_shutdown():
			exit(-1)
"""


	#rospy.spin()



