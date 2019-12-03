#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
from robot_arm_PMC.srv import *
from cv_bridge import CvBridge,CvBridgeError

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as Image2
from utils import label_map_util

from utils import visualization_utils as vis_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from sensor_msgs.msg import Image
from robot_arm_PMC.msg import coordinate
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
import time
import server
import client
from client import rgb_callback
from client import point_cloud_callback
from client import handle_function_willie



def server_srv():
    rospy.init_node('detection_server',anonymous=True)
    s=rospy.Service('object_detection',detection_PMC,handle_function)
    rospy.loginfo('Ready to caculate the bbox')
    rospy.spin()


if rospy.is_shutdown():
    exit(-1)

print("fuck")
rospy.init_node('client_node',anonymous=True)
print("test_for_c")
rospy.Subscriber("/camera/rgb/image_raw",Image,client.rgb_callback)
rospy.Subscriber("/c1/camera/depth/points",PointCloud2,point_cloud_callback)
time.sleep(5)
rospy.wait_for_service('object_detection')
print("test_for_c-")
detection_client=rospy.ServiceProxy('object_detection',detection_PMC)
print("test_for_c--")


#s=rospy.Service('object_detection_willie',ObjDetectionWithName,handle_function_willie)
s=rospy.Service('object_detection_willie',detection_PMC_half,handle_function_willie)
detection_client_willie=rospy.ServiceProxy('object_detection_willie',detection_PMC_half)
print("test_for_c+++")
rospy.wait_for_service('object_detection_willie')
#print("the object is at: ",detection_result.data)
print("test_for_c+++++++")


rospy.loginfo('Ready to caculate the bbox')

print ('waiting service: object_detection_willie')
print("test_for_c--------")
#rospy.wait_for_service('/haf_grasping/PointCloudWithBBox')
#grasp_srv_client=rospy.ServiceProxy('/haf_grasping/PointCloudWithBBox',PointCloudWithBBox)


rate=rospy.Rate(1)
rospy.spin()
