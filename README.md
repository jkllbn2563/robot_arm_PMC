# robot_arm_PMC


###start object_detection server
"""
python server.py

"""
###start object_detection client
"""
python client.py

"""
###give objection a trigger
"""
rosservice call /object_detection_willie "{}"

"""
###model link(ICERA)
"""
https://drive.google.com/open?id=1ZOw4uix6qr7-z34TTkWSDHz1s2ZHiIcv

mkdir ssd_mobilenet_v1_coco_2017_11_17
mv output_inference_graph_v7.pb ssd_mobilenet_v1_coco_2017_11_17

"""


