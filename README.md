# robot_arm_PMC

### start object_detection server

python server.py



### start object_detection client

python client.py


### give objection a trigger

rosservice call /object_detection_willie "{}"



### model link(ICERA)


https://drive.google.com/open?id=1ZOw4uix6qr7-z34TTkWSDHz1s2ZHiIcv

cd ssd_mobilenet_v1_coco_2017_11_17
tar zxvf output_inference_graph_v7.pb
vim server.py

PATH_TO_FROZEN_GRAPG = ......"change here"


