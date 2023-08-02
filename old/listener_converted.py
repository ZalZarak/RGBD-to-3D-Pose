from roslibpy import Ros, Topic, Message
import struct
from main import visualize_points

pcl = 0

def callback(message):
    """data_flat = message["data"]
    points = []
    for i in range(0, len(data_flat), 3):
        points.append([data_flat[i], data_flat[i+1], data_flat[i+2]])"""

    global pcl
    pcl += 1
    print(pcl)
    #if pcl == 5:
        #visualize_points(points)



def listen():
    # Create a ROS instance
    ros = Ros("localhost", 9090)

    # Create a topic object for subscribing to the chatter topic
    topic = Topic(ros, "/pointcloud2_converted", "std_msgs/Float64MultiArray")

    # Set the callback function to be invoked when a message is received
    topic.subscribe(callback)

    # Start the ROS connection
    ros.run_forever()

if __name__ == "__main__":
    listen()
