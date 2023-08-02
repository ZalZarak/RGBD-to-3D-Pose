from roslibpy import Ros, Topic, Message
import struct
from main import visualize_points

pcl = 0


def callback(message):
    """print("haha")
    # Extract necessary fields from the message
    fields = message['fields']
    point_step = message['point_step']
    data = message['data'].encode()

    # Calculate the number of points in the point cloud
    num_points = len(data) // point_step

    # Extract x, y, z, and intensity values from the point cloud data
    points = []
    for i in range(num_points):
        offset = i * point_step
        x = struct.unpack_from('f', data, offset + fields[0]['offset'])[0]
        y = struct.unpack_from('f', data, offset + fields[1]['offset'])[0]
        z = struct.unpack_from('f', data, offset + fields[2]['offset'])[0]
        color = struct.unpack_from('f', data, offset + fields[3]['offset'])[0]
        points.append((x, y, z, color))
    print(points)"""

    global pcl
    pcl += 1
    print(pcl)
    if pcl == 5:
        visualize_points(points)



def listen():
    # Create a ROS instance
    ros = Ros("localhost", 9090)

    # Create a topic object for subscribing to the chatter topic
    topic = Topic(ros, "/camera/depth/color/points", "sensor_msgs/PointCloud2")

    # Set the callback function to be invoked when a message is received
    topic.subscribe(callback)

    # Start the ROS connection
    ros.run_forever()

if __name__ == "__main__":
    listen()
