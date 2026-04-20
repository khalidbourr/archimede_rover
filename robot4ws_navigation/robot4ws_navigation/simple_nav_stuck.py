# =============================
# ROS2 HUMBLE - SIMPLE NAV STACK
# Nodes:
# 1. MapHandlerNode
# 2. PlannerNode (A*)
# 3. PurePursuitController
# =============================

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from heapq import heappush, heappop
import math
import tf2_ros


# =============================
# UTILS
# =============================

def world_to_map(x, y, map_info):
    mx = int((x - map_info.origin.position.x) / map_info.resolution)
    my = int((y - map_info.origin.position.y) / map_info.resolution)
    return mx, my


def map_to_world(mx, my, map_info):
    x = mx * map_info.resolution + map_info.origin.position.x
    y = my * map_info.resolution + map_info.origin.position.y
    return x, y

def get_yaw(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

def transform_to_matrix(transform):
    t = transform.transform.translation
    q = transform.transform.rotation

    _, _, yaw = get_yaw(q)

    R = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])

    T = np.array([t.x, t.y])

    return R, T


def trim(val, max_val):
    max_val = abs(max_val)
    if abs(val) > max_val:
        return np.sign(val) * max_val
    else:
        return val


# =============================
# 1. MAP HANDLER NODE
# =============================

class MapHandlerNode(Node):
    def __init__(self):
        super().__init__('map_handler')

        self.declare_parameter('width', 500)
        self.declare_parameter('height', 500)
        self.declare_parameter('resolution', 0.2)

        w = self.get_parameter('width').value
        h = self.get_parameter('height').value
        res = self.get_parameter('resolution').value

        self.map = -1 * np.ones((h, w), dtype=np.int8)

        self.grid = OccupancyGrid()
        self.grid.info.resolution = res
        self.grid.info.width = w
        self.grid.info.height = h
        self.grid.info.origin.position.x = -w * res / 2.0
        self.grid.info.origin.position.y = -h * res / 2.0
        self.grid.header.frame_id = 'map'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(OccupancyGrid, 'submap', self.submap_callback, 10)

        self.pub = self.create_publisher(OccupancyGrid, 'global_map', 10)

        self.get_logger().info('Map handler node started')

    def submap_callback(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF not available: {e}")
            return

        R, T = transform_to_matrix(transform)

        sub = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

        h, w = sub.shape

        # grid indices
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))

        # convert to local metric coordinates using submap origin
        px = msg.info.origin.position.x + xs * msg.info.resolution
        py = msg.info.origin.position.y + ys * msg.info.resolution

        points = np.stack([px, py], axis=-1).reshape(-1, 2)

        # apply rotation + translation
        transformed = (R @ points.T).T + T

        mx, my = world_to_map(transformed[:, 0], transformed[:, 1], self.grid.info)

        # flatten submap
        values = sub.flatten()

        # valid indices
        valid = (
            (mx >= 0) & (mx < self.grid.info.width) &
            (my >= 0) & (my < self.grid.info.height) &
            (values != -1)
        )

        mx = mx[valid]
        my = my[valid]
        values = values[valid]

        # merge
        current_vals = self.map[my, mx]

        unknown_mask = current_vals == -1
        self.map[my[unknown_mask], mx[unknown_mask]] = values[unknown_mask]

        known_mask = ~unknown_mask
        self.map[my[known_mask], mx[known_mask]] = np.maximum(
            current_vals[known_mask], values[known_mask])

        self.publish()

    def publish(self):
        self.grid.data = self.map.flatten().tolist()
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.grid.header.frame_id = 'map'
        self.pub.publish(self.grid)


# =============================
# 2. PLANNER NODE (A*)
# =============================

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner')

        self.map = None
        self.map_info = None
        self.goal = None
        self.robot_pose = None

        self.create_subscription(OccupancyGrid, 'global_map', self.map_cb, 10)
        self.create_subscription(PoseStamped, 'goal', self.goal_cb, 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)

        self.pub = self.create_publisher(Path, 'path', 10)

        self.get_logger().info('Planner node started')

    def map_cb(self, msg):
        self.map = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.try_plan()

    def goal_cb(self, msg):
        self.goal = msg.pose
        self.try_plan()

    def odom_cb(self, msg):
        self.robot_pose = msg.pose.pose

    def try_plan(self):
        if self.map is None or self.goal is None or self.robot_pose is None:
            return

        start = world_to_map(self.robot_pose.position.x,
                             self.robot_pose.position.y,
                             self.map_info)

        goal = world_to_map(self.goal.position.x,
                            self.goal.position.y,
                            self.map_info)

        path = self.a_star(start, goal)
        if path:
            self.publish_path(path)

    def a_star(self, start, goal):
        h, w = self.map.shape

        open_set = []
        heappush(open_set, (0, start))

        came_from = {}
        g = {start: 0}

        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                return self.reconstruct(came_from, current)

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = current[0]+dx, current[1]+dy

                if not (0 <= nx < w and 0 <= ny < h):
                    continue

                cell = self.map[ny, nx]
                if cell == 100:
                    continue
                elif cell == 0:
                    cost = 1
                else: 
                    cost = 5

                tentative = g[current] + cost

                if (nx, ny) not in g or tentative < g[(nx, ny)]:
                    g[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'

        for mx, my in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            x, y = map_to_world(mx, my, self.map_info)
            pose.pose.position.x = x
            pose.pose.position.y = y
            msg.poses.append(pose)

        self.pub.publish(msg)


# =============================
# 3. CONTROLLER (simplified PURE PURSUIT)
# =============================

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')

        # TODO: set this parameter better, not great to have it here like this
        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                True
            )
        ])

        self.path = None
        self.pose = None

        self.lookahead_distance_min = 0.8
        self.lookahead_distance_max = 1.6
        self.current_idx = 0
        self.target_tolerance = 0.2

        self.speed = 0.2
        self.max_ang_speed = 0.2

        self.last_time = self.get_clock().now().nanoseconds * 1e-9

        self.create_subscription(Path, 'path', self.path_cb, 1)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 1)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Node started')

    def path_cb(self, msg):
        self.path = msg.poses
        self.current_idx = 0
        self.get_logger().info('New path received')

    def odom_cb(self, msg):
        self.pose = msg.pose.pose

    def control_loop(self):
        if not self.path or self.pose is None:
            return

        # check time paused
        time_now = self.get_clock().now().nanoseconds * 1e-9
        if time_now - self.last_time < 1e-6:
            return
        self.last_time = time_now

        rx = self.pose.position.x
        ry = self.pose.position.y

        # Waypoint progress
        while self.current_idx <= len(self.path) - 1:
            p = self.path[self.current_idx].pose.position
            if math.hypot(p.x - rx, p.y - ry) < self.target_tolerance:
                self.current_idx += 1
            else:
                break

        if self.current_idx == len(self.path):
            self.get_logger().info('Goal reached')
            self.pub.publish(Twist())
            # TODO: shutdown node, or wait without stamping every time
            return

        target = self.path[self.current_idx].pose.position

        # Lookahead
        accumulated = math.hypot(target.x - rx, target.y - ry)
        if accumulated < self.lookahead_distance_max:
            for i in range(self.current_idx, len(self.path) - 1):
                p1 = self.path[i].pose.position
                p2 = self.path[i + 1].pose.position

                segment = math.hypot(p2.x - p1.x, p2.y - p1.y)
                accumulated += segment

                if accumulated >= self.lookahead_distance_min:
                    if accumulated <= self.lookahead_distance_max:
                        self.current_idx = i+1
                        target = self.path[self.current_idx].pose.position
                    break

        # control
        angle_to_target = math.atan2(target.y - ry, target.x - rx)

        yaw = get_yaw(self.pose.orientation)
        angle_error = angle_to_target - yaw

        # TODO: angular vel might be scaled to linear one
        cmd = Twist()
        cmd.linear.x = self.speed
        cmd.angular.z = trim(angle_error / 5, self.max_ang_speed)

        self.pub.publish(cmd)


# =============================
# MAIN ENTRY
# =============================

def main(args=None):
    rclpy.init(args=args)

    map_node = MapHandlerNode()
    planner_node = PlannerNode()
    controller_node = ControllerNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(map_node)
    executor.add_node(planner_node)
    executor.add_node(controller_node)

    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
