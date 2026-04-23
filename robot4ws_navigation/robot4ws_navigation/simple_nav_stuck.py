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
from std_msgs.msg import String
from heapq import heappush, heappop
import math
import json


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
        self.grid = None
        self.create_subscription(String, 'global_map_update', self._map_cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, 'global_map', 10)
        self.get_logger().info('Map handler node started (waiting for first globalMap)')

    def _map_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"global_map_update: invalid JSON: {e}")
            return

        cols = int(payload['cols'])
        rows = int(payload['rows'])
        res = float(payload['cellSize'])
        ox = float(payload.get('originX', 0.0))
        oy = float(payload.get('originY', 0.0))

        # Rebuild from scratch — tuple space is the authoritative state
        grid_map = -1 * np.ones((rows, cols), dtype=np.int8)
        for c in payload.get('cells', []):
            gx, gy = c['gx'], c['gy']
            if 0 <= gx < cols and 0 <= gy < rows:
                grid_map[gy, gx] = 0 if c['traversable'] else 100

        self.grid = OccupancyGrid()
        self.grid.info.resolution = res
        self.grid.info.width = cols
        self.grid.info.height = rows
        self.grid.info.origin.position.x = ox
        self.grid.info.origin.position.y = oy
        self.grid.header.frame_id = 'map'
        self.grid.data = grid_map.flatten().tolist()
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.grid)

        known = int(((grid_map == 0) | (grid_map == 100)).sum())
        self.get_logger().info(f'Map updated: {cols}x{rows}, {known} known cells')


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
        self.get_logger().info(f'map received: {msg.info.width}x{msg.info.height}, {sum(1 for c in msg.data if c != -1)} known cells')
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

        h, w = self.map.shape
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            self.get_logger().warn(
                f'start {start} outside grid {w}x{h} — skipping plan')
            return
        if not (0 <= goal[0] < w and 0 <= goal[1] < h):
            self.get_logger().warn(
                f'goal {goal} outside grid {w}x{h} — skipping plan')
            return

        path = self.a_star(start, goal)
        if path:
            self.publish_path(path)
        else:
            self.get_logger().warn('A* found no path')

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

        self.speed = 0.5
        self.max_ang_speed = 0.6

        self.last_time = self.get_clock().now().nanoseconds * 1e-9

        self.create_subscription(Path, 'path', self.path_cb, 1)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 1)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.status_pub = self.create_publisher(String, 'nav_status', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.goal_reached_published = False
        self._last_status = None

        self.get_logger().info('Node started')

    def _set_status(self, status):
        if self._last_status != status:
            self.status_pub.publish(String(data=status))
            self._last_status = status

    def path_cb(self, msg):
        self.path = msg.poses
        self.current_idx = 0
        self.goal_reached_published = False
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
            self.pub.publish(Twist())
            if not self.goal_reached_published:
                self._set_status('goal_reached')
                self.goal_reached_published = True
                self.get_logger().info('Goal reached')
            return

        self._set_status('navigating')

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
        angle_error = math.atan2(math.sin(angle_to_target - yaw),
                                 math.cos(angle_to_target - yaw))

        # TODO: angular vel might be scaled to linear one
        cmd = Twist()
        cmd.linear.x = self.speed * max(0.0, math.cos(angle_error))
        cmd.angular.z = trim(angle_error, self.max_ang_speed)

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
