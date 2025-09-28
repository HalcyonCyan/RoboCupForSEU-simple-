#!/usr/bin/env python3

import rospy
import actionlib
import math
import time
import random
import os
import cv2
from collections import deque
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

class WaypointNavigator:
    def __init__(self):
        rospy.init_node('waypoint_navigator', anonymous=True)
        
        # 定义地点到坐标的映射
        self.location_mapping = {
            "厨房": (-1.1, 1.3, 1.57),
            "餐厅": (-0.9, 1.1, 0.00),
            "客厅": (1.25, -0.3, 1.57),
            "卧室": (3.68, 1.13, 0.00),
            "书房": (3.65, -1.7, 0.00)
        }
        
        # 定义固定起点和终点
        self.start_point = (-3.5, -2.8, 1.57)
        self.end_point = (-3.5, -2.65, 1.57)
        
        # 从文件读取导航顺序
        self.middle_waypoints = self.read_waypoints_from_file()
        
        # 如果文件读取失败或不足5个点，使用随机顺序
        if len(self.middle_waypoints) < 5:
            rospy.logwarn("从文件读取的航点不足5个，使用随机顺序")
            self.middle_waypoints = [
                (-1.1, 1.3, 1.57),
                (-0.9, 1.1, 0.00),
                (1.25, -0.3, 1.57),
                (3.68, 1.13, 0.00),
                (3.65, -1.7, 0.00)   
            ]
            random.shuffle(self.middle_waypoints)
            self.waypoint_source = "random"
        else:
            self.waypoint_source = "file"
        
        # 组合完整的航点列表：起点 + 中间点 + 终点
        self.waypoints = [self.start_point] + self.middle_waypoints + [self.end_point]
        
        self.loop = rospy.get_param('~loop', False)  # 默认只跑一圈
        
        # 显示航点顺序
        self.display_waypoint_order()
        
        # 计时相关变量
        self.start_time = None
        self.end_time = None
        self.navigation_started = False
        self.circle_completed = False
        
        # 卡住检测相关变量
        self.position_history = deque(maxlen=20)  # 保存最近20个位置
        self.stuck_threshold_distance = 0.1  # 5秒内移动距离小于10cm认为卡住
        self.stuck_threshold_time = 5.0  # 5秒检测周期
        self.last_stuck_check_time = None
        self.stuck_count = 0  # 连续卡住次数
        self.last_global_plan = None  # 保存全局路径
        
        # TF2相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 图像相关
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.latest_image = None
        self.image_received = False
        
        # 激光雷达数据
        self.laser_data = None
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        # 全局路径订阅
        self.plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.plan_callback)
        
        # 创建速度发布器用于小幅度旋转调整和恢复行为
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 创建结果目录
        self.results_dir = "/home/halcyon/catkin_ws/src/match_nav/results"
        self.create_results_directory()
        
        # 等待必要的服务
        rospy.sleep(2.0)
        
        # 连接到move_base动作服务器
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        if not self.client.wait_for_server(rospy.Duration(30.0)):
            rospy.logerr("Could not connect to move_base action server")
            return
        
        rospy.loginfo("Connected to move_base action server")
        
        # 等待系统初始化
        rospy.sleep(5.0)
        
        # 检查TF树
        self.check_tf_tree()
        
        # 开始导航
        self.navigate_waypoints()
    
    def laser_callback(self, msg):
        """激光雷达回调函数"""
        self.laser_data = msg
    
    def plan_callback(self, msg):
        """全局路径回调函数"""
        self.last_global_plan = msg
    
    def create_results_directory(self):
        """创建结果目录"""
        try:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
                rospy.loginfo("创建结果目录: %s", self.results_dir)
            else:
                rospy.loginfo("结果目录已存在: %s", self.results_dir)
        except Exception as e:
            rospy.logerr("创建结果目录失败: %s", str(e))
    
    def image_callback(self, msg):
        """图像回调函数"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", str(e))
    
    def take_photo(self, location_name, waypoint_index):
        """在指定地点拍照并保存"""
        if self.latest_image is None:
            rospy.logwarn("没有接收到图像，无法拍照")
            return False
        
        try:
            # 生成文件名：地点_序号_时间戳.jpg
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{location_name}_{waypoint_index}.jpg"
            filepath = os.path.join(self.results_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, self.latest_image)
            rospy.loginfo("✓ 在 %s 拍照保存: %s", location_name, filename)
            return True
        except Exception as e:
            rospy.logerr("拍照保存失败: %s", str(e))
            return False
    
    def wait_for_image(self, timeout=5.0):
        """等待接收到图像"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.image_received and self.latest_image is not None:
                return True
            rospy.sleep(0.1)
        return False
    
    def read_waypoints_from_file(self):
        """从文件读取航点顺序（不分行扫描）"""
        file_path = "/queue/queue.txt"
        waypoints = []
        found_locations = set()  # 用于记录已找到的地点，避免重复
        
        try:
            if not os.path.exists(file_path):
                rospy.logwarn("航点文件不存在: %s", file_path)
                return waypoints
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().replace('\n', '')  # 移除换行符，将整个文件作为一行处理
                rospy.loginfo("读取文件内容: %s", content)
                
                # 按字符顺序扫描，寻找地点名称
                i = 0
                while i < len(content):
                    # 检查从当前位置开始是否能匹配到任何地点名称
                    matched = False
                    for location, coordinates in self.location_mapping.items():
                        # 如果当前位置开始的子字符串以地点名称开头
                        if content[i:].startswith(location):
                            if location not in found_locations:
                                waypoints.append(coordinates)
                                found_locations.add(location)
                                rospy.loginfo("找到地点: %s -> (%.2f, %.2f, %.2f)", 
                                             location, coordinates[0], coordinates[1], coordinates[2])
                                i += len(location)  # 跳过已匹配的地点名称
                                matched = True
                                break
                            else:
                                rospy.logwarn("跳过重复地点: %s", location)
                                i += len(location)  # 跳过重复的地点名称
                                matched = True
                                break
                    
                    # 如果没有匹配到任何地点名称，移动到下一个字符
                    if not matched:
                        i += 1
                
                # 如果找到的地点不足5个，检查是否是因为有地点没有被识别
                if len(waypoints) < 5:
                    missing_locations = set(self.location_mapping.keys()) - found_locations
                    if missing_locations:
                        rospy.logwarn("文件中未找到以下地点: %s", ", ".join(missing_locations))
            
            rospy.loginfo("从文件中成功读取 %d 个航点", len(waypoints))
            
        except Exception as e:
            rospy.logerr("读取航点文件时出错: %s", str(e))
        
        return waypoints
    
    def get_location_name(self, coordinates):
        """根据坐标获取地点名称"""
        for name, coords in self.location_mapping.items():
            if (abs(coordinates[0] - coords[0]) < 0.01 and 
                abs(coordinates[1] - coords[1]) < 0.01 and 
                abs(coordinates[2] - coords[2]) < 0.01):
                return name
        return "未知地点"
    
    def display_waypoint_order(self):
        """显示航点顺序"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("本次运行的航点顺序（来源: %s）：", self.waypoint_source)
        rospy.loginfo("起点: (-3.5, -3.0, -1.57)")
        
        for i, point in enumerate(self.middle_waypoints):
            location_name = self.get_location_name(point)
            rospy.loginfo("第%d个点: %s (%.2f, %.2f, %.2f)", 
                         i+1, location_name, point[0], point[1], point[2])
        
        rospy.loginfo("终点: (-3.5, -3.0, -1.57)")
        rospy.loginfo("=" * 60)
        
        # 显示完整的航点列表
        rospy.loginfo("完整航点序列 (%d个点):", len(self.waypoints))
        for i, wp in enumerate(self.waypoints):
            if i == 0:
                rospy.loginfo("  0: 起点 (%.2f, %.2f, %.2f)", wp[0], wp[1], wp[2])
            elif i == len(self.waypoints) - 1:
                rospy.loginfo("  %d: 终点 (%.2f, %.2f, %.2f)", i, wp[0], wp[1], wp[2])
            else:
                location_name = self.get_location_name(wp)
                rospy.loginfo("  %d: %s (%.2f, %.2f, %.2f)", i, location_name, wp[0], wp[1], wp[2])
    
    def check_tf_tree(self):
        """检查TF树是否完整"""
        try:
            # 等待必要的TF变换
            self.tf_buffer.can_transform('map', 'base_footprint', rospy.Time(), rospy.Duration(10.0))
            rospy.loginfo("TF tree is complete: map -> base_footprint")
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("TF tree incomplete: %s", str(e))
            return False
    
    def get_current_pose(self):
        """通过TF获取当前位置和朝向"""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
            
            # 提取位置
            position = transform.transform.translation
            
            # 提取朝向并转换为欧拉角
            orientation = transform.transform.rotation
            (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            
            return position, yaw
        except Exception as e:
            rospy.logwarn("Could not get current pose from TF: %s", str(e))
            return None, None
    
    def distance_to_waypoint(self, x, y):
        """计算到目标点的距离"""
        trans, _ = self.get_current_pose()
        if trans is None:
            return float('inf')
        
        dx = x - trans.x
        dy = y - trans.y
        return math.sqrt(dx*dx + dy*dy)
    
    def angle_difference(self, target_yaw, current_yaw):
        """计算两个角度之间的最小差值（-π 到 π）"""
        diff = target_yaw - current_yaw
        # 将角度差标准化到 [-π, π] 范围内
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def is_start_or_end_point(self, waypoint_index):
        """判断是否是起点或终点"""
        return waypoint_index == 0 or waypoint_index == len(self.waypoints) - 1
    
    def update_position_history(self):
        """更新位置历史记录"""
        current_pos, _ = self.get_current_pose()
        if current_pos is not None:
            self.position_history.append((current_pos.x, current_pos.y, time.time()))
    
    def is_stuck(self):
        """检测是否卡住"""
        if len(self.position_history) < 5:  # 至少需要5个点才能判断
            return False
        
        current_time = time.time()
        
        # 只在达到检测时间间隔时进行检查
        if self.last_stuck_check_time is not None and current_time - self.last_stuck_check_time < 1.0:
            return False
        
        self.last_stuck_check_time = current_time
        
        # 获取最近5秒内的位置数据
        recent_positions = [(x, y, t) for x, y, t in self.position_history 
                           if current_time - t <= self.stuck_threshold_time]
        
        if len(recent_positions) < 3:  # 数据点不足
            return False
        
        # 计算移动的总距离
        total_distance = 0
        for i in range(1, len(recent_positions)):
            x1, y1, t1 = recent_positions[i-1]
            x2, y2, t2 = recent_positions[i]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        rospy.loginfo("卡住检测: 最近%.1f秒内移动了%.3f米", self.stuck_threshold_time, total_distance)
        
        # 如果总移动距离小于阈值，认为卡住
        if total_distance < self.stuck_threshold_distance:
            self.stuck_count += 1
            rospy.logwarn("检测到卡住! 连续卡住次数: %d", self.stuck_count)
            return True
        
        # 重置连续卡住计数
        self.stuck_count = 0
        return False
    
    def analyze_stuck_situation(self):
        """分析卡住的情况：墙角还是路中间"""
        if self.laser_data is None:
            rospy.logwarn("没有激光雷达数据，无法分析卡住情况")
            return "unknown"
        
        try:
            # 分析前方障碍物分布
            ranges = list(self.laser_data.ranges)
            num_samples = len(ranges)
            
            # 将激光数据分为前方、左前方、右前方区域
            front_start = int(num_samples * 0.4)  # 前方40%区域
            front_end = int(num_samples * 0.6)
            left_front_start = int(num_samples * 0.2)  # 左前方20%区域
            left_front_end = int(num_samples * 0.4)
            right_front_start = int(num_samples * 0.6)  # 右前方20%区域
            right_front_end = int(num_samples * 0.8)
            
            # 计算各区域的最小距离
            front_ranges = ranges[front_start:front_end]
            left_front_ranges = ranges[left_front_start:left_front_end]
            right_front_ranges = ranges[right_front_start:right_front_end]
            
            # 过滤无效数据
            valid_front = [r for r in front_ranges if r > self.laser_data.range_min and r < self.laser_data.range_max]
            valid_left = [r for r in left_front_ranges if r > self.laser_data.range_min and r < self.laser_data.range_max]
            valid_right = [r for r in right_front_ranges if r > self.laser_data.range_min and r < self.laser_data.range_max]
            
            if not valid_front:
                rospy.loginfo("前方没有有效激光数据")
                return "unknown"
            
            min_front = min(valid_front)
            min_left = min(valid_left) if valid_left else float('inf')
            min_right = min(valid_right) if valid_right else float('inf')
            
            rospy.loginfo("激光分析: 前方=%.2fm, 左前方=%.2fm, 右前方=%.2fm", 
                         min_front, min_left, min_right)
            
            # 判断逻辑
            corner_threshold = 0.5  # 墙角判断阈值
            middle_threshold = 1.0  # 路中间判断阈值
            
            # 如果前方和至少一侧距离很近，可能是墙角
            if min_front < corner_threshold and (min_left < corner_threshold or min_right < corner_threshold):
                rospy.loginfo("判断为墙角卡住")
                return "corner"
            # 如果前方有障碍但两侧相对开阔，可能是路中间
            elif min_front < middle_threshold and min_left > middle_threshold and min_right > middle_threshold:
                rospy.loginfo("判断为路中间卡住")
                return "middle"
            else:
                rospy.loginfo("判断为一般性卡住")
                return "general"
                
        except Exception as e:
            rospy.logwarn("分析卡住情况失败: %s", str(e))
            return "unknown"
    
    def recovery_behavior(self, stuck_type="general"):
        """执行恢复行为，根据卡住类型采取不同策略"""
        rospy.loginfo("执行恢复行为，类型: %s", stuck_type)
        
        try:
            if stuck_type == "corner":
                # 墙角卡住：主要后退策略
                return self.corner_recovery()
            elif stuck_type == "middle":
                # 路中间卡住：小幅直行尝试
                return self.middle_recovery()
            else:
                # 一般性卡住：综合策略
                return self.general_recovery()
            
        except Exception as e:
            rospy.logerr("恢复行为执行失败: %s", str(e))
            return False
    
    def corner_recovery(self):
        """墙角卡住的恢复策略：主要后退"""
        rospy.loginfo("执行墙角恢复策略: 后退并旋转...")
        
        # 第一步：后退0.4米
        start_pos, _ = self.get_current_pose()
        if start_pos is None:
            return False
        
        twist = Twist()
        twist.linear.x = -0.15  # 较慢的后退速度，更安全
        
        # 后退直到移动足够距离或超时
        start_time = time.time()
        while time.time() - start_time < 4.0 and not rospy.is_shutdown():
            current_pos, _ = self.get_current_pose()
            if current_pos is not None:
                distance_moved = math.sqrt((current_pos.x - start_pos.x)**2 + 
                                         (current_pos.y - start_pos.y)**2)
                if distance_moved >= 0.4:
                    break
            
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # 第二步：较大角度旋转（远离墙角）
        twist.angular.z = 0.4  # 固定顺时针旋转
        start_time = time.time()
        while time.time() - start_time < 3.0 and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        rospy.loginfo("✓ 墙角恢复行为完成")
        return True
    
    def middle_recovery(self):
        """路中间卡住的恢复策略：小幅直行尝试"""
        rospy.loginfo("执行路中间恢复策略: 小幅前进后退...")
        
        # 第一步：小幅前进0.2米
        start_pos, _ = self.get_current_pose()
        if start_pos is None:
            return False
        
        twist = Twist()
        twist.linear.x = 0.1  # 很慢的前进速度
        
        # 前进一小段
        start_time = time.time()
        while time.time() - start_time < 3.0 and not rospy.is_shutdown():
            current_pos, _ = self.get_current_pose()
            if current_pos is not None:
                distance_moved = math.sqrt((current_pos.x - start_pos.x)**2 + 
                                         (current_pos.y - start_pos.y)**2)
                if distance_moved >= 0.2:
                    break
            
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # 第二步：小幅后退0.15米
        start_pos, _ = self.get_current_pose()
        twist.linear.x = -0.1
        
        start_time = time.time()
        while time.time() - start_time < 3.0 and not rospy.is_shutdown():
            current_pos, _ = self.get_current_pose()
            if current_pos is not None:
                distance_moved = math.sqrt((current_pos.x - start_pos.x)**2 + 
                                         (current_pos.y - start_pos.y)**2)
                if distance_moved >= 0.15:
                    break
            
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # 第三步：轻微随机旋转
        twist.angular.z = random.choice([-0.2, 0.2])
        start_time = time.time()
        while time.time() - start_time < 2.0 and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        rospy.loginfo("✓ 路中间恢复行为完成")
        return True
    
    def general_recovery(self):
        """一般性卡住的恢复策略：综合方法"""
        rospy.loginfo("执行一般恢复策略: 后退并旋转...")
        
        # 第一步：后退0.3米
        start_pos, _ = self.get_current_pose()
        if start_pos is None:
            return False
        
        twist = Twist()
        twist.linear.x = -0.2
        
        # 后退直到移动足够距离或超时
        start_time = time.time()
        while time.time() - start_time < 3.0 and not rospy.is_shutdown():
            current_pos, _ = self.get_current_pose()
            if current_pos is not None:
                distance_moved = math.sqrt((current_pos.x - start_pos.x)**2 + 
                                         (current_pos.y - start_pos.y)**2)
                if distance_moved >= 0.3:
                    break
            
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # 第二步：随机旋转
        twist.angular.z = random.choice([-0.3, 0.3])
        start_time = time.time()
        while time.time() - start_time < 2.0 and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        
        # 停止
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        rospy.loginfo("✓ 一般恢复行为完成")
        return True
    
    def adjust_orientation(self, target_yaw, timeout=30.0):
        """小幅度旋转调整朝向直到与目标相同"""
        rospy.loginfo("开始调整朝向，目标朝向: %.2f 弧度", target_yaw)
        
        start_time = time.time()
        orientation_adjusted = False
        max_angular_speed = 0.3  # 最大角速度 rad/s
        min_angular_speed = 0.05  # 最小角速度 rad/s
        angle_tolerance = 0.03   
        
        while not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            # 获取当前朝向
            _, current_yaw = self.get_current_pose()
            if current_yaw is None:
                rospy.logwarn("无法获取当前朝向，继续尝试...")
                rospy.sleep(0.1)
                continue
            
            # 计算角度差
            angle_diff = self.angle_difference(target_yaw, current_yaw)
            abs_angle_diff = abs(angle_diff)
            
            rospy.loginfo("当前朝向: %.2f, 目标朝向: %.2f, 角度差: %.2f 弧度 (约 %.1f 度)", 
                         current_yaw, target_yaw, angle_diff, math.degrees(abs_angle_diff))
            
            # 检查是否已达到目标朝向
            if abs_angle_diff < angle_tolerance:
                rospy.loginfo("✓ 朝向调整完成! 最终角度差: %.2f 弧度", angle_diff)
                orientation_adjusted = True
                break
            
            # 计算角速度（根据角度差调整速度）
            angular_speed = max(min_angular_speed, min(max_angular_speed, abs_angle_diff * 0.5))
            if angle_diff < 0:
                angular_speed = -angular_speed  # 顺时针旋转
            
            # 发布旋转命令
            twist = Twist()
            twist.angular.z = angular_speed
            self.cmd_vel_pub.publish(twist)
            
            rospy.sleep(0.1)  # 控制循环频率
        
        # 停止机器人
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        if not orientation_adjusted:
            rospy.logwarn("朝向调整超时，可能未完全对准目标朝向")
        
        return orientation_adjusted
    
    def start_navigation_timer(self):
        """开始导航计时"""
        if not self.navigation_started:
            self.start_time = time.time()
            self.navigation_started = True
            rospy.loginfo("导航计时开始! 开始时间: %.2f", self.start_time)
            rospy.loginfo("从起点 (-3.5, -3.0) 开始运行一圈...")
    
    def stop_navigation_timer(self):
        """停止导航计时并显示结果"""
        if self.navigation_started and self.end_time is None and self.circle_completed:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            minutes = int(duration // 60)
            seconds = duration % 60
            
            rospy.loginfo("=" * 60)
            rospy.loginfo("一圈导航完成!")
            rospy.loginfo(" 开始时间: %.2f", self.start_time)
            rospy.loginfo(" 结束时间: %.2f", self.end_time)
            rospy.loginfo(" 总耗时: %d 分 %.2f 秒", minutes, seconds)
            rospy.loginfo("=" * 60)
    
    def create_goal(self, x, y, yaw):
        """创建导航目标"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # 设置位置
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        
        # 设置朝向（四元数）
        quat = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        
        return goal
    
    def navigate_waypoints(self):
        """按顺序导航到各个航点"""
        if len(self.waypoints) == 0:
            rospy.logwarn("没有提供航点!")
            return
        
        waypoint_index = 0
        max_attempts = 3  # 每个航点的最大尝试次数
        # 移除 max_stuck_recoveries 限制，改为无限次恢复尝试
        
        while not rospy.is_shutdown() and len(self.waypoints) > 0:
            # 获取当前航点
            wp = self.waypoints[waypoint_index]
            x, y, target_yaw = wp[0], wp[1], wp[2]
            
            # 如果是第一个点（起点），开始计时
            if waypoint_index == 0 and not self.navigation_started:
                self.start_navigation_timer()
            
            # 显示导航信息
            if waypoint_index == 0:
                point_type = "起点"
                location_name = "起点"
            elif waypoint_index == len(self.waypoints) - 1:
                point_type = "终点"
                location_name = "终点"
            else:
                point_type = "航点"
                location_name = self.get_location_name(wp)
            
            rospy.loginfo("=== 导航到 %s %d/%d: %s (%.2f, %.2f, %.2f) ===", 
                        point_type, waypoint_index + 1, len(self.waypoints), 
                        location_name, x, y, target_yaw)
            
            attempt = 0
            success = False
            stuck_recovery_count = 0
            
            while attempt < max_attempts and not rospy.is_shutdown():
                attempt += 1
                rospy.loginfo("尝试 %d/%d", attempt, max_attempts)
                
                # 重置卡住检测相关变量
                self.position_history.clear()
                self.last_stuck_check_time = None
                self.stuck_count = 0
                
                # 创建并发送目标
                goal = self.create_goal(x, y, target_yaw)
                self.client.send_goal(goal)
                
                # 等待结果，设置超时时间
                wait_time = 60.0  # 60秒超时
                start_wait_time = time.time()
                stuck_detected_in_this_attempt = False
                
                while not rospy.is_shutdown():
                    # 更新位置历史用于卡住检测
                    self.update_position_history()
                    
                    # 检查是否卡住（只在导航过程中检测，不包含起点终点）
                    if not self.is_start_or_end_point(waypoint_index) and self.is_stuck():
                        if self.stuck_count >= 3:  # 检测到3次卡住就立即执行恢复
                            rospy.logwarn("连续检测到3次卡住，立即执行恢复程序...")
                            
                            # 分析卡住类型
                            stuck_type = self.analyze_stuck_situation()
                            rospy.logwarn("卡住类型: %s，执行恢复行为...", stuck_type)
                            
                            self.client.cancel_goal()  # 取消当前目标
                            
                            # 执行恢复行为
                            if self.recovery_behavior(stuck_type):
                                stuck_recovery_count += 1
                                rospy.loginfo("恢复行为完成 (第%d次恢复)，重新发送目标", stuck_recovery_count)
                                
                                # 重置卡住计数
                                self.stuck_count = 0
                                self.position_history.clear()
                                
                                # 重新发送目标
                                goal = self.create_goal(x, y, target_yaw)
                                self.client.send_goal(goal)
                                start_wait_time = time.time()  # 重置等待时间
                                stuck_detected_in_this_attempt = True
                            else:
                                rospy.logwarn("恢复行为失败")
                    
                    # 检查动作结果
                    state = self.client.get_state()
                    if state == actionlib.GoalStatus.SUCCEEDED:
                        rospy.loginfo("✓ 成功到达 %s %d (%s)", point_type, waypoint_index + 1, location_name)
                        success = True
                        break
                    elif state in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.PREEMPTED]:
                        rospy.logwarn("✗ 未能到达 %s %d (%s). 状态: %d", 
                                    point_type, waypoint_index + 1, location_name, state)
                        break
                    
                    # 检查超时
                    if time.time() - start_wait_time > wait_time:
                        rospy.logwarn("导航到 %s %d (%s) 超时", point_type, waypoint_index + 1, location_name)
                        self.client.cancel_goal()
                        break
                    
                    # 检查是否接近目标点（作为备用成功条件）
                    distance = self.distance_to_waypoint(x, y)
                    if distance < 0.06:  # 如果距离小于10cm，认为成功
                        rospy.loginfo("✓ 足够接近 %s %d (%s) (距离: %.2f米)", 
                                    point_type, waypoint_index + 1, location_name, distance)
                        success = True
                        self.client.cancel_goal()
                        break
                    
                    rospy.sleep(0.5)  # 检查间隔
                
                if success:
                    break
                else:
                    rospy.logwarn("尝试 %d 失败", attempt)
                    if attempt < max_attempts:
                        rospy.loginfo("5秒后重试...")
                        rospy.sleep(5.0)
            
            if success:
                # 判断是否是起点或终点
                is_start_end = self.is_start_or_end_point(waypoint_index)
                
                if not is_start_end:
                    # 只有中间航点才进行朝向调整和拍照
                    rospy.loginfo("位置到达成功，开始朝向调整...")
                    orientation_success = self.adjust_orientation(target_yaw)
                    
                    if orientation_success:
                        rospy.loginfo(" 位置和朝向都调整完成")
                    else:
                        rospy.logwarn(" 位置到达但朝向调整未完全完成")
                    
                    # 在成功到达后拍照（只对中间航点拍照）
                    rospy.loginfo("到达地点，准备拍照...")
                    # 等待图像稳定
                    rospy.sleep(2.0)
                    # 拍照
                    self.take_photo(location_name, waypoint_index)
                else:
                    rospy.loginfo("到达起点/终点，跳过朝向调整和拍照")
                
                waypoint_index += 1
                
                # 检查是否完成一圈（到达终点）
                if waypoint_index >= len(self.waypoints):
                    self.circle_completed = True
                    self.stop_navigation_timer()
                    rospy.loginfo("一圈导航完成!")
                    break
                else:
                    rospy.loginfo("前往下一个航点...")
            else:
                rospy.logwarn("经过 %d 次尝试后跳过 %s %d (%s)", 
                            max_attempts, point_type, waypoint_index + 1, location_name)
                waypoint_index += 1  # 跳过失败的航点
                
                # 如果跳过的是终点，也要标记完成
                if waypoint_index >= len(self.waypoints):
                    self.circle_completed = True
                    self.stop_navigation_timer()
                    rospy.loginfo("一圈导航完成（跳过终点）!")
                    break
            
            rospy.sleep(2.0)  # 航点间暂停

def main():
    try:
        navigator = WaypointNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("航点导航被中断")
    except Exception as e:
        rospy.logerr("航点导航错误: %s", str(e))

if __name__ == '__main__':
    main()