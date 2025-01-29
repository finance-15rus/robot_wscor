import sys
import time
import json
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vision_system import VisionSystemManager

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Энкодер с увеличенным количеством слоев
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8)
        )
        
        # Декодер с симметричной архитектурой
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class Payload:
    def __init__(self, size=0.2, mass=1.0):
        self.size = size
        self.mass = mass
        self.position = np.array([0.8, 0.8, 0.1])
        self.is_gripped = False
        self.color = (0.7, 0.1, 0.1)
        self.grip_points = self.calculate_grip_points()
        
        # Добавляем параметры для отслеживания состояния
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.force = np.zeros(3)
        self.orientation = np.zeros(3)
        
    def calculate_grip_points(self):
        """Расчет точек для захвата с улучшенной логикой"""
        points = []
        s = self.size / 2
        
        # Определяем основные точки захвата по углам
        for x in [-s, s]:
            for y in [-s, s]:
                for z in [-s, s]:
                    point = np.array([x, y, z])
                    # Добавляем небольшое смещение для лучшего захвата
                    offset = np.random.normal(0, 0.01, 3)
                    points.append(point + offset)
        
        # Добавляем точки захвата по центрам граней
        face_centers = [
            np.array([s, 0, 0]),
            np.array([-s, 0, 0]),
            np.array([0, s, 0]),
            np.array([0, -s, 0]),
            np.array([0, 0, s]),
            np.array([0, 0, -s])
        ]
        points.extend(face_centers)
        
        return points
    
    def update_state(self, dt):
        """Обновление состояния груза"""
        if not self.is_gripped:
            # Применяем гравитацию
            self.acceleration[2] = -9.81
            self.velocity += self.acceleration * dt
            self.position += self.velocity * dt
            
            # Проверка столкновения с полом
            if self.position[2] < self.size/2:
                self.position[2] = self.size/2
                self.velocity[2] = 0

class IndustrialRobot:
    def __init__(self):
        # Параметры робота
        self.joint_angles = np.zeros(6)
        self.base_height = 0.4
        self.base_radius = 0.3
        self.shoulder_length = 0.8
        self.forearm_length = 0.7
        self.wrist_length = 0.4
        self.end_effector_length = 0.2
        
        # Цвета компонентов
        self.base_color = (0.2, 0.2, 0.2)
        self.robot_color = (1.0, 0.5, 0.0)
        self.joint_color = (0.8, 0.8, 0.8)
        self.detail_color = (0.3, 0.3, 0.3)
        
        # Параметры захвата
        self.gripper_state = 0.0  # 0 - закрыт, 1 - открыт
        self.gripper_width = 0.2
        self.finger_length = 0.3
        self.finger_segments = 3
        self.grip_force = 50.0  # Сила захвата в процентах
        
        # Создаем груз
        self.payload = Payload()
        
        # Ограничения суставов
        self.joint_limits = [
            (-180, 180),  # База
            (-90, 90),    # Плечо
            (-180, 0),    # Локоть
            (-180, 180),  # Запястье 1
            (-90, 90),    # Запястье 2
            (-180, 180)   # Инструмент
        ]
        
        # Параметры динамики
        self.joint_velocities = np.zeros(6)
        self.joint_accelerations = np.zeros(6)
        self.max_velocities = np.array([180, 150, 150, 180, 180, 270])  # градусов/с
        self.max_accelerations = np.array([90, 75, 75, 90, 90, 135])    # градусов/с²
        
        # Система компьютерного зрения
        self.vision_system = VisionSystemManager()
        
        # Параметры безопасности
        self.safety_limits = {
            'velocity': 0.8,  # 80% от максимальной скорости
            'acceleration': 0.7,  # 70% от максимального ускорения
            'workspace': 1.5,  # Радиус рабочей зоны
            'min_height': 0.1,  # Минимальная высота над поверхностью
            'collision_margin': 0.05  # Запас для предотвращения столкновений
        }
        
    def get_end_effector_position(self):
        """Расчет положения захвата с учетом всех трансформаций"""
        x, y, z = 0, 0, self.base_height
        angle_sum = 0
        
        # Преобразование для базы
        angle_sum = np.radians(self.joint_angles[0])
        
        # Преобразование для плеча
        z += self.shoulder_length * np.sin(np.radians(self.joint_angles[1]))
        xy = self.shoulder_length * np.cos(np.radians(self.joint_angles[1]))
        x += xy * np.cos(angle_sum)
        y += xy * np.sin(angle_sum)
        
        # Преобразование для локтя
        angle2 = np.radians(self.joint_angles[1] + self.joint_angles[2])
        z += self.forearm_length * np.sin(angle2)
        xy = self.forearm_length * np.cos(angle2)
        x += xy * np.cos(angle_sum)
        y += xy * np.sin(angle_sum)
        
        # Преобразование для запястья
        angle3 = np.radians(self.joint_angles[1] + self.joint_angles[2] + self.joint_angles[3])
        z += self.wrist_length * np.sin(angle3)
        xy = self.wrist_length * np.cos(angle3)
        x += xy * np.cos(angle_sum)
        y += xy * np.sin(angle_sum)
        
        return np.array([x, y, z])
    
    def update_dynamics(self, dt):
        """Обновление динамики робота"""
        # Обновление скоростей и ускорений с учетом ограничений
        for i in range(6):
            # Применение ограничений ускорения
            self.joint_accelerations[i] = np.clip(
                self.joint_accelerations[i],
                -self.max_accelerations[i],
                self.max_accelerations[i]
            )
            
            # Обновление скоростей
            new_velocity = self.joint_velocities[i] + self.joint_accelerations[i] * dt
            self.joint_velocities[i] = np.clip(
                new_velocity,
                -self.max_velocities[i],
                self.max_velocities[i]
            )
            
            # Обновление углов
            new_angle = self.joint_angles[i] + self.joint_velocities[i] * dt
            self.joint_angles[i] = np.clip(
                new_angle,
                self.joint_limits[i][0],
                self.joint_limits[i][1]
            )
    
    def check_safety_constraints(self):
        """Проверка ограничений безопасности"""
        # Проверка скоростей
        velocity_check = all(abs(v) <= limit * self.safety_limits['velocity'] 
                           for v, limit in zip(self.joint_velocities, self.max_velocities))
        
        # Проверка рабочей зоны
        position = self.get_end_effector_position()
        distance_from_base = np.linalg.norm(position[:2])
        workspace_check = distance_from_base <= self.safety_limits['workspace']
        
        # Проверка высоты
        height_check = position[2] >= self.safety_limits['min_height']
        
        return velocity_check and workspace_check and height_check
    
    def apply_grip_force(self):
        """Применение силы захвата"""
        if self.payload.is_gripped:
            # Расчет реальной силы захвата
            max_force = 100.0  # Максимальная сила в Ньютонах
            applied_force = max_force * (self.grip_force / 100.0)
            
            # Обновление состояния груза
            self.payload.force = np.array([0, 0, applied_force])
            
            # Проверка надежности захвата
            if applied_force < self.payload.mass * 9.81:
                self.payload.is_gripped = False
                print("Warning: Grip force insufficient to hold payload!")

class RobotVisualizerWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.robot = IndustrialRobot()
        self.camera_distance = 3.0
        self.camera_rotation = [45.0, 45.0]
        self.last_pos = QPoint()
        
        # Буферы для визуализации
        self.frame_buffer = None
        self.vision_results = None
        self.last_frame_time = time.time()
        
        # Параметры отрисовки
        self.show_axes = True
        self.show_grid = True
        self.show_workspace = True
        self.show_trajectory = True
        self.trajectory_points = []
        self.max_trajectory_points = 1000
        
        # Параметры освещения
        self.light_position = [5.0, 5.0, 5.0, 1.0]
        self.light_ambient = [0.2, 0.2, 0.2, 1.0]
        self.light_diffuse = [1.0, 1.0, 1.0, 1.0]
        self.light_specular = [1.0, 1.0, 1.0, 1.0]
        
        # Материалы
        self.material_ambient = [0.2, 0.2, 0.2, 1.0]
        self.material_diffuse = [0.8, 0.8, 0.8, 1.0]
        self.material_specular = [1.0, 1.0, 1.0, 1.0]
        self.material_shininess = 50.0
        
        # Инициализация таймера для обновления
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_simulation)
        self.update_timer.start(16)  # 60 FPS

    def initializeGL(self):
        """Инициализация OpenGL с расширенными настройками"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Настройка освещения
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.light_specular)
        
        # Настройка материалов
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.material_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.material_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.material_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.material_shininess)
        
        # Настройка фона
        glClearColor(0.15, 0.15, 0.15, 1.0)
        
        # Включение сглаживания
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        
    def resizeGL(self, w, h):
        """Обновление размера окна с настройкой перспективы"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Отрисовка сцены с расширенной функциональностью"""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Очистка буферов
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Установка камеры
        x = self.camera_distance * math.sin(math.radians(self.camera_rotation[0])) * math.cos(math.radians(self.camera_rotation[1]))
        y = self.camera_distance * math.sin(math.radians(self.camera_rotation[0])) * math.sin(math.radians(self.camera_rotation[1]))
        z = self.camera_distance * math.cos(math.radians(self.camera_rotation[0]))
        
        gluLookAt(x, y, z, 0, 0, 0.5, 0, 0, 1)
        
        # Отрисовка элементов сцены
        if self.show_grid:
            self.draw_grid()
            
        if self.show_axes:
            self.draw_axes()
            
        if self.show_workspace:
            self.draw_workspace()
            
        # Обновление и отрисовка робота
        self.robot.update_dynamics(dt)
        self.draw_robot()
        
        # Отрисовка груза
        self.draw_payload()
        
        # Отрисовка траектории
        if self.show_trajectory:
            self.update_trajectory()
            self.draw_trajectory()
            
        # Захват кадра для компьютерного зрения
        self.capture_frame()
        
    def draw_grid(self):
        """Отрисовка улучшенной сетки"""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        
        # Основная сетка
        glColor3f(0.3, 0.3, 0.3)
        for i in range(-10, 11):
            glVertex3f(i, -10, 0)
            glVertex3f(i, 10, 0)
            glVertex3f(-10, i, 0)
            glVertex3f(10, i, 0)
        
        # Выделенные оси
        glColor3f(0.5, 0, 0)  # X - красный
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        
        glColor3f(0, 0.5, 0)  # Y - зеленый
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        
        glColor3f(0, 0, 0.5)  # Z - синий
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_axes(self):
        """Отрисовка осей координат с подписями"""
        glDisable(GL_LIGHTING)
        
        # Рисуем оси
        glBegin(GL_LINES)
        # X ось (красная)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(1.2, 0, 0)
        
        # Y ось (зеленая)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1.2, 0)
        
        # Z ось (синяя)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1.2)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def draw_workspace(self):
        """Отрисовка границ рабочей зоны"""
        glDisable(GL_LIGHTING)
        glColor4f(0.5, 0.5, 0.5, 0.2)
        
        # Рисуем полупрозрачную сферу
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_LINE)
        gluSphere(quad, self.robot.safety_limits['workspace'], 32, 32)
        
        glEnable(GL_LIGHTING)
        
    def update_trajectory(self):
        """Обновление точек траектории"""
        current_pos = self.robot.get_end_effector_position()
        self.trajectory_points.append(current_pos)
        
        # Ограничение количества точек
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
            
    def draw_trajectory(self):
        """Отрисовка траектории движения"""
        if len(self.trajectory_points) < 2:
            return
            
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # Желтый цвет для траектории
        glBegin(GL_LINE_STRIP)
        
        for point in self.trajectory_points:
            glVertex3fv(point)
            
        glEnd()
        glEnable(GL_LIGHTING)
        
    def draw_robot(self):
        """Отрисовка робота с улучшенными деталями"""
        glPushMatrix()
        
        self.draw_enhanced_base()
        
        glRotatef(self.robot.joint_angles[0], 0, 0, 1)
        self.draw_enhanced_rotating_base()
        
        glTranslatef(0, 0, self.robot.base_height)
        glRotatef(self.robot.joint_angles[1], 0, 1, 0)
        self.draw_enhanced_shoulder()
        
        glTranslatef(self.robot.shoulder_length, 0, 0)
        glRotatef(self.robot.joint_angles[2], 0, 1, 0)
        self.draw_enhanced_forearm()
        
        glTranslatef(self.robot.forearm_length, 0, 0)
        glRotatef(self.robot.joint_angles[3], 1, 0, 0)
        self.draw_enhanced_wrist()
        
        glTranslatef(self.robot.wrist_length, 0, 0)
        glRotatef(self.robot.joint_angles[4], 0, 1, 0)
        glRotatef(self.robot.joint_angles[5], 1, 0, 0)
        self.draw_enhanced_end_effector()
        
        glPopMatrix()
        
    def capture_frame(self):
        """Захват кадра для системы компьютерного зрения"""
        try:
            width = self.width()
            height = self.height()
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            
            # Захват буфера цвета
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8)
            image = image.reshape(height, width, 3)
            image = np.flipud(image)
            
            self.frame_buffer = image
            
            # Обработка кадра системой компьютерного зрения
            if self.robot.vision_system.vision_system.is_trained:
                # Передача кадра в систему компьютерного зрения
                result = self.robot.vision_system.process_frame(image)
                if result is not None:
                    self.vision_results = result.detection_map
                    # Обновление отображения результатов
                    self.update_vision_display(result)
                    
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            
    def update_vision_display(self, result):
        """Обновление отображения результатов компьютерного зрения"""
        if hasattr(self, 'vision_window'):
            plt.figure(1)
            plt.clf()
            
            # Отображение исходного кадра
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGR2RGB))
            plt.title('Original Frame')
            plt.axis('off')
            
            # Отображение результатов детекции
            plt.subplot(222)
            plt.imshow(result.detection_map, cmap='viridis')
            plt.title(f'Detection Result (Conf: {result.confidence:.2f})')
            plt.colorbar()
            plt.axis('off')
            
            # График уверенности детекции
            plt.subplot(223)
            plt.plot(self.robot.vision_system.vision_system.detection_confidences[-100:])
            plt.title('Detection Confidence History')
            plt.xlabel('Frame')
            plt.ylabel('Confidence')
            
            # Отображение обработанного кадра с наложением
            plt.subplot(224)
            vis_frame = self.robot.vision_system.visualize_current_detection(
                self.frame_buffer.copy()
            )
            plt.imshow(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            plt.title('Detection Visualization')
            plt.axis('off')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

    def mouseMoveEvent(self, event):
        """Обработка движений мыши"""
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            # Вращение камеры
            self.camera_rotation[1] += dx * 0.5
            self.camera_rotation[0] = np.clip(
                self.camera_rotation[0] + dy * 0.5,
                1, 179
            )
            self.update()
            
        elif event.buttons() & Qt.RightButton:
            # Управление роботом
            if event.modifiers() & Qt.ShiftModifier:
                # Управление локтем и запястьем
                self.robot.joint_angles[2] = np.clip(
                    self.robot.joint_angles[2] + dy * 0.5,
                    *self.robot.joint_limits[2]
                )
                self.robot.joint_angles[3] = np.clip(
                    self.robot.joint_angles[3] + dx * 0.5,
                    *self.robot.joint_limits[3]
                )
            else:
                # Управление базой и плечом
                self.robot.joint_angles[0] += dx * 0.5
                self.robot.joint_angles[1] = np.clip(
                    self.robot.joint_angles[1] + dy * 0.5,
                    *self.robot.joint_limits[1]
                )
            self.update()
        
        self.last_pos = event.pos()

    def wheelEvent(self, event):
        """Обработка колесика мыши"""
        if event.modifiers() & Qt.ControlModifier:
            # Управление захватом
            delta = np.sign(event.angleDelta().y()) * 0.1
            self.robot.gripper_state = np.clip(
                self.robot.gripper_state + delta,
                0.0, 1.0
            )
        else:
            # Приближение/отдаление камеры
            self.camera_distance = np.clip(
                self.camera_distance - event.angleDelta().y() / 120.0,
                2.0, 10.0
            )
        self.update()

    def keyPressEvent(self, event):
        """Обработка нажатий клавиш"""
        key = event.key()
        
        if key == Qt.Key_Space:
            # Аварийная остановка
            self.emit_emergency_stop()
        elif key == Qt.Key_R:
            # Сброс положения
            self.reset_robot_position()
        elif key == Qt.Key_G:
            # Управление захватом
            self.toggle_gripper()
        elif key == Qt.Key_V:
            # Включение/выключение отображения результатов зрения
            self.toggle_vision_display()
        
        self.update()

    def emit_emergency_stop(self):
        """Сигнал аварийной остановки"""
        # Остановка всех движений
        self.robot.joint_velocities = np.zeros(6)
        self.robot.joint_accelerations = np.zeros(6)
        # Отпускание груза
        self.robot.payload.is_gripped = False
        print("Emergency stop activated!")

    def reset_robot_position(self):
        """Сброс положения робота"""
        self.robot.joint_angles = np.zeros(6)
        self.robot.joint_velocities = np.zeros(6)
        self.robot.joint_accelerations = np.zeros(6)
        self.robot.gripper_state = 0.0
        self.trajectory_points.clear()

    def toggle_gripper(self):
        """Управление захватом"""
        if not self.robot.payload.is_gripped:
            # Проверка возможности захвата
            if self.check_grip_collision():
                self.robot.payload.is_gripped = True
                self.robot.gripper_state = 0.5
        else:
            self.robot.payload.is_gripped = False
            self.robot.gripper_state = 0.0

    def toggle_vision_display(self):
        """Включение/выключение отображения результатов зрения"""
        if hasattr(self, 'vision_window'):
            plt.close('all')
            del self.vision_window
        else:
            self.setup_vision_window()

    def setup_vision_window(self):
        """Настройка окна визуализации компьютерного зрения"""
        plt.figure(1)
        plt.subplot(221)
        plt.title('Original Frame')
        plt.subplot(222)
        plt.title('Detection Result')
        plt.subplot(223)
        plt.title('Detection Confidence')
        plt.subplot(224)
        plt.title('Detection Visualization')
        plt.show(block=False)
        self.vision_window = plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Robot Simulator with Enhanced Computer Vision")
        
        # Инициализация системы компьютерного зрения
        self.vision_manager = VisionSystemManager()
        
        self.setup_ui()
        self.setup_timer()
        self.setup_vision_window()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # 3D визуализация
        self.visualizer = RobotVisualizerWidget()
        layout.addWidget(self.visualizer, stretch=2)
        
        # Панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Настройка вкладок
        tabs = QTabWidget()
        
        # Добавляем вкладки
        tabs.addTab(self.create_joints_tab(), "Joints")
        tabs.addTab(self.create_vision_tab(), "Computer Vision")
        tabs.addTab(self.create_gripper_tab(), "Gripper")
        
        control_layout.addWidget(tabs)
        
        # Добавляем кнопки управления
        control_layout.addLayout(self.create_control_buttons())
        
        layout.addWidget(control_panel)
        
        # Размер окна
        self.setGeometry(100, 100, 1400, 800)

   def setup_timer(self):
        """Настройка таймеров обновления"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(30)  # 30 мс = ~33 fps
        
        self.vision_timer = QTimer()
        self.vision_timer.timeout.connect(self.update_vision)
        self.vision_timer.start(100)  # 100 мс = 10 fps для зрения

    def create_joints_tab(self):
        """Создание вкладки управления суставами"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Слайдеры для суставов
        self.joint_sliders = []
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Tool"]
        
        for i, name in enumerate(joint_names):
            group = QGroupBox(f"{name} Joint")
            group_layout = QGridLayout()
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*[int(x) for x in self.visualizer.robot.joint_limits[i]])
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, idx=i: self.update_joint(idx, val))
            
            value_label = QLabel("0°")
            slider.valueChanged.connect(lambda val, label=value_label: label.setText(f"{val}°"))
            
            group_layout.addWidget(QLabel("Position:"), 0, 0)
            group_layout.addWidget(slider, 0, 1)
            group_layout.addWidget(value_label, 0, 2)
            
            velocity_label = QLabel("Velocity: 0.00")
            torque_label = QLabel("Torque: 0.00")
            
            group_layout.addWidget(velocity_label, 1, 0, 1, 2)
            group_layout.addWidget(torque_label, 2, 0, 1, 2)
            
            group.setLayout(group_layout)
            layout.addWidget(group)
            
            self.joint_sliders.append({
                'slider': slider,
                'value': value_label,
                'velocity': velocity_label,
                'torque': torque_label
            })
        
        return tab

    def create_vision_tab(self):
        """Создание вкладки компьютерного зрения"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Информация о системе зрения
        vision_info = QGroupBox("Vision System Status")
        vision_info_layout = QVBoxLayout()
        
        self.vision_status = QLabel("Vision System: Initializing")
        self.object_detection = QLabel("Detected Objects: None")
        self.confidence_score = QLabel("Detection Confidence: 0%")
        self.learning_progress = QProgressBar()
        self.learning_progress.setRange(0, 100)
        
        # Добавление статистики
        self.detection_stats = QLabel("Detection Statistics: N/A")
        
        vision_info_layout.addWidget(self.vision_status)
        vision_info_layout.addWidget(self.object_detection)
        vision_info_layout.addWidget(self.confidence_score)
        vision_info_layout.addWidget(QLabel("Learning Progress:"))
        vision_info_layout.addWidget(self.learning_progress)
        vision_info_layout.addWidget(self.detection_stats)
        
        vision_info.setLayout(vision_info_layout)
        layout.addWidget(vision_info)
        
        # Настройки зрения
        vision_settings = QGroupBox("Vision Settings")
        vision_settings_layout = QVBoxLayout()
        
        # Кнопки управления моделью
        model_buttons = QHBoxLayout()
        
        save_model_button = QPushButton("Save Model")
        save_model_button.clicked.connect(self.save_vision_model)
        
        load_model_button = QPushButton("Load Model")
        load_model_button.clicked.connect(self.load_vision_model)
        
        model_buttons.addWidget(save_model_button)
        model_buttons.addWidget(load_model_button)
        
        vision_settings_layout.addLayout(model_buttons)
        
        # Настройки обучения
        train_settings = QGroupBox("Training Settings")
        train_layout = QGridLayout()
        
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(100, 5000)
        self.min_samples_spin.setValue(1000)
        self.min_samples_spin.setSingleStep(100)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 256)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setSingleStep(16)
        
        train_layout.addWidget(QLabel("Min Samples:"), 0, 0)
        train_layout.addWidget(self.min_samples_spin, 0, 1)
        train_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        train_layout.addWidget(self.batch_size_spin, 1, 1)
        
        train_settings.setLayout(train_layout)
        vision_settings_layout.addWidget(train_settings)
        
        # Кнопки управления обучением
        train_button = QPushButton("Train Vision System")
        train_button.clicked.connect(self.train_vision_system)
        reset_button = QPushButton("Reset Training")
        reset_button.clicked.connect(self.reset_vision_system)
        
        vision_settings_layout.addWidget(train_button)
        vision_settings_layout.addWidget(reset_button)
        
        vision_settings.setLayout(vision_settings_layout)
        layout.addWidget(vision_settings)
        
        return tab

    def create_gripper_tab(self):
        """Создание вкладки управления захватом"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Группа управления захватом
        grip_group = QGroupBox("Gripper Control")
        grip_layout = QVBoxLayout()
        
        self.grip_button = QPushButton("Grip")
        self.grip_button.setCheckable(True)
        self.grip_button.clicked.connect(self.toggle_gripper)
        
        # Слайдер силы захвата
        grip_force = QSlider(Qt.Horizontal)
        grip_force.setRange(0, 100)
        grip_force.setValue(50)
        grip_force.valueChanged.connect(self.update_grip_force)
        
        grip_layout.addWidget(self.grip_button)
        grip_layout.addWidget(QLabel("Grip Force:"))
        grip_layout.addWidget(grip_force)
        
        # Информация о захвате
        self.grip_status = QLabel("Status: Ready")
        self.grip_force_label = QLabel("Force: 50%")
        self.object_status = QLabel("Object: Not detected")
        
        grip_layout.addWidget(self.grip_status)
        grip_layout.addWidget(self.grip_force_label)
        grip_layout.addWidget(self.object_status)
        
        grip_group.setLayout(grip_layout)
        layout.addWidget(grip_group)
        
        # Расширенные настройки захвата
        advanced_group = QGroupBox("Advanced Gripper Settings")
        advanced_layout = QVBoxLayout()
        
        # Настройки пальцев
        finger_settings = QGroupBox("Finger Settings")
        finger_layout = QGridLayout()
        
        self.finger_width = QDoubleSpinBox()
        self.finger_width.setRange(0.1, 0.5)
        self.finger_width.setValue(0.2)
        self.finger_width.setSingleStep(0.01)
        
        self.finger_length = QDoubleSpinBox()
        self.finger_length.setRange(0.1, 0.5)
        self.finger_length.setValue(0.3)
        self.finger_length.setSingleStep(0.01)
        
        finger_layout.addWidget(QLabel("Width:"), 0, 0)
        finger_layout.addWidget(self.finger_width, 0, 1)
        finger_layout.addWidget(QLabel("Length:"), 1, 0)
        finger_layout.addWidget(self.finger_length, 1, 1)
        
        finger_settings.setLayout(finger_layout)
        advanced_layout.addWidget(finger_settings)
        
        # Настройки захвата
        grip_settings = QGroupBox("Grip Settings")
        grip_settings_layout = QGridLayout()
        
        self.grip_speed = QDoubleSpinBox()
        self.grip_speed.setRange(0.1, 2.0)
        self.grip_speed.setValue(1.0)
        self.grip_speed.setSingleStep(0.1)
        
        self.grip_threshold = QDoubleSpinBox()
        self.grip_threshold.setRange(0.1, 1.0)
        self.grip_threshold.setValue(0.5)
        self.grip_threshold.setSingleStep(0.05)
        
        grip_settings_layout.addWidget(QLabel("Speed:"), 0, 0)
        grip_settings_layout.addWidget(self.grip_speed, 0, 1)
        grip_settings_layout.addWidget(QLabel("Threshold:"), 1, 0)
        grip_settings_layout.addWidget(self.grip_threshold, 1, 1)
        
        grip_settings.setLayout(grip_settings_layout)
        advanced_layout.addWidget(grip_settings)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        return tab
    
    def create_control_buttons(self):
        """Создание кнопок управления"""
        buttons_layout = QHBoxLayout()
        
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_simulation)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop_simulation)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_simulation)
        
        buttons_layout.addWidget(start_button)
        buttons_layout.addWidget(stop_button)
        buttons_layout.addWidget(reset_button)
        
        emergency_stop = QPushButton("EMERGENCY STOP")
        emergency_stop.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:pressed {
                background-color: darkred;
            }
        """)
        emergency_stop.clicked.connect(self.emergency_stop)
        buttons_layout.addWidget(emergency_stop)
        
        return buttons_layout

    def update_simulation(self):
        """Обновление симуляции"""
        self.visualizer.update()
        if not self.visualizer.robot.payload.is_gripped:
            if self.visualizer.check_grip_collision():
                self.grip_status.setText("Status: Ready to grip")
                self.object_status.setText("Object: In range")
            else:
                self.grip_status.setText("Status: Ready")
                self.object_status.setText("Object: Out of range")

    def update_vision(self):
        """Обновление системы компьютерного зрения"""
        if self.visualizer.frame_buffer is not None:
            # Обработка кадра
            result = self.vision_manager.process_frame(self.visualizer.frame_buffer)
            
            if result is not None:
                # Обновление интерфейса
                self.vision_status.setText("Vision System: Active")
                self.object_detection.setText(f"Detected Objects: {len(result.objects_info)}")
                self.confidence_score.setText(f"Detection Confidence: {result.confidence:.1f}%")
                
                # Обновление статистики
                stats = self.vision_manager.get_detection_statistics()
                if stats:
                    stats_text = (f"Avg Confidence: {stats['avg_confidence']:.1f}%\n"
                                f"Avg Objects: {stats['avg_objects']:.1f}\n"
                                f"Total Frames: {stats['total_frames']}")
                    self.detection_stats.setText(stats_text)
                
                # Обновление визуализации
                visualization = self.vision_manager.visualize_current_detection(
                    self.visualizer.frame_buffer
                )
                self.update_vision_display(visualization, result)
            
            # Обновление прогресса обучения
            if not self.vision_manager.vision_system.is_trained:
                progress = len(self.vision_manager.vision_system.training_data) / \
                          self.vision_manager.vision_system.min_training_samples * 100
                self.learning_progress.setValue(int(progress))
    
    def update_vision_display(self, frame, result):
        """Обновление отображения результатов компьютерного зрения"""
        if hasattr(self, 'vision_window'):
            plt.figure(1)
            plt.clf()
            
            # Отображение исходного кадра с обнаруженными объектами
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Detection Results')
            plt.axis('off')
            
            # Отображение карты уверенности
            if result.detection_map is not None:
                plt.subplot(222)
                plt.imshow(result.detection_map, cmap='viridis')
                plt.title(f'Confidence Map ({result.confidence:.2f})')
                plt.colorbar()
                plt.axis('off')
            
            # График истории уверенности
            plt.subplot(223)
            history = self.vision_manager.vision_system.detection_confidences[-100:]
            plt.plot(history)
            plt.title('Confidence History')
            plt.xlabel('Frame')
            plt.ylabel('Confidence')
            
            # Отображение отдельных обнаруженных объектов
            plt.subplot(224)
            if result.objects_info:
                best_object = max(result.objects_info, 
                                key=lambda x: x['area'] * x['circularity'])
                x, y, w, h = best_object['bbox']
                object_roi = frame[y:y+h, x:x+w]
                plt.imshow(cv2.cvtColor(object_roi, cv2.COLOR_BGR2RGB))
                plt.title(f'Best Object (Area: {best_object["area"]:.0f})')
            else:
                plt.text(0.5, 0.5, 'No objects detected', 
                        ha='center', va='center')
            plt.axis('off')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
    
    def save_vision_model(self):
        """Сохранение модели компьютерного зрения"""
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.DirectoryOnly)
            file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
            
            if file_dialog.exec_():
                save_path = file_dialog.selectedFiles()[0]
                self.vision_manager.vision_system.save_model(save_path)
                QMessageBox.information(self, "Success", 
                                     "Vision system model saved successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", 
                              f"Failed to save vision system model: {str(e)}")
    
    def load_vision_model(self):
        """Загрузка модели компьютерного зрения"""
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.DirectoryOnly)
            file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
            
            if file_dialog.exec_():
                load_path = file_dialog.selectedFiles()[0]
                if self.vision_manager.vision_system.load_model(load_path):
                    QMessageBox.information(self, "Success", 
                                         "Vision system model loaded successfully!")
                else:
                    QMessageBox.warning(self, "Error", 
                                     "Failed to load vision system model")
        except Exception as e:
            QMessageBox.warning(self, "Error", 
                              f"Failed to load vision system model: {str(e)}")

    def train_vision_system(self):
        """Запуск обучения системы компьютерного зрения"""
        try:
            # Проверка наличия достаточного количества данных
            if len(self.vision_manager.vision_system.training_data) < self.min_samples_spin.value():
                QMessageBox.warning(self, "Warning", 
                                  "Not enough training data collected yet.")
                return
            
            # Обновление параметров
            self.vision_manager.vision_system.min_training_samples = self.min_samples_spin.value()
            self.vision_manager.vision_system.batch_size = self.batch_size_spin.value()
            
            # Запуск обучения
            self.vision_status.setText("Vision System: Training...")
            QApplication.processEvents()
            
            success = self.vision_manager.vision_system.train_model()
            
            if success:
                self.vision_status.setText("Vision System: Training completed")
                QMessageBox.information(self, "Success", 
                                     "Vision system training completed successfully!")
            else:
                self.vision_status.setText("Vision System: Training failed")
                QMessageBox.warning(self, "Error", 
                                 "Vision system training failed")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", 
                              f"Error during vision system training: {str(e)}")
    
    def reset_vision_system(self):
        """Сброс системы компьютерного зрения"""
        reply = QMessageBox.question(self, "Reset Vision System",
                                   "Are you sure you want to reset the vision system?\n"
                                   "All training data and learned parameters will be lost.",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            try:
                # Создание новой системы зрения
                self.vision_manager = VisionSystemManager()
                
                # Сброс отображения
                self.vision_status.setText("Vision System: Reset")
                self.object_detection.setText("Detected Objects: None")
                self.confidence_score.setText("Detection Confidence: 0%")
                self.learning_progress.setValue(0)
                self.detection_stats.setText("Detection Statistics: N/A")
                
                QMessageBox.information(self, "Success", 
                                     "Vision system reset successfully!")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", 
                                  f"Failed to reset vision system: {str(e)}")
    
    def toggle_gripper(self, checked):
        """Управление захватом"""
        if checked:
            if self.visualizer.check_grip_collision():
                self.visualizer.robot.payload.is_gripped = True
                self.grip_status.setText("Status: Object gripped")
                self.grip_button.setText("Release")
                # Анимация захвата
                self.visualizer.robot.gripper_state = 0.5
            else:
                self.grip_button.setChecked(False)
                self.grip_status.setText("Status: Cannot reach object")
        else:
            self.visualizer.robot.payload.is_gripped = False
            self.grip_status.setText("Status: Released")
            self.grip_button.setText("Grip")
            # Открытие захвата
            self.visualizer.robot.gripper_state = 0.0
        
        self.visualizer.update()
    
    def update_grip_force(self, value):
        """Обновление силы захвата"""
        self.grip_force_label.setText(f"Force: {value}%")
        self.visualizer.robot.grip_force = value
        
        if self.visualizer.robot.payload.is_gripped:
            # Проверка достаточности силы захвата
            if value < 30:  # Минимальная сила для удержания
                self.grip_status.setText("Warning: Low grip force")
            else:
                self.grip_status.setText("Status: Object gripped")
    
    def start_simulation(self):
        """Запуск симуляции"""
        self.timer.start()
        self.vision_timer.start()
        self.grip_status.setText("Status: Simulation running")
    
    def stop_simulation(self):
        """Остановка симуляции"""
        self.timer.stop()
        self.vision_timer.stop()
        self.grip_status.setText("Status: Simulation stopped")
    
    def reset_simulation(self):
        """Сброс симуляции"""
        # Сброс положения робота
        self.visualizer.robot.joint_angles = np.zeros(6)
        self.visualizer.robot.joint_velocities = np.zeros(6)
        self.visualizer.robot.joint_accelerations = np.zeros(6)
        
        # Сброс захвата
        self.visualizer.robot.payload.is_gripped = False
        self.visualizer.robot.gripper_state = 0.0
        self.grip_button.setChecked(False)
        self.grip_button.setText("Grip")
        
        # Сброс груза
        self.visualizer.robot.payload.position = np.array([0.8, 0.8, 0.1])
        self.visualizer.robot.payload.velocity = np.zeros(3)
        self.visualizer.robot.payload.acceleration = np.zeros(3)
        
        # Сброс слайдеров
        for slider_data in self.joint_sliders:
            slider_data['slider'].setValue(0)
            slider_data['velocity'].setText("Velocity: 0.00")
            slider_data['torque'].setText("Torque: 0.00")
        
        # Обновление статусов
        self.grip_status.setText("Status: Reset")
        self.object_status.setText("Object: Not detected")
        
        # Очистка траектории
        self.visualizer.trajectory_points.clear()
        
        self.visualizer.update()

    def emergency_stop(self):
        """Аварийная остановка"""
        # Остановка симуляции
        self.stop_simulation()
        
        # Остановка движений робота
        self.visualizer.robot.joint_velocities = np.zeros(6)
        self.visualizer.robot.joint_accelerations = np.zeros(6)
        
        # Отпускание груза
        self.visualizer.robot.payload.is_gripped = False
        self.grip_button.setChecked(False)
        
        # Обновление статусов
        self.grip_status.setText("Status: EMERGENCY STOP")
        
        # Визуальное оповещение
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Emergency Stop Activated")
        msg.setInformativeText("The robot has been stopped for safety reasons.")
        msg.setWindowTitle("Emergency Stop")
        msg.exec_()
        
        # Логирование события
        print(f"Emergency stop activated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def update_joint(self, joint_index, value):
        """Обновление положения сустава"""
        self.visualizer.robot.joint_angles[joint_index] = value
        
        # Обновление отображения скорости и момента
        velocity = self.visualizer.robot.joint_velocities[joint_index]
        torque = abs(velocity * self.visualizer.robot.joint_accelerations[joint_index])
        
        self.joint_sliders[joint_index]['velocity'].setText(f"Velocity: {velocity:.2f}")
        self.joint_sliders[joint_index]['torque'].setText(f"Torque: {torque:.2f}")
        
        self.visualizer.update()
    
    def save_system_state(self):
        """Сохранение состояния системы"""
        try:
            state = {
                'joint_angles': self.visualizer.robot.joint_angles.tolist(),
                'gripper_state': self.visualizer.robot.gripper_state,
                'camera_position': {
                    'distance': self.visualizer.camera_distance,
                    'rotation': self.visualizer.camera_rotation
                },
                'payload': {
                    'position': self.visualizer.robot.payload.position.tolist(),
                    'is_gripped': self.visualizer.robot.payload.is_gripped
                },
                'vision_system': {
                    'is_trained': self.vision_manager.vision_system.is_trained,
                    'min_samples': self.vision_manager.vision_system.min_training_samples,
                    'batch_size': self.vision_manager.vision_system.batch_size
                }
            }
            
            with open('system_state.json', 'w') as f:
                json.dump(state, f, indent=4)
                
        except Exception as e:
            print(f"Error saving system state: {str(e)}")
    
    def load_system_state(self):
        """Загрузка состояния системы"""
        try:
            if os.path.exists('system_state.json'):
                with open('system_state.json', 'r') as f:
                    state = json.load(f)
                
                # Восстановление состояния робота
                self.visualizer.robot.joint_angles = np.array(state['joint_angles'])
                self.visualizer.robot.gripper_state = state['gripper_state']
                
                # Восстановление камеры
                self.visualizer.camera_distance = state['camera_position']['distance']
                self.visualizer.camera_rotation = state['camera_position']['rotation']
                
                # Восстановление груза
                self.visualizer.robot.payload.position = np.array(state['payload']['position'])
                self.visualizer.robot.payload.is_gripped = state['payload']['is_gripped']
                
                # Восстановление параметров системы зрения
                vision_state = state['vision_system']
                self.vision_manager.vision_system.min_training_samples = vision_state['min_samples']
                self.vision_manager.vision_system.batch_size = vision_state['batch_size']
                
                # Обновление интерфейса
                self.update_ui_from_state(state)
                
        except Exception as e:
            print(f"Error loading system state: {str(e)}")
    
    def update_ui_from_state(self, state):
        """Обновление интерфейса из загруженного состояния"""
        # Обновление слайдеров суставов
        for i, angle in enumerate(state['joint_angles']):
            self.joint_sliders[i]['slider'].setValue(int(angle))
        
        # Обновление состояния захвата
        self.grip_button.setChecked(state['payload']['is_gripped'])
        if state['payload']['is_gripped']:
            self.grip_button.setText("Release")
            self.grip_status.setText("Status: Object gripped")
        else:
            self.grip_button.setText("Grip")
            self.grip_status.setText("Status: Ready")
        
        # Обновление параметров обучения
        self.min_samples_spin.setValue(state['vision_system']['min_samples'])
        self.batch_size_spin.setValue(state['vision_system']['batch_size'])
        
        self.visualizer.update()
    
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        try:
            reply = QMessageBox.question(self, 'Quit',
                                       'Save system state before closing?',
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
                
            if reply == QMessageBox.Yes:
                self.save_system_state()
            
            # Остановка таймеров
            self.timer.stop()
            self.vision_timer.stop()
            
            # Закрытие окон визуализации
            plt.close('all')
            
            event.accept()
            
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            event.accept()

def main():
    """Точка входа в программу"""
    try:
        app = QApplication(sys.argv)
        
        # Установка стиля приложения
        app.setStyle('Fusion')
        
        # Создание темной палитры
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        app.setPalette(dark_palette)
        
        # Создание и отображение главного окна
        window = MainWindow()
        window.show()
        
        # Запуск цикла событий
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()