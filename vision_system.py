# vision_system.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import os
from datetime import datetime

class CustomVisionDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image

class EnhancedAutoEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=8):
        super(EnhancedAutoEncoder, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, latent_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class EnhancedVisionSystem:
    def __init__(self, model_path=None):
        self.model = EnhancedAutoEncoder()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        
        self.training_data = []
        self.validation_data = []
        self.is_trained = False
        self.min_training_samples = 1000
        self.batch_size = 64
        
        # Метрики для отслеживания
        self.training_losses = []
        self.validation_losses = []
        self.detection_confidences = []
        self.learning_rates = []
        
        # Параметры аугментации
        self.augmentation_params = {
            'noise_factor': 0.1,
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'rotation_range': (-10, 10)
        }
        
        # Загрузка модели если путь указан
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def save_model(self, path="vision_model"):
        """Сохранение модели и параметров"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{path}_{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        
        # Сохранение модели
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'detection_confidences': self.detection_confidences,
            'learning_rates': self.learning_rates
        }, os.path.join(save_path, 'model.pth'))
        
        # Сохранение скейлера и кластеризатора
        np.save(os.path.join(save_path, 'scaler.npy'), {
            'mean_': self.scaler.mean_,
            'scale_': self.scaler.scale_,
            'var_': self.scaler.var_
        })
        
        np.save(os.path.join(save_path, 'kmeans.npy'), {
            'cluster_centers_': self.kmeans.cluster_centers_,
            'labels_': self.kmeans.labels_
        })
        
        # Сохранение конфигурации
        config = {
            'is_trained': self.is_trained,
            'min_training_samples': self.min_training_samples,
            'batch_size': self.batch_size,
            'augmentation_params': self.augmentation_params
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        print(f"Model saved to {save_path}")
        
    def load_model(self, path):
        """Загрузка модели и параметров"""
        try:
            # Загрузка модели
            checkpoint = torch.load(os.path.join(path, 'model.pth'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.training_losses = checkpoint['training_losses']
            self.validation_losses = checkpoint['validation_losses']
            self.detection_confidences = checkpoint['detection_confidences']
            self.learning_rates = checkpoint['learning_rates']
            
            # Загрузка скейлера
            scaler_data = np.load(os.path.join(path, 'scaler.npy'), allow_pickle=True).item()
            self.scaler.mean_ = scaler_data['mean_']
            self.scaler.scale_ = scaler_data['scale_']
            self.scaler.var_ = scaler_data['var_']
            
            # Загрузка кластеризатора
            kmeans_data = np.load(os.path.join(path, 'kmeans.npy'), allow_pickle=True).item()
            self.kmeans.cluster_centers_ = kmeans_data['cluster_centers_']
            self.kmeans.labels_ = kmeans_data['labels_']
            
            # Загрузка конфигурации
            with open(os.path.join(path, 'config.json'), 'r') as f:
                config = json.load(f)
                
            self.is_trained = config['is_trained']
            self.min_training_samples = config['min_training_samples']
            self.batch_size = config['batch_size']
            self.augmentation_params = config['augmentation_params']
            
            print(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    def augment_data(self, image):
        """Расширенная аугментация данных"""
        augmented = image.copy()
        
        # Добавление шума
        noise = np.random.normal(0, self.augmentation_params['noise_factor'], 
                               augmented.shape)
        augmented = np.clip(augmented + noise, 0, 255)
        
        # Изменение яркости
        brightness = np.random.uniform(*self.augmentation_params['brightness_range'])
        augmented = np.clip(augmented * brightness, 0, 255)
        
        # Изменение контраста
        contrast = np.random.uniform(*self.augmentation_params['contrast_range'])
        augmented = np.clip((augmented - 128) * contrast + 128, 0, 255)
        
        # Поворот изображения
        angle = np.random.uniform(*self.augmentation_params['rotation_range'])
        rows, cols = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (cols, rows))
        
        return augmented
        
    def prepare_data(self, frame):
        """Подготовка данных для обработки"""
        # Преобразование в RGB если нужно
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        # Изменение размера для консистентности
        frame = cv2.resize(frame, (224, 224))
        
        # Нормализация
        frame = frame.astype(np.float32) / 255.0
        
        return frame
        
    def train_model(self):
        """Улучшенное обучение модели"""
        if len(self.training_data) < self.min_training_samples:
            print("Not enough training data")
            return False
            
        try:
            print("Starting vision system training...")
            
            # Разделение данных на обучающую и валидационную выборки
            split_idx = int(len(self.training_data) * 0.8)
            train_data = self.training_data[:split_idx]
            val_data = self.training_data[split_idx:]
            
            # Создание даталоадеров
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            train_dataset = CustomVisionDataset(train_data, transform)
            val_dataset = CustomVisionDataset(val_data, transform)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                    shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            # Обучение
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):
                # Обучение
                self.model.train()
                train_loss = 0
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    
                avg_train_loss = train_loss / len(train_loader)
                self.training_losses.append(avg_train_loss)
                
                # Валидация
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        output = self.model(batch)
                        loss = self.criterion(output, batch)
                        val_loss += loss.item()
                        
                avg_val_loss = val_loss / len(val_loader)
                self.validation_losses.append(avg_val_loss)
                
                # Обновление learning rate
                self.scheduler.step(avg_val_loss)
                self.learning_rates.append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/100]')
                    print(f'Train Loss: {avg_train_loss:.4f}')
                    print(f'Val Loss: {avg_val_loss:.4f}')
                    print(f'LR: {self.learning_rates[-1]:.6f}')
            
            # Обучение K-means на закодированных представлениях
            self.model.eval()
            encoded_data = []
            with torch.no_grad():
                for batch in train_loader:
                    encoded = self.model.encode(batch)
                    encoded_data.append(encoded.numpy())
                    
            encoded_data = np.vstack(encoded_data)
            self.kmeans.fit(encoded_data)
            
            self.is_trained = True
            print("Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
            
    def process_frame(self, frame):
        """Обработка кадра с расширенной функциональностью"""
        try:
            # Подготовка кадра
            processed_frame = self.prepare_data(frame)
            
            # Сбор данных для обучения
            if not self.is_trained:
                if len(self.training_data) < self.min_training_samples:
                    # Аугментация данных
                    augmented_frame = self.augment_data(processed_frame)
                    self.training_data.append(processed_frame)
                    self.training_data.append(augmented_frame)
                    return None, 0.0
                elif len(self.training_data) == self.min_training_samples:
                    print(f"Collected {len(self.training_data)} frames, starting training...")
                    self.train_model()
            
            # Обработка обученной системой
            if self.is_trained:
                # Преобразование для обработки
                pixels = processed_frame.reshape(-1, 3)
                pixels_tensor = torch.FloatTensor(pixels)
                
                # Обработка по батчам
                encoded_batches = []
                with torch.no_grad():
                    for i in range(0, len(pixels_tensor), self.batch_size):
                        batch = pixels_tensor[i:i+self.batch_size]
                        encoded = self.model.encode(batch)
                        encoded_batches.append(encoded.numpy())
                
                encoded_data = np.vstack(encoded_batches)
                
                # Применение кластеризации
                clusters = self.kmeans.predict(encoded_data)
                
                # Реконструкция изображения
                result_image = clusters.reshape(processed_frame.shape[:2])
                
                # Расчет уверенности определения
                distances = self.kmeans.transform(encoded_data)
                confidence = np.exp(-np.min(distances, axis=1)).mean()
                self.detection_confidences.append(confidence)
                
                # Пост-обработка результатов
                result_image = self.post_process_detection(result_image)
                
                return result_image, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return None, 0.0
            
    def post_process_detection(self, detection_map):
        """Пост-обработка результатов детекции"""
        # Морфологические операции для улучшения результатов
        kernel = np.ones((5,5), np.uint8)
        detection_map = cv2.morphologyEx(detection_map.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        detection_map = cv2.morphologyEx(detection_map, 
                                       cv2.MORPH_OPEN, kernel)
        
        # Удаление мелких объектов
        n_components, output, stats, centroids = \
            cv2.connectedComponentsWithStats(detection_map, connectivity=8)
            
        sizes = stats[1:, -1]
        n_components = n_components - 1
        min_size = 100  # минимальный размер объекта
        
        cleaned_map = np.zeros((output.shape))
        for i in range(0, n_components):
            if sizes[i] >= min_size:
                cleaned_map[output == i + 1] = 1
                
        return cleaned_map
        
    def get_object_info(self, detection_map):
        """Получение информации об обнаруженных объектах"""
        # Поиск контуров объектов
        contours, _ = cv2.findContours(detection_map.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        objects_info = []
        for contour in contours:
            # Расчет характеристик объекта
            area = cv2.contourArea(contour)
            if area < 100:  # игнорируем слишком маленькие объекты
                continue
                
            # Получение ограничивающего прямоугольника
            x, y, w, h = cv2.boundingRect(contour)
            
            # Расчет центра масс
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
                
            # Определение формы объекта
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            objects_info.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'circularity': circularity,
                'contour': contour
            })
            
        return objects_info
        
    def visualize_detection(self, frame, detection_map, objects_info):
        """Визуализация результатов детекции"""
        vis_frame = frame.copy()
        
        # Наложение маски обнаружения
        mask_overlay = np.zeros_like(vis_frame)
        mask_overlay[detection_map > 0] = [0, 255, 0]  # зеленый цвет для объектов
        vis_frame = cv2.addWeighted(vis_frame, 1, mask_overlay, 0.3, 0)
        
        # Отрисовка информации об объектах
        for obj in objects_info:
            # Ограничивающий прямоугольник
            x, y, w, h = obj['bbox']
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Центр масс
            cx, cy = obj['center']
            cv2.circle(vis_frame, (cx, cy), 4, (255, 0, 0), -1)
            
            # Информация о характеристиках
            text = f"Area: {int(obj['area'])} Circ: {obj['circularity']:.2f}"
            cv2.putText(vis_frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        return vis_frame

class DetectionResult:
    """Класс для хранения результатов детекции"""
    def __init__(self, detection_map=None, confidence=0.0, objects_info=None):
        self.detection_map = detection_map
        self.confidence = confidence
        self.objects_info = objects_info if objects_info is not None else []
        self.timestamp = datetime.now()
        
    def get_best_object(self):
        """Получение наиболее подходящего объекта для захвата"""
        if not self.objects_info:
            return None
            
        # Выбор объекта с наибольшей площадью и круглостью
        best_object = max(self.objects_info, 
                         key=lambda x: x['area'] * x['circularity'])
        return best_object

class VisionSystemManager:
    """Менеджер системы компьютерного зрения"""
    def __init__(self, model_path=None):
        self.vision_system = EnhancedVisionSystem(model_path)
        self.current_result = None
        self.detection_history = []
        self.max_history = 100
        
    def process_frame(self, frame):
        """Обработка кадра с сохранением истории"""
        detection_map, confidence = self.vision_system.process_frame(frame)
        
        if detection_map is not None:
            objects_info = self.vision_system.get_object_info(detection_map)
            result = DetectionResult(detection_map, confidence, objects_info)
            
            self.current_result = result
            self.detection_history.append(result)
            
            # Ограничение размера истории
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
                
            return result
        return None
        
    def get_detection_statistics(self):
        """Получение статистики детекции"""
        if not self.detection_history:
            return None
            
        confidences = [r.confidence for r in self.detection_history]
        object_counts = [len(r.objects_info) for r in self.detection_history]
        
        return {
            'avg_confidence': np.mean(confidences),
            'avg_objects': np.mean(object_counts),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'total_frames': len(self.detection_history)
        }
        
    def visualize_current_detection(self, frame):
        """Визуализация текущей детекции"""
        if self.current_result is None:
            return frame
            
        return self.vision_system.visualize_detection(
            frame,
            self.current_result.detection_map,
            self.current_result.objects_info
        )