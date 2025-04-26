#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object detection module.

This module uses YOLO models to detect vehicles and license plates.
"""

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """Class that uses a YOLO model for vehicle detection."""
    
    # Vehicle classes in the COCO dataset
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    def __init__(self, model_path="models/yolov8n.pt"):
        """
        Loads the YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        """
        Detects vehicles in a video frame.
        
        Args:
            frame (numpy.ndarray): Video frame to be processed.
            
        Returns:
            list: Bounding boxes of detected vehicles.
        """
        detections = self.model(frame)[0]
        results = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.VEHICLE_CLASSES:
                results.append([x1, y1, x2, y2, score])
                
        return results
    
    @staticmethod
    def assign_plate_to_vehicle(license_plate, vehicles):
        """
        Assigns a license plate to the most appropriate vehicle.
        
        Args:
            license_plate (list): License plate bounding box [x1, y1, x2, y2, score].
            vehicles (list): List of tracked vehicles.
            
        Returns:
            list or None: Matching vehicle information or None if no match found.
        """
        x1, y1, x2, y2 = license_plate[:4]
        
        for vehicle in vehicles:
            xv1, yv1, xv2, yv2, vehicle_id = vehicle
            
            # Check if the license plate box is within the vehicle box
            if x1 > xv1 and y1 > yv1 and x2 < xv2 and y2 < yv2:
                return vehicle
                
        return None


class LicensePlateDetector:
    """Class that uses a custom-trained YOLO model for license plate detection."""
    
    def __init__(self, model_path="models/license_plate_detector.pt"):
        """
        Loads the YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        """
        Detects license plates in a video frame.
        
        Args:
            frame (numpy.ndarray): Video frame to be processed.
            
        Returns:
            list: Bounding boxes of detected license plates.
        """
        detections = self.model(frame)[0]
        results = []
        
        for detection in detections.boxes.data.tolist():
            results.append(detection)
                
        return results
