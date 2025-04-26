#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vehicle Tracking Module.

This module tracks vehicles using the SORT algorithm.
"""

import numpy as np
from sort.sort import Sort


class VehicleTracker:
    """Vehicle tracking class using the SORT algorithm."""
    
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initializes the SORT tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep a lost object before considering it disappeared.
            min_hits (int): Minimum number of consecutive detections before considering a track as valid.
            iou_threshold (float): Intersection-over-Union (IoU) threshold for association.
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        
    def update(self, detections):
        """
        Updates the tracker based on the detections in the current frame.
        
        Args:
            detections (list): Bounding boxes of detected vehicles.
            
        Returns:
            numpy.ndarray: Updated tracking results.
        """
        if not detections:
            return []
            
        return self.tracker.update(np.asarray(detections))
