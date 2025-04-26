#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolation module for missing frames.

This module fills in missing frames for detected vehicles and license plates using linear interpolation.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def interpolate_missing_frames(input_csv, output_csv):
    """
    Reads detection results from a CSV file and fills missing frames using interpolation.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    # Read the CSV file
    data = pd.read_csv(input_csv)
    
    # Find unique car IDs
    unique_car_ids = data['car_id'].unique()
    
    # Create the result DataFrame
    interpolated_data = []
    
    for car_id in unique_car_ids:
        # Filter data for the current car
        car_data = data[data['car_id'] == car_id]
        car_frames = car_data['frame_nmr'].values
        
        # Get the first and last frame numbers
        first_frame = car_frames.min()
        last_frame = car_frames.max()
        
        # Get vehicle and license plate bounding boxes
        car_bboxes = np.array([parse_bbox(bbox) for bbox in car_data['car_bbox'].values])
        plate_bboxes = np.array([parse_bbox(bbox) for bbox in car_data['license_plate_bbox'].values])
        
        # Interpolate for each frame
        for frame in range(first_frame, last_frame + 1):
            # If the frame exists in original data, use its values
            if frame in car_frames:
                frame_data = car_data[car_data['frame_nmr'] == frame].iloc[0]
                row = {
                    'frame_nmr': frame,
                    'car_id': car_id,
                    'car_bbox': frame_data['car_bbox'],
                    'license_plate_bbox': frame_data['license_plate_bbox'],
                    'license_plate_bbox_score': frame_data['license_plate_bbox_score'],
                    'license_number': frame_data['license_number'],
                    'license_number_score': frame_data['license_number_score']
                }
            # If the frame is missing, interpolate
            else:
                # Interpolation for car bounding box
                car_bbox = interpolate_bbox(car_frames, car_bboxes, frame)
                
                # Interpolation for license plate bounding box
                plate_bbox = interpolate_bbox(car_frames, plate_bboxes, frame)
                
                row = {
                    'frame_nmr': frame,
                    'car_id': car_id,
                    'car_bbox': format_bbox(car_bbox),
                    'license_plate_bbox': format_bbox(plate_bbox),
                    'license_plate_bbox_score': 0,
                    'license_number': get_best_license_number(car_data),
                    'license_number_score': 0
                }
                
            interpolated_data.append(row)
    
    # Create a DataFrame and save it to CSV
    df = pd.DataFrame(interpolated_data)
    df.to_csv(output_csv, index=False)


def parse_bbox(bbox_str):
    """
    Converts a bounding box string to a list of numbers.
    
    Args:
        bbox_str (str): Bounding box in string format.
        
    Returns:
        list: List of numbers in [x1, y1, x2, y2] format.
    """
    # Clean and parse "[x1 y1 x2 y2]" format
    cleaned = bbox_str.replace('[', '').replace(']', '').replace('  ', ' ')
    return [float(x) for x in cleaned.split()]


def format_bbox(bbox):
    """
    Converts a bounding box list to string format.
    
    Args:
        bbox (list): Bounding box in [x1, y1, x2, y2] format.
        
    Returns:
        str: Bounding box as a string "[x1 y1 x2 y2]".
    """
    return f"[{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}]"


def interpolate_bbox(frames, bboxes, target_frame):
    """
    Calculates the interpolated bounding box for a given frame.
    
    Args:
        frames (numpy.ndarray): Existing frame numbers.
        bboxes (numpy.ndarray): Existing bounding boxes.
        target_frame (int): Target frame number.
        
    Returns:
        numpy.ndarray: Interpolated bounding box.
    """
    # Create a separate interpolation function for each coordinate
    interp_funcs = [interp1d(frames, bboxes[:, i], kind='linear', fill_value="extrapolate") 
                    for i in range(4)]
    
    # Calculate coordinates for the target frame
    return np.array([interp_func(target_frame) for interp_func in interp_funcs])


def get_best_license_number(car_data):
    """
    Returns the license number with the highest confidence score.
    
    Args:
        car_data (pandas.DataFrame): Car data.
        
    Returns:
        str: Best license number.
    """
    # Select the license number with the highest confidence score
    best_idx = car_data['license_number_score'].idxmax()
    return car_data.loc[best_idx, 'license_number']
