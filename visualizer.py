#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module.

This module visualizes detection and tracking results on a video.
"""

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def visualize_results(video_path, csv_path, output_path):
    """
    Visualizes detection and tracking results on a video.
    
    Args:
        video_path (str): Path to the input video file.
        csv_path (str): CSV file containing interpolated results.
        output_path (str): Path to the output video file.
    """
    # Read the CSV file
    results = pd.read_csv(csv_path)
    
    # Initialize video reader
    cap = cv2.VideoCapture(str(video_path))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Store the best license plate images and numbers
    license_plates = {}
    
    # Find the best license plate image for each car
    for car_id in results['car_id'].unique():
        car_results = results[results['car_id'] == car_id]
        
        # Find the frame with the highest license plate score
        best_idx = car_results['license_number_score'].idxmax()
        best_frame = car_results.loc[best_idx]
        
        # Save the license number
        license_number = best_frame['license_number']
        
        # Go to the frame number and capture the plate image
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame['frame_nmr'])
        ret, frame = cap.read()
        
        if ret:
            # Get the license plate bounding box
            x1, y1, x2, y2 = parse_bbox(best_frame['license_plate_bbox'])
            
            # Crop the license plate image
            license_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Resize the plate image
            aspect_ratio = (x2 - x1) / (y2 - y1)
            license_crop = cv2.resize(license_crop, (int(400 * aspect_ratio), 400))
            
            # Save the cropped plate
            license_plates[car_id] = {
                'license_crop': license_crop,
                'license_number': license_number
            }
    
    # Reset the video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process each frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_nmr in tqdm(range(total_frames), desc="Visualizing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find all cars in this frame
        frame_results = results[results['frame_nmr'] == frame_nmr]
        
        # For each car
        for _, row in frame_results.iterrows():
            car_id = row['car_id']
            
            # Draw the car bounding box
            car_bbox = parse_bbox(row['car_bbox'])
            x1, y1, x2, y2 = [int(coord) for coord in car_bbox]
            draw_border(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            
            # Draw the license plate bounding box
            plate_bbox = parse_bbox(row['license_plate_bbox'])
            px1, py1, px2, py2 = [int(coord) for coord in plate_bbox]
            
            # If license plate information is available, show the plate and number
            if car_id in license_plates:
                license_crop = license_plates[car_id]['license_crop']
                license_number = license_plates[car_id]['license_number']
                
                # Get the dimensions of the license plate image
                h, w, _ = license_crop.shape
                
                try:
                    # Place the license plate image
                    frame[int(y1) - h - 100:int(y1) - 100,
                          int((x2 + x1 - w) / 2):int((x2 + x1 + w) / 2), :] = license_crop
                    
                    # Create a white background for the license number
                    frame[int(y1) - h - 400:int(y1) - h - 100,
                          int((x2 + x1 - w) / 2):int((x2 + x1 + w) / 2), :] = (255, 255, 255)
                    
                    # Write the license number
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_number, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                    
                    cv2.putText(frame,
                                license_number,
                                (int((x2 + x1 - text_width) / 2), int(y1 - h - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)
                except Exception as e:
                    print(f"Visualization error: {e}")
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video created: {output_path}")            



def parse_bbox(bbox_str):
    """
    Converts a bounding box string representation to a list format.
    
    Args:
        bbox_str (str): A string in the format "[x1 y1 x2 y2]".
        
    Returns:
        list: A list in the format [x1, y1, x2, y2].
    """
    # Clean the string from "[" and "]"
    bbox_str = bbox_str.replace('[', '').replace(']', '')
    
    # Standardize spaces
    bbox_str = ' '.join(bbox_str.split())
    
    # Split and convert to float
    coords = [float(coord) for coord in bbox_str.split(' ')]
    
    return coords


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, line_length_x=50, line_length_y=50):
    """
    Draws a bordered frame with corner markers on the image.
    
    Args:
        img (numpy.ndarray): Image.
        top_left (tuple): Top-left corner coordinates (x1, y1).
        bottom_right (tuple): Bottom-right corner coordinates (x2, y2).
        color (tuple): BGR color value.
        thickness (int): Line thickness.
        line_length_x (int): Horizontal line length.
        line_length_y (int): Vertical line length.
        
    Returns:
        numpy.ndarray: Image with drawn frame.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Top-left corner
    img = cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    img = cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    
    # Top-right corner
    img = cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    img = cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    
    # Bottom-left corner
    img = cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    img = cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    
    # Bottom-right corner
    img = cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    img = cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    
    return img
