#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper Functions Module.

This module contains helper functions used throughout the project.
"""

def write_results_to_csv(results, output_path):
    """
    Writes detection and tracking results to a CSV file.
    
    Args:
        results (dict): Dictionary of result data.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        # Write CSV header row
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 
                                                'license_number', 'license_number_score'))

        # Write results for each frame
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                car_data = results[frame_nmr][car_id]
                
                # Check if necessary data is available
                if 'car' in car_data and 'license_plate' in car_data and 'text' in car_data['license_plate']:
                    # Write CSV row
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*car_data['car']['bbox']),
                        '[{} {} {} {}]'.format(*car_data['license_plate']['bbox']),
                        car_data['license_plate']['bbox_score'],
                        car_data['license_plate']['text'],
                        car_data['license_plate']['text_score']
                    ))


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


def format_bbox(bbox):
    """
    Converts a bounding box list to string format.
    
    Args:
        bbox (list): A list in the format [x1, y1, x2, y2].
        
    Returns:
        str: A string in the format "[x1 y1 x2 y2]".
    """
    return f"[{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}]"


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
