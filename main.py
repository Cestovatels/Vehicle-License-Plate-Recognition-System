"""
Vehicle License Plate Recognition System Main Program

This module detects vehicles from video images, recognizes their license plates and visualizes the results.
"""

import os
import argparse
from pathlib import Path

from detector import VehicleDetector, LicensePlateDetector
from tracker import VehicleTracker
from ocr import LicensePlateReader
from utils import write_results_to_csv
from interpolator import interpolate_missing_frames
from visualizer import visualize_results


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Vehicle License Plate Recognition System")
    parser.add_argument("--video_path", type=str, default="data/sample.mp4", help="Path to the video file to be processed")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--vehicle_model", type=str, default="models/yolov8n.pt", help="YOLO model for vehicle detection")
    parser.add_argument("--plate_model", type=str, default="models/license_plate_detector.pt", help="YOLO model for license plate detection")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization")
    return parser.parse_args()


def main():
    """Manages the main program flow."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare the pathways
    video_path = Path(args.video_path)
    results_csv_path = Path(args.output_dir) / "results.csv"
    interpolated_csv_path = Path(args.output_dir) / "results_interpolated.csv"
    output_video_path = Path(args.output_dir) / f"{video_path.stem}_output.mp4"
    
    # Initialize models and handlers
    vehicle_detector = VehicleDetector(model_path=args.vehicle_model)
    plate_detector = LicensePlateDetector(model_path=args.plate_model)
    vehicle_tracker = VehicleTracker()
    plate_reader = LicensePlateReader()
    
    print(f"Video processing: {video_path}")
    
    # Identify vehicles and license plates
    results = process_video(video_path, vehicle_detector, plate_detector, 
                           vehicle_tracker, plate_reader)
    
    # Write results to CSV
    write_results_to_csv(results, results_csv_path)
    print(f"Raw results were recorded: {results_csv_path}")
    
   # Fill missing frames with interpolation
    interpolate_missing_frames(results_csv_path, interpolated_csv_path)
    print(f"Interpolation is complete: {interpolated_csv_path}")
    
    # Visualization
    if not args.skip_visualization:
        visualize_results(video_path, interpolated_csv_path, output_video_path)
        print(f"Visualization completed: {output_video_path}")
    
    print("The process is complete!")


def process_video(video_path, vehicle_detector, plate_detector, vehicle_tracker, plate_reader):
    """Performs operations on video frames and sums the results."""
    from tqdm import tqdm
    import cv2
    
    results = {}
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_nmr in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        results[frame_nmr] = {}
        
        # Detect vehicles
        vehicle_detections = vehicle_detector.detect(frame)
        
        # Follow the vehicles
        tracked_vehicles = vehicle_tracker.update(vehicle_detections)
        
        # Detect license plates
        license_plate_detections = plate_detector.detect(frame)
        
        # For each license plate
        for license_plate in license_plate_detections:
            # Put the license plate on the car
            vehicle = vehicle_detector.assign_plate_to_vehicle(license_plate, tracked_vehicles)
            
            if vehicle is not None:
                # Read the license plate
                x1, y1, x2, y2 = license_plate[:4]
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                plate_text, confidence = plate_reader.read_plate(plate_crop)
                
                if plate_text:
                    vehicle_id = vehicle[4]  # Vehicle ID
                    results[frame_nmr][vehicle_id] = {
                        'car': {'bbox': vehicle[:4]},
                        'license_plate': {
                            'bbox': license_plate[:4],
                            'bbox_score': license_plate[4],
                            'text': plate_text,
                            'text_score': confidence
                        }
                    }
    
    cap.release()
    return results


if __name__ == "__main__":
    main()