#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License Plate Recognition Module.

This module reads and validates license plate texts using EasyOCR.
"""

import string
import cv2
import easyocr


class LicensePlateReader:
    """Class for reading license plates using EasyOCR."""
    
    # Mappings for character-to-digit and digit-to-character conversions
    CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    INT_TO_CHAR = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
    
    def __init__(self, lang=['en'], gpu=False):
        """
        Initializes the EasyOCR reader.
        
        Args:
            lang (list): List of languages to use.
            gpu (bool): Whether to use GPU or not.
        """
        self.reader = easyocr.Reader(lang, gpu=gpu)
        
    def read_plate(self, plate_img):
        """
        Reads text from a license plate image.
        
        Args:
            plate_img (numpy.ndarray): License plate image.
            
        Returns:
            tuple: (plate_text, confidence_score) or (None, None).
        """
        # Preprocess the image
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        
        # Read text with OCR
        detections = self.reader.readtext(thresh)
        
        # Analyze results
        for bbox, text, score in detections:
            text = text.upper().replace(' ', '')
            
            if self._is_valid_plate_format(text):
                return self._format_plate(text), score
                
        return None, None
        
    def _is_valid_plate_format(self, text):
        """
        Checks if the text is in a valid license plate format.
        
        Args:
            text (str): Text to check.
            
        Returns:
            bool: True if text is a valid plate format, False otherwise.
        """
        if len(text) != 7:
            return False
            
        # First two characters must be letters
        if not ((text[0] in string.ascii_uppercase or text[0] in self.INT_TO_CHAR) and
                (text[1] in string.ascii_uppercase or text[1] in self.INT_TO_CHAR)):
            return False
            
        # Next two characters must be digits
        if not ((text[2] in string.digits or text[2] in self.CHAR_TO_INT) and
                (text[3] in string.digits or text[3] in self.CHAR_TO_INT)):
            return False
            
        # Last three characters must be letters
        if not ((text[4] in string.ascii_uppercase or text[4] in self.INT_TO_CHAR) and
                (text[5] in string.ascii_uppercase or text[5] in self.INT_TO_CHAR) and
                (text[6] in string.ascii_uppercase or text[6] in self.INT_TO_CHAR)):
            return False
                
        return True
        
    def _format_plate(self, text):
        """
        Converts license plate text into the correct format.
        
        Args:
            text (str): Text to format.
            
        Returns:
            str: Formatted license plate text.
        """
        formatted_text = ''
        
        # Map according to character position
        mapping = {
            0: self.INT_TO_CHAR, 1: self.INT_TO_CHAR,  # First two characters: digit -> letter
            2: self.CHAR_TO_INT, 3: self.CHAR_TO_INT,  # Middle two characters: letter -> digit
            4: self.INT_TO_CHAR, 5: self.INT_TO_CHAR, 6: self.INT_TO_CHAR  # Last three characters: digit -> letter
        }
        
        for i in range(7):
            if text[i] in mapping[i]:
                formatted_text += mapping[i][text[i]]
            else:
                formatted_text += text[i]
                
        return formatted_text
