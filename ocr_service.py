# ocr_service.py

import cv2
import numpy as np
from PIL import Image
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only import PaddleOCR
try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
    logger.info("PaddleOCR is available")
except ImportError as e:
    logger.error(f"PaddleOCR not available: {e}")
    PADDLEOCR_AVAILABLE = False

class InsuranceFormROI:
    """Define ROI coordinates for different insurance form fields"""
    
    # Insurance form fields with typical positions (normalized 0-1)
    FIELD_DEFINITIONS = {
        'policy_number': {
            'roi': (0.1, 0.05, 0.4, 0.15),  # x, y, w, h (normalized)
            'type': 'alphanumeric'
        },
        'cert_number': {
            'roi': (0.5, 0.05, 0.4, 0.15),
            'type': 'alphanumeric'
        },
        'name': {
            'roi': (0.1, 0.15, 0.8, 0.1),
            'type': 'text'
        },
        'hkid': {
            'roi': (0.1, 0.25, 0.3, 0.08),
            'type': 'hkid_pattern'
        },
        'name_of_policyowner': {
            'roi': (0.1, 0.35, 0.8, 0.08),
            'type': 'text'
        },
        'employee_member': {
            'roi': (0.1, 0.43, 0.4, 0.08),
            'type': 'text'
        },
        'name_of_insured': {
            'roi': (0.1, 0.51, 0.8, 0.08),
            'type': 'text'
        },
        'occupation': {
            'roi': (0.1, 0.59, 0.4, 0.08),
            'type': 'text'
        },
        'passport_number': {
            'roi': (0.5, 0.25, 0.4, 0.08),
            'type': 'passport_pattern'
        },
        'date_of_birth': {
            'roi': (0.1, 0.67, 0.25, 0.08),
            'type': 'date'
        },
        'sex': {
            'roi': (0.4, 0.67, 0.1, 0.08),
            'type': 'gender'
        },
        'doctors_name': {
            'roi': (0.1, 0.75, 0.6, 0.08),
            'type': 'text'
        },
        'address': {
            'roi': (0.1, 0.83, 0.8, 0.12),
            'type': 'text'
        },
        'treatment_date': {
            'roi': (0.7, 0.75, 0.25, 0.08),
            'type': 'date'
        },
        'account_holder_name': {
            'roi': (0.1, 0.15, 0.4, 0.08),
            'type': 'text'
        },
        'currency': {
            'roi': (0.1, 0.23, 0.15, 0.08),
            'type': 'currency'
        },
        'bank_name': {
            'roi': (0.25, 0.23, 0.35, 0.08),
            'type': 'text'
        },
        'hkd_bank_name': {
            'roi': (0.6, 0.23, 0.35, 0.08),
            'type': 'text'
        },
        'bank_no': {
            'roi': (0.1, 0.31, 0.15, 0.08),
            'type': 'numeric'
        },
        'branch_no': {
            'roi': (0.25, 0.31, 0.15, 0.08),
            'type': 'numeric'
        },
        'bank_account_no': {
            'roi': (0.4, 0.31, 0.25, 0.08),
            'type': 'numeric'
        },
        'patient_name': {
            'roi': (0.1, 0.05, 0.4, 0.08),
            'type': 'text'
        },
        'date_of_admission': {
            'roi': (0.1, 0.13, 0.2, 0.08),
            'type': 'date'
        },
        'date_of_discharge': {
            'roi': (0.35, 0.13, 0.2, 0.08),
            'type': 'date'
        },
        'consultation_date': {
            'roi': (0.6, 0.13, 0.2, 0.08),
            'type': 'date'
        },
        'symptoms_duration': {
            'roi': (0.1, 0.25, 0.8, 0.15),
            'type': 'text'
        },
        'symptoms_complaints': {
            'roi': (0.1, 0.4, 0.8, 0.2),
            'type': 'text'
        },
        'hospitalization_details': {
            'roi': (0.1, 0.6, 0.8, 0.15),
            'type': 'text'
        },
        'final_diagnosis': {
            'roi': (0.1, 0.75, 0.8, 0.1),
            'type': 'text'
        },
        'operation_date': {
            'roi': (0.1, 0.85, 0.2, 0.08),
            'type': 'date'
        },
        'operation_procedure': {
            'roi': (0.35, 0.85, 0.6, 0.08),
            'type': 'text'
        }
    }

class OCRService:
    def __init__(self):
        """Initialize OCR service with PaddleOCR only"""
        self.confidence_threshold = 0.5  # Single threshold for PaddleOCR
        
        # Initialize PaddleOCR lazily
        self._paddleocr_reader = None
        self.roi_detector = InsuranceFormROI()
        
        # Check if PaddleOCR is available
        if not PADDLEOCR_AVAILABLE:
            logger.error("PaddleOCR not available! Please install paddleocr")
            raise ImportError("PaddleOCR not available")
        
        logger.info("OCR Service initialized with PaddleOCR only")
        
    @property
    def paddleocr_reader(self):
        """Lazy initialization of PaddleOCR"""
        if self._paddleocr_reader is None and PADDLEOCR_AVAILABLE:
            try:
                logger.info("Initializing PaddleOCR...")
                self._paddleocr_reader = paddleocr.PaddleOCR(
                    use_angle_cls=True, 
                    lang='en', 
                    show_log=False,
                    use_gpu=False  # Set to True if you have GPU support
                )
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self._paddleocr_reader = None
        return self._paddleocr_reader

    def extract_roi(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract Region of Interest from image"""
        try:
            h, w = image.shape[:2]
            x, y, roi_w, roi_h = roi
            
            # Convert normalized coordinates to pixels
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + roi_w) * w)
            y2 = int((y + roi_h) * h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            return image[y1:y2, x1:x2]
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return np.array([])

    def preprocess_roi(self, roi_image: np.ndarray, field_type: str) -> np.ndarray:
        """Enhanced preprocessing for specific field types"""
        try:
            if roi_image.size == 0:
                return roi_image
                
            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image.copy()
            
            # Apply field-specific preprocessing
            if field_type in ['text', 'alphanumeric']:
                # Enhanced preprocessing for text
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                # Morphological operations to connect broken characters
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
            elif field_type in ['numeric', 'date']:
                # Sharp thresholding for printed numbers/dates
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif field_type == 'hkid_pattern':
                # Special processing for HKID format
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                _, processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            else:
                # Default processing
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            
            return processed
        except Exception as e:
            logger.error(f"Error in preprocessing ROI: {e}")
            return roi_image

    def extract_with_paddleocr(self, roi_image: np.ndarray) -> Tuple[str, float]:
        """Extract text from ROI using PaddleOCR"""
        if not PADDLEOCR_AVAILABLE or self.paddleocr_reader is None:
            return "", 0.0
            
        try:
            # Convert grayscale back to RGB for PaddleOCR
            if len(roi_image.shape) == 2:
                roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
            else:
                roi_image_rgb = roi_image
                
            results = self.paddleocr_reader.ocr(roi_image_rgb, cls=True)
            
            if not results or not results[0]:
                return "", 0.0
            
            texts = []
            confidences = []
            
            for line in results[0]:
                bbox, (text, confidence) = line
                if confidence >= 0.3:  # Lower threshold for individual detections
                    texts.append(text)
                    confidences.append(confidence)
            
            if not texts:
                return "", 0.0
                
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            return combined_text, avg_confidence
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {str(e)}")
            return "", 0.0

    def process_field_roi(self, image: np.ndarray, field_name: str, field_config: Dict) -> Dict:
        """Process a specific field using ROI-based PaddleOCR"""
        roi = field_config['roi']
        field_type = field_config['type']
        
        # Extract ROI
        roi_image = self.extract_roi(image, roi)
        
        if roi_image.size == 0:
            return {
                'field': field_name,
                'text': '',
                'confidence': 0.0,
                'method_used': 'none',
                'roi_size': (0, 0)
            }
        
        # Preprocess ROI for field type
        processed_roi = self.preprocess_roi(roi_image, field_type)
        
        result = {
            'field': field_name,
            'text': '',
            'confidence': 0.0,
            'method_used': 'none',
            'roi_size': roi_image.shape[:2],
            'attempts': {}
        }
        
        # Use PaddleOCR
        logger.info(f"Processing {field_name} with PaddleOCR...")
        text, confidence = self.extract_with_paddleocr(processed_roi)
        result['attempts']['paddleocr'] = {'text': text, 'confidence': confidence}
        
        result['text'] = text
        result['confidence'] = confidence
        result['method_used'] = 'PaddleOCR'
        
        return result

    def validate_field_content(self, field_name: str, text: str) -> Dict:
        """Validate and clean extracted field content"""
        validation_result = {
            'original': text,
            'cleaned': text,
            'is_valid': True,
            'validation_notes': []
        }
        
        if not text.strip():
            validation_result['is_valid'] = False
            validation_result['validation_notes'].append('Empty field')
            return validation_result
        
        cleaned_text = text.strip()
        
        try:
            # Field-specific validation and cleaning
            if 'hkid' in field_name.lower():
                # HKID format: A123456(7)
                pattern = r'[A-Z]{1,2}\d{6}\(\d\)'
                match = re.search(pattern, cleaned_text.upper().replace(' ', ''))
                if match:
                    cleaned_text = match.group()
                else:
                    validation_result['validation_notes'].append('Invalid HKID format')
                    
            elif 'date' in field_name.lower():
                # Various date formats
                date_patterns = [
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                    r'\d{1,2}\.\d{1,2}\.\d{2,4}',
                    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'
                ]
                found_date = False
                for pattern in date_patterns:
                    match = re.search(pattern, cleaned_text)
                    if match:
                        cleaned_text = match.group()
                        found_date = True
                        break
                if not found_date:
                    validation_result['validation_notes'].append('No valid date format found')
                    
            elif 'passport' in field_name.lower():
                # Passport format varies by country
                passport_pattern = r'[A-Z]{1,2}\d{6,9}'
                match = re.search(passport_pattern, cleaned_text.upper().replace(' ', ''))
                if match:
                    cleaned_text = match.group()
                else:
                    validation_result['validation_notes'].append('Possible passport format issue')
                    
            elif field_name.lower() == 'sex':
                # Gender validation
                gender_map = {'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'}
                upper_text = cleaned_text.upper()
                if upper_text in gender_map:
                    cleaned_text = gender_map[upper_text]
                else:
                    validation_result['validation_notes'].append('Unknown gender format')
                    
            elif 'currency' in field_name.lower():
                # Currency validation
                currencies = ['HKD', 'USD', 'EUR', 'GBP', 'CNY', 'JPY']
                upper_text = cleaned_text.upper()
                if any(curr in upper_text for curr in currencies):
                    for curr in currencies:
                        if curr in upper_text:
                            cleaned_text = curr
                            break
                else:
                    validation_result['validation_notes'].append('Unknown currency')
        except Exception as e:
            logger.error(f"Error in field validation: {e}")
            validation_result['validation_notes'].append(f'Validation error: {str(e)}')
        
        validation_result['cleaned'] = cleaned_text
        validation_result['is_valid'] = len(validation_result['validation_notes']) == 0
        
        return validation_result

    def process_insurance_form(self, image: np.ndarray) -> Dict:
        """Process entire insurance form using ROI-based field extraction with PaddleOCR"""
        logger.info("Starting ROI-based insurance form processing with PaddleOCR...")
        
        results = {
            'processing_method': 'ROI-based field extraction (PaddleOCR)',
            'total_fields': len(self.roi_detector.FIELD_DEFINITIONS),
            'processed_fields': {},
            'summary': {
                'successful_extractions': 0,
                'high_confidence_fields': 0,
                'validated_fields': 0
            }
        }
        
        # Process each defined field
        for field_name, field_config in self.roi_detector.FIELD_DEFINITIONS.items():
            logger.info(f"Processing field: {field_name}")
            
            try:
                # Extract and process field
                field_result = self.process_field_roi(image, field_name, field_config)
                
                # Validate field content
                if field_result['text']:
                    validation = self.validate_field_content(field_name, field_result['text'])
                    field_result['validation'] = validation
                    field_result['final_text'] = validation['cleaned']
                    
                    results['summary']['successful_extractions'] += 1
                    
                    if field_result['confidence'] > 0.7:
                        results['summary']['high_confidence_fields'] += 1
                        
                    if validation['is_valid']:
                        results['summary']['validated_fields'] += 1
                else:
                    field_result['validation'] = {'is_valid': False, 'validation_notes': ['No text extracted']}
                    field_result['final_text'] = ''
                
                results['processed_fields'][field_name] = field_result
                
            except Exception as e:
                logger.error(f"Error processing field {field_name}: {e}")
                # Add error field result
                results['processed_fields'][field_name] = {
                    'field': field_name,
                    'text': '',
                    'confidence': 0.0,
                    'method_used': 'error',
                    'roi_size': (0, 0),
                    'validation': {'is_valid': False, 'validation_notes': [f'Processing error: {str(e)}']},
                    'final_text': ''
                }
        
        # Calculate success rates
        total_fields = len(self.roi_detector.FIELD_DEFINITIONS)
        results['summary']['extraction_rate'] = results['summary']['successful_extractions'] / total_fields if total_fields > 0 else 0
        results['summary']['validation_rate'] = results['summary']['validated_fields'] / total_fields if total_fields > 0 else 0
        
        logger.info(f"PaddleOCR processing complete. Success rate: {results['summary']['extraction_rate']:.2%}")
        
        return results

    def hierarchical_ocr(self, image: np.ndarray) -> Dict:
        """Main entry point - Process image with PaddleOCR only"""
        try:
            return self.process_insurance_form(image)
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            # Return error result
            return {
                'processing_method': 'ROI-based field extraction (error)',
                'total_fields': len(self.roi_detector.FIELD_DEFINITIONS),
                'processed_fields': {},
                'summary': {
                    'successful_extractions': 0,
                    'high_confidence_fields': 0,
                    'validated_fields': 0,
                    'extraction_rate': 0,
                    'validation_rate': 0
                },
                'error': str(e)
            }