import easyocr
import pytesseract
import cv2
import numpy as np
from PIL import Image
import paddleocr
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        """Initialize OCR models with confidence thresholds"""
        self.confidence_thresholds = {
            'easyocr': 0.7,
            'tesseract': 0.6,
            'paddleocr': 0.5
        }
        
        # Initialize models lazily
        self._easyocr_reader = None
        self._paddleocr_reader = None
        
    @property
    def easyocr_reader(self):
        """Lazy initialization of EasyOCR"""
        if self._easyocr_reader is None:
            logger.info("Initializing EasyOCR...")
            self._easyocr_reader = easyocr.Reader(['en'])
        return self._easyocr_reader
    
    @property
    def paddleocr_reader(self):
        """Lazy initialization of PaddleOCR"""
        if self._paddleocr_reader is None:
            logger.info("Initializing PaddleOCR...")
            self._paddleocr_reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self._paddleocr_reader

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh

    def extract_with_easyocr(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using EasyOCR"""
        try:
            logger.info("Running EasyOCR...")
            results = self.easyocr_reader.readtext(image)
            
            extracted_data = []
            total_confidence = 0
            text_parts = []
            
            for (bbox, text, confidence) in results:
                if confidence >= self.confidence_thresholds['easyocr']:
                    text_parts.append(text)
                    extracted_data.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    total_confidence += confidence
            
            avg_confidence = total_confidence / len(results) if results else 0
            full_text = ' '.join(text_parts)
            
            logger.info(f"EasyOCR completed. Confidence: {avg_confidence:.2f}")
            return full_text, avg_confidence, extracted_data
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return "", 0.0, []

    def extract_with_tesseract(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using Tesseract"""
        try:
            logger.info("Running Tesseract...")
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            extracted_data = []
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                if text and confidence >= self.confidence_thresholds['tesseract']:
                    text_parts.append(text)
                    extracted_data.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    })
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            full_text = ' '.join(text_parts)
            
            logger.info(f"Tesseract completed. Confidence: {avg_confidence:.2f}")
            return full_text, avg_confidence, extracted_data
            
        except Exception as e:
            logger.error(f"Tesseract failed: {str(e)}")
            return "", 0.0, []

    def extract_with_paddleocr(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using PaddleOCR"""
        try:
            logger.info("Running PaddleOCR...")
            results = self.paddleocr_reader.ocr(image, cls=True)
            
            extracted_data = []
            text_parts = []
            confidences = []
            
            for line in results[0] or []:
                bbox, (text, confidence) = line
                
                if confidence >= self.confidence_thresholds['paddleocr']:
                    text_parts.append(text)
                    extracted_data.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            full_text = ' '.join(text_parts)
            
            logger.info(f"PaddleOCR completed. Confidence: {avg_confidence:.2f}")
            return full_text, avg_confidence, extracted_data
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {str(e)}")
            return "", 0.0, []

    def hierarchical_ocr(self, image: np.ndarray) -> Dict:
        """
        Perform hierarchical OCR: EasyOCR → Tesseract → PaddleOCR
        Based on confidence thresholds
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        results = {
            'final_text': '',
            'final_confidence': 0.0,
            'method_used': '',
            'all_attempts': {},
            'extracted_data': []
        }
        
        # Try EasyOCR first
        text, confidence, data = self.extract_with_easyocr(processed_image)
        results['all_attempts']['easyocr'] = {
            'text': text,
            'confidence': confidence,
            'data': data
        }
        
        if confidence >= self.confidence_thresholds['easyocr']:
            results['final_text'] = text
            results['final_confidence'] = confidence
            results['method_used'] = 'EasyOCR'
            results['extracted_data'] = data
            return results
        
        # Try Tesseract if EasyOCR failed
        logger.info("EasyOCR confidence too low, trying Tesseract...")
        text, confidence, data = self.extract_with_tesseract(processed_image)
        results['all_attempts']['tesseract'] = {
            'text': text,
            'confidence': confidence,
            'data': data
        }
        
        if confidence >= self.confidence_thresholds['tesseract']:
            results['final_text'] = text
            results['final_confidence'] = confidence
            results['method_used'] = 'Tesseract'
            results['extracted_data'] = data
            return results
        
        # Try PaddleOCR as last resort
        logger.info("Tesseract confidence too low, trying PaddleOCR...")
        text, confidence, data = self.extract_with_paddleocr(processed_image)
        results['all_attempts']['paddleocr'] = {
            'text': text,
            'confidence': confidence,
            'data': data
        }
        
        results['final_text'] = text
        results['final_confidence'] = confidence
        results['method_used'] = 'PaddleOCR'
        results['extracted_data'] = data
        
        return results

    def extract_insurance_fields(self, text: str) -> Dict[str, str]:
        """
        Extract specific insurance claim form fields using regex patterns
        """
        fields = {
            'policy_number': '',
            'claim_number': '',
            'insured_name': '',
            'date_of_loss': '',
            'claim_amount': '',
            'phone_number': '',
            'email': '',
            'incident_description': ''
        }
        
        # Policy number patterns
        policy_patterns = [
            r'Policy\s*(?:Number|No\.?|#)?\s*:?\s*([A-Z0-9\-]+)',
            r'Policy\s*([A-Z0-9\-]{6,})',
            r'POL\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        # Claim number patterns
        claim_patterns = [
            r'Claim\s*(?:Number|No\.?|#)?\s*:?\s*([A-Z0-9\-]+)',
            r'Claim\s*([A-Z0-9\-]{6,})',
            r'CLM\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        # Name patterns
        name_patterns = [
            r'(?:Insured|Name|Customer)\s*(?:Name)?\s*:?\s*([A-Za-z\s]+)',
            r'Name\s*:?\s*([A-Za-z\s]{2,})'
        ]
        
        # Date patterns
        date_patterns = [
            r'(?:Date\s*of\s*Loss|Incident\s*Date|Loss\s*Date)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
        ]
        
        # Amount patterns
        amount_patterns = [
            r'(?:Claim\s*Amount|Amount|Total|Damage)\s*:?\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        # Phone patterns
        phone_patterns = [
            r'(?:Phone|Tel|Mobile)\s*:?\s*((?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
        ]
        
        # Email patterns
        email_patterns = [
            r'(?:Email|E-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ]
        
        # Apply patterns
        patterns_map = {
            'policy_number': policy_patterns,
            'claim_number': claim_patterns,
            'insured_name': name_patterns,
            'date_of_loss': date_patterns,
            'claim_amount': amount_patterns,
            'phone_number': phone_patterns,
            'email': email_patterns
        }
        
        for field, patterns in patterns_map.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields[field] = match.group(1).strip()
                    break
        
        return fields