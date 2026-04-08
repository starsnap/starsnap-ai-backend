"""
이미지 처리 유틸리티 함수들
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import base64

def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    바이트 데이터를 OpenCV 이미지로 디코딩
    
    Args:
        image_bytes (bytes): 이미지 바이트 데이터
        
    Returns:
        Optional[np.ndarray]: BGR 포맷의 이미지 배열, 실패 시 None
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    BGR 이미지를 RGB로 변환
    
    Args:
        image (np.ndarray): BGR 포맷 이미지
        
    Returns:
        np.ndarray: RGB 포맷 이미지
    """
    return image[..., ::-1]

def get_image_dimensions(image: np.ndarray) -> Dict[str, int]:
    """
    이미지의 너비와 높이 반환
    
    Args:
        image (np.ndarray): 이미지 배열
        
    Returns:
        Dict[str, int]: {'width': int, 'height': int}
    """
    height, width = image.shape[:2]
    return {'width': width, 'height': height}

def crop_image_by_bbox(image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    """(x, y, w, h) bbox 기준으로 이미지를 안전하게 크롭한다."""
    if image is None or len(bbox) != 4:
        return None

    x, y, w, h = bbox
    height, width = image.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, x1 + max(0, int(w)))
    y2 = min(height, y1 + max(0, int(h)))

    if x1 >= x2 or y1 >= y2:
        return None

    return image[y1:y2, x1:x2].copy()


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> Optional[str]:
    """OpenCV 이미지를 base64 문자열로 인코딩한다."""
    if image is None:
        return None

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return None

    return base64.b64encode(encoded.tobytes()).decode("ascii")


def encode_image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> Optional[bytes]:
    """OpenCV 이미지를 바이트로 인코딩한다."""
    if image is None:
        return None

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return None

    return encoded.tobytes()
