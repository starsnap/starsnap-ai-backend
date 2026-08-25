"""
이미지 처리 유틸리티 함수들
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError


def _inspect_image_metadata(
    image_bytes: bytes,
    max_pixels: int = 60_000_000,
) -> Optional[Dict[str, int]]:
    """Validate an encoded image and read metadata without decoding its pixels."""
    if not image_bytes or max_pixels <= 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    return None
                try:
                    orientation = int(image.getexif().get(274, 1))
                except (TypeError, ValueError):
                    orientation = 1
                if orientation not in range(1, 9):
                    orientation = 1
            # Reading EXIF can advance Pillow's parser, while verify() must run
            # immediately after open(), so validate through a fresh header parser.
            with Image.open(BytesIO(image_bytes)) as verify_image:
                verify_image.verify()
            return {
                "width": int(width),
                "height": int(height),
                "orientation": orientation,
            }
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError, ValueError):
        return None


def inspect_image_dimensions(
    image_bytes: bytes,
    max_pixels: int = 60_000_000,
) -> Optional[Dict[str, int]]:
    """Return validated storage dimensions before applying EXIF orientation."""
    metadata = _inspect_image_metadata(image_bytes, max_pixels=max_pixels)
    if metadata is None:
        return None
    return {"width": metadata["width"], "height": metadata["height"]}


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the EXIF display orientation to a raw OpenCV decode."""
    if orientation == 2:
        oriented = np.fliplr(image)
    elif orientation == 3:
        oriented = np.rot90(image, 2)
    elif orientation == 4:
        oriented = np.flipud(image)
    elif orientation == 5:
        oriented = np.transpose(image, (1, 0, 2))
    elif orientation == 6:
        oriented = np.rot90(image, 3)
    elif orientation == 7:
        oriented = np.flipud(np.fliplr(np.transpose(image, (1, 0, 2))))
    elif orientation == 8:
        oriented = np.rot90(image, 1)
    else:
        oriented = image
    return np.ascontiguousarray(oriented)


def decode_image_bytes(
    image_bytes: bytes,
    max_pixels: int = 60_000_000,
) -> Optional[np.ndarray]:
    """
    바이트 데이터를 OpenCV 이미지로 디코딩
    
    Args:
        image_bytes (bytes): 이미지 바이트 데이터
        
    Returns:
        Optional[np.ndarray]: EXIF 방향을 적용한 BGR 포맷 배열, 실패 시 None
    """
    metadata = _inspect_image_metadata(image_bytes, max_pixels=max_pixels)
    if metadata is None:
        return None

    arr = np.frombuffer(image_bytes, np.uint8)
    # Decode storage pixels consistently, then apply EXIF ourselves. OpenCV's
    # implicit orientation handling has differed across versions/platforms.
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        return None
    height, width = bgr.shape[:2]
    if width != metadata["width"] or height != metadata["height"]:
        return None
    return _apply_exif_orientation(bgr, metadata["orientation"])

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

def resize_image_to_max_dim(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """긴 변이 max_dim을 넘으면 비율을 유지해 축소한다.

    Returns:
        (resized_image, scale)
        - scale은 원본 대비 축소 비율(0 < scale <= 1)
        - resize가 필요 없으면 scale=1.0
    """
    if image is None or max_dim is None or max_dim <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0

    scale = max_dim / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale

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
