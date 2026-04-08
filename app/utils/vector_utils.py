"""
벡터 처리 유틸리티 함수들
"""
import numpy as np

def vec_to_bytes(v: np.ndarray) -> bytes:
    """
    numpy 벡터를 바이트로 변환
    
    Args:
        v (np.ndarray): 변환할 벡터
        
    Returns:
        bytes: float32로 변환된 바이트 데이터
    """
    return v.astype('float32').tobytes()

def bytes_to_vec(b: bytes) -> np.ndarray:
    """
    바이트를 numpy 벡터로 변환
    
    Args:
        b (bytes): 변환할 바이트 데이터
        
    Returns:
        np.ndarray: float32 벡터
    """
    return np.frombuffer(b, dtype='float32')

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    L2 정규화 (벡터의 길이를 1로 만듦)
    
    Args:
        v (np.ndarray): 정규화할 벡터
        
    Returns:
        np.ndarray: L2 정규화된 벡터
    """
    v = v.astype('float32')
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        v1 (np.ndarray): 첫 번째 벡터
        v2 (np.ndarray): 두 번째 벡터
        
    Returns:
        float: 코사인 유사도 (-1 ~ 1)
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

