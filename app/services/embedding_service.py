"""
얼굴 임베딩 서비스 - 비즈니스 로직 계층
"""
from typing import Optional, Dict, Any
import numpy as np
from insightface.app import FaceAnalysis

from app.utils import (
    l2_normalize,
    cosine_similarity,
    decode_image_bytes,
    bgr_to_rgb,
    get_image_dimensions,
    crop_image_by_bbox,
    encode_image_to_bytes,
)
from db import db
from app.models import Star
from config import Config


class EmbeddingService:
    """
    얼굴 임베딩 추출 및 저장 서비스
    """
    
    def __init__(
        self,
        providers: list[str] | None = None,
        model_name: str | None = None,
        det_size: tuple[int, int] | None = None,
        default_min_similarity: float | None = None,
    ):
        """
        임베딩 서비스 초기화
        
        Args:
            providers (list): insightface 실행 제공자
        """
        if providers is None:
            providers = Config.ARCFACE_PROVIDERS
        if model_name is None:
            model_name = Config.ARCFACE_MODEL_NAME
        if det_size is None:
            det = Config.ARCFACE_DET_SIZE
            det_size = (det, det)
        if default_min_similarity is None:
            default_min_similarity = Config.MATCH_MIN_SIMILARITY

        self.default_min_similarity = default_min_similarity
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.face_app = FaceAnalysis(name=model_name, providers=providers)
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_face_embedding(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        이미지에서 얼굴 임베딩 추출
        
        Args:
            image_bytes (bytes): 이미지 바이트 데이터
            
        Returns:
            Optional[Dict]: 다음 정보 포함:
                - embedding: np.array (정규화된 임베딩)
                - bbox: [x, y, w, h] 좌표
                - confidence: 감지 신뢰도
                - width: 이미지 너비
                - height: 이미지 높이
                반실패 시 None
        """
        # 이미지 디코딩
        bgr = decode_image_bytes(image_bytes)
        if bgr is None:
            return None
        
        # BGR -> RGB 변환
        rgb = bgr_to_rgb(bgr)
        
        # 얼굴 감지
        faces = self.face_app.get(rgb)
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택 (바운딩박스 면적 기준)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # 임베딩 정규화
        embedding = face.embedding  # numpy array
        embedding = l2_normalize(embedding)
        
        # 바운딩박스 변환: (x1, y1, x2, y2) -> (x, y, w, h)
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        
        # 신뢰도 추출
        confidence = float(face.det_score) if hasattr(face, 'det_score') else None
        
        # 이미지 크기
        dimensions = get_image_dimensions(bgr)
        
        return {
            "embedding": embedding,
            "bbox": bbox,
            "confidence": confidence,
            **dimensions
        }

    def save_star_embedding(self, star_id: str, embedding: np.ndarray) -> bool:
        """Star 테이블의 face_image_vector 필드에 임베딩 저장"""
        star = db.session.get(Star, star_id)
        if star is None:
            return False

        # pgvector는 list[float] 형태로 저장한다.
        star.face_image_vector = embedding.astype(np.float32).tolist()
        db.session.commit()
        return True

    def get_star_embedding_vector(self, star_id: str) -> Optional[np.ndarray]:
        """Star ID로 임베딩 벡터 조회"""
        star = db.session.get(Star, star_id)
        if not star or star.face_image_vector is None:
            return None

        return np.asarray(star.face_image_vector, dtype=np.float32)

    def find_most_similar_star(
        self,
        query_embedding: np.ndarray,
        min_similarity: float | None = None,
    ) -> Optional[Dict[str, Any]]:
        """입력 임베딩과 가장 유사한 Star 1건을 반환한다."""
        if min_similarity is None:
            min_similarity = self.default_min_similarity

        query = l2_normalize(query_embedding)
        best_star = None
        best_similarity = -1.0

        stars = db.session.query(Star).filter(Star.face_image_vector.isnot(None)).all()
        for star in stars:
            star_vec = np.asarray(star.face_image_vector, dtype=np.float32)
            if star_vec.shape[0] != query.shape[0]:
                continue

            similarity = float(cosine_similarity(query, l2_normalize(star_vec)))
            if similarity > best_similarity:
                best_similarity = similarity
                best_star = star

        if best_star is None or best_similarity < min_similarity:
            return None

        return {
            "star": best_star.to_dict(),
            "similarity": best_similarity
        }

    def extract_largest_face_for_test(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """가장 큰 얼굴 1개를 잘라서 이미지 바이트와 함께 반환한다."""
        bgr = decode_image_bytes(image_bytes)
        if bgr is None:
            return None

        rgb = bgr_to_rgb(bgr)
        faces = self.face_app.get(rgb)
        if len(faces) == 0:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        x1, y1, x2, y2 = map(int, face.bbox[:4])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        confidence = float(face.det_score) if hasattr(face, 'det_score') else None

        crop = crop_image_by_bbox(bgr, bbox)
        if crop is None:
            return None

        face_image_bytes = encode_image_to_bytes(crop)
        if face_image_bytes is None:
            return None

        dimensions = get_image_dimensions(bgr)
        return {
            "bbox": bbox,
            "confidence": confidence,
            "face_image_bytes": face_image_bytes,
            **dimensions,
        }
