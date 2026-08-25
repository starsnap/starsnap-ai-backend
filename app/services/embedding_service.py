"""
얼굴 임베딩 서비스 - 비즈니스 로직 계층
"""
from typing import Optional, Dict, Any
import numpy as np
from insightface.app import FaceAnalysis
import logging
import time

from app.utils import (
    l2_normalize,
    cosine_similarity,
    decode_image_bytes,
    bgr_to_rgb,
    get_image_dimensions,
    resize_image_to_max_dim,
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
        allowed_modules: list[str] | None = None,
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
        self.max_image_dim = int(getattr(Config, "ARCFACE_MAX_IMAGE_DIM", 1280) or 0)
        self.max_image_pixels = int(
            getattr(Config, "AI_FACE_ANALYSIS_MAX_PIXELS", 60_000_000) or 60_000_000
        )
        self.max_faces = int(getattr(Config, "AI_FACE_ANALYSIS_MAX_FACES", 10) or 10)
        self.model_name = model_name
        if allowed_modules is None:
            # 임베딩 추출/매칭에 필요한 모듈만 로드해 초기화와 추론 오버헤드를 줄인다.
            allowed_modules = ["detection", "recognition"]

        logger = logging.getLogger(__name__)

        self.default_min_similarity = default_min_similarity
        self.providers = providers  # 외부에서 실제 provider 목록 확인용
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        # 부팅 시점에 실제 ORT provider 가용 상태를 남긴다.
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            logger.info("[onnxruntime] available_providers=%s", available)
            if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in available:
                logger.warning(
                    "[onnxruntime] CUDAExecutionProvider requested but unavailable. "
                    "Requested=%s, available=%s",
                    providers,
                    available,
                )
        except Exception as e:
            logger.warning("[onnxruntime] failed to inspect providers: %s", e)

        self.face_app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules,
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_size = det_size
        logger.info(
            "[insightface] configured providers=%s ctx_id=%s det_size=%s model=%s allowed_modules=%s",
            providers,
            ctx_id,
            det_size,
            model_name,
            allowed_modules,
        )

        # 첫 실요청에서 1분 가까운 지연이 발생하지 않도록 시작 시점에 1회 워밍업.
        self._warmup()

    def _warmup(self) -> None:
        logger = logging.getLogger(__name__)
        try:
            h, w = self.det_size
            # 고정 크기 더미 이미지를 한 번 태워 detector/recognizer 초기 CUDA 경로를 준비한다.
            dummy = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            t0 = time.monotonic()
            _ = self.face_app.get(dummy)
            logger.info("[insightface] warmup done in %.0fms det_size=%s", (time.monotonic() - t0) * 1000, self.det_size)
        except Exception as e:
            logger.warning("[insightface] warmup failed: %s", e)

    @property
    def active_providers(self) -> list[str]:
        """실제 로딩된 ONNX 세션의 provider 목록을 반환한다.
        모델이 없으면 초기화 시 설정된 providers를 반환한다."""
        try:
            models = getattr(self.face_app, "models", {})
            if models:
                first_model = next(iter(models.values()))
                session = getattr(first_model, "session", None)
                if session is not None:
                    return list(session.get_providers())
        except Exception:
            pass
        return list(self.providers)

    @property
    def is_gpu(self) -> bool:
        """GPU(CUDA) 로 실행 중이면 True."""
        return "CUDAExecutionProvider" in self.active_providers

    @property
    def device(self) -> str:
        """'GPU' 또는 'CPU' 문자열 반환."""
        return "GPU" if self.is_gpu else "CPU"

    @staticmethod
    def _face_sort_key(face: Any) -> tuple[float, float, float, float, float, float]:
        """큰 얼굴 우선, 이후 좌표/신뢰도 순으로 안정적인 처리 순서를 만든다."""
        bbox = np.asarray(face.bbox[:4], dtype=np.float64)
        x1, y1, x2, y2 = (float(value) for value in bbox)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        confidence = float(getattr(face, "det_score", 0.0) or 0.0)
        return (-area, x1, y1, x2, y2, -confidence)

    @staticmethod
    def _bbox_in_original_coordinates(
        face: Any,
        scale: float,
        image_width: int,
        image_height: int,
    ) -> Optional[list[int]]:
        """원본 좌표로 환산하고 image bounds로 clamp한 x/y/w/h를 반환한다."""
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        if scale and scale != 1.0:
            bbox = [int(round(value / scale)) for value in bbox]

        x, y, width, height = bbox
        clamped_x1 = min(max(x, 0), image_width)
        clamped_y1 = min(max(y, 0), image_height)
        clamped_x2 = min(max(x + width, 0), image_width)
        clamped_y2 = min(max(y + height, 0), image_height)
        if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
            return None
        return [
            clamped_x1,
            clamped_y1,
            clamped_x2 - clamped_x1,
            clamped_y2 - clamped_y1,
        ]

    def extract_face_embeddings(
        self,
        image_bytes: bytes,
        max_faces: int | None = None,
        max_dim: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        """이미지의 여러 얼굴에서 정규화된 512차원 임베딩을 추출한다.

        이미지 디코딩 실패는 ``None``을 반환하고, 유효한 이미지에서 얼굴이 없는
        경우에는 ``faces=[]``인 결과를 반환한다. 얼굴 순서는 면적 내림차순과 좌표
        순으로 고정되어 같은 입력에 대해 안정적인 ``faceIndex``를 제공한다.
        """
        logger = logging.getLogger(__name__)
        t_step = time.monotonic()

        bgr = decode_image_bytes(
            image_bytes,
            max_pixels=int(getattr(self, "max_image_pixels", 60_000_000)),
        )
        logger.info("[embedding] decode=%.0fms", (time.monotonic() - t_step) * 1000)
        if bgr is None:
            return None

        original_dimensions = get_image_dimensions(bgr)
        if max_dim is None:
            max_dim = self.max_image_dim
        if max_faces is None:
            max_faces = self.max_faces
        max_faces = max(1, int(max_faces))

        t_step = time.monotonic()
        processed_bgr, scale = resize_image_to_max_dim(
            bgr,
            int(max_dim) if max_dim is not None else 0,
        )
        logger.info("[embedding] resize=%.0fms scale=%.4f", (time.monotonic() - t_step) * 1000, scale)

        t_step = time.monotonic()
        rgb = bgr_to_rgb(processed_bgr)
        logger.info("[embedding] bgr2rgb=%.0fms", (time.monotonic() - t_step) * 1000)

        t_step = time.monotonic()
        detected_faces = list(self.face_app.get(rgb))
        detected_face_count = len(detected_faces)
        logger.info(
            "[embedding] face_detection=%.0fms num_faces=%d",
            (time.monotonic() - t_step) * 1000,
            detected_face_count,
        )

        sorted_faces = sorted(detected_faces, key=self._face_sort_key)
        selected_faces = sorted_faces[:max_faces]
        extracted_faces: list[Dict[str, Any]] = []

        t_step = time.monotonic()
        for face in selected_faces:
            bbox = self._bbox_in_original_coordinates(
                face,
                scale,
                original_dimensions["width"],
                original_dimensions["height"],
            )
            if bbox is None:
                raise ValueError("detected face has an out-of-bounds or empty bbox")

            raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                raise ValueError("detected face has no embedding")

            embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != 512 or not np.all(np.isfinite(embedding)):
                raise ValueError(
                    "detected face embedding must be finite and 512-dimensional "
                    f"(dim={embedding.shape[0]})"
                )

            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("detected face embedding has a non-positive norm")

            normalized_embedding = l2_normalize(embedding)
            extracted_faces.append({
                "face_index": len(extracted_faces),
                "embedding": normalized_embedding,
                "bbox": bbox,
                "confidence": (
                    float(face.det_score)
                    if hasattr(face, "det_score") and face.det_score is not None
                    else None
                ),
            })

        logger.info(
            "[embedding] normalize=%.0fms processed_faces=%d truncated=%s",
            (time.monotonic() - t_step) * 1000,
            len(extracted_faces),
            detected_face_count > max_faces,
        )
        return {
            "faces": extracted_faces,
            "detected_face_count": detected_face_count,
            "processed_face_count": len(extracted_faces),
            "truncated": detected_face_count > max_faces,
            **original_dimensions,
        }

    def extract_face_embedding(self, image_bytes: bytes, max_dim: int | None = None) -> Optional[Dict[str, Any]]:
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
        result = self.extract_face_embeddings(
            image_bytes=image_bytes,
            max_faces=1,
            max_dim=max_dim,
        )
        if result is None or not result["faces"]:
            return None

        face = result["faces"][0]
        return {
            "embedding": face["embedding"],
            "bbox": face["bbox"],
            "confidence": face["confidence"],
            "width": result["width"],
            "height": result["height"],
        }

    def save_star_embedding_with_reason(self, star_id: str, embedding: np.ndarray) -> tuple[bool, Optional[str]]:
        """Star 테이블의 face_image_vector 필드에 임베딩 저장.

        Returns:
            (success, reason)
            reason examples: star_not_found, invalid_embedding_dim:<n>, db_error:<ExceptionName>
        """
        star = db.session.get(Star, star_id)
        if star is None:
            return False, "star_not_found"

        # pgvector는 list[float] 형태로 저장한다.
        try:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim != 1:
                # flatten if necessary
                vec = vec.ravel()

            expected_dim = 512
            if vec.shape[0] != expected_dim:
                return False, f"invalid_embedding_dim:{vec.shape[0]}"

            # convert to Python list for SQLAlchemy/pgvector
            star.face_image_vector = vec.tolist()
            db.session.commit()
            return True, None
        except Exception as e:
            # rollback on error and log for debugging
            try:
                db.session.rollback()
            except Exception:
                pass
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("Failed to save star embedding for %s: %s", star_id, e)
            return False, f"db_error:{e.__class__.__name__}"

    def save_star_embedding(self, star_id: str, embedding: np.ndarray) -> bool:
        """Backward-compatible wrapper: 상세 사유 없이 성공 여부만 반환."""
        success, _ = self.save_star_embedding_with_reason(star_id=star_id, embedding=embedding)
        return success

    def get_star_embedding_vector(self, star_id: str) -> Optional[np.ndarray]:
        """Star ID로 임베딩 벡터 조회"""
        star = db.session.get(Star, star_id)
        if not star or star.face_image_vector is None:
            return None

        return np.asarray(star.face_image_vector, dtype=np.float32)

    def load_star_embedding_index(self) -> Dict[str, Any]:
        """등록된 Star 임베딩을 요청 단위 행렬 인덱스로 한 번에 읽는다.

        활성 상태이고 얼굴 벡터가 있는 Star만 등록 완료 대상으로 간주한다. ID로
        정렬해 동일 점수의 tie-break도 재현 가능하게 유지한다.
        """
        rows = (
            db.session.query(Star.id, Star.face_image_vector)
            .filter(Star.state.is_(True), Star.face_image_vector.isnot(None))
            .order_by(Star.id.asc())
            .all()
        )

        star_ids: list[str] = []
        normalized_vectors: list[np.ndarray] = []
        for star_id, stored_embedding in rows:
            vector = np.asarray(stored_embedding, dtype=np.float32).reshape(-1)
            if vector.shape[0] != 512 or not np.all(np.isfinite(vector)):
                continue
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0.0:
                continue
            star_ids.append(str(star_id))
            normalized_vectors.append(vector / norm)

        matrix = (
            np.vstack(normalized_vectors).astype(np.float32, copy=False)
            if normalized_vectors
            else np.empty((0, 512), dtype=np.float32)
        )
        return {"star_ids": star_ids, "embeddings": matrix}

    def find_best_star_matches(
        self,
        face_embeddings: list[np.ndarray],
        star_index: Optional[Dict[str, Any]] = None,
        min_similarity: float | None = None,
    ) -> list[Optional[Dict[str, Any]]]:
        """여러 얼굴을 등록 스타 행렬과 한 번에 비교해 임계값 통과 결과만 반환한다."""
        if not face_embeddings:
            return []
        if min_similarity is None:
            min_similarity = self.default_min_similarity
        if star_index is None:
            star_index = self.load_star_embedding_index()

        star_ids = list(star_index.get("star_ids", []))
        star_matrix = np.asarray(
            star_index.get("embeddings", np.empty((0, 512), dtype=np.float32)),
            dtype=np.float32,
        )
        if not star_ids or star_matrix.shape != (len(star_ids), 512):
            return [None] * len(face_embeddings)

        query_matrix = np.vstack([
            np.asarray(embedding, dtype=np.float32).reshape(-1)
            for embedding in face_embeddings
        ])
        if query_matrix.shape[1] != 512 or not np.all(np.isfinite(query_matrix)):
            raise ValueError("face embeddings must be finite 512-dimensional vectors")

        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("face embeddings must have a positive finite norm")
        query_matrix = query_matrix / norms

        similarities = query_matrix @ star_matrix.T
        best_indexes = np.argmax(similarities, axis=1)
        results: list[Optional[Dict[str, Any]]] = []
        for face_row, star_row in enumerate(best_indexes):
            similarity = float(np.clip(similarities[face_row, star_row], -1.0, 1.0))
            if similarity < float(min_similarity):
                results.append(None)
                continue
            results.append({
                "starId": star_ids[int(star_row)],
                "similarity": similarity,
            })
        return results

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

    def extract_largest_face_for_test(self, image_bytes: bytes, max_dim: int | None = None) -> Optional[Dict[str, Any]]:
        """가장 큰 얼굴 1개를 잘라서 이미지 바이트와 함께 반환한다."""
        logger = logging.getLogger(__name__)
        t_step = time.monotonic()

        bgr = decode_image_bytes(image_bytes)
        logger.info("[extract_largest_face] decode=%.0fms", (time.monotonic() - t_step) * 1000)
        if bgr is None:
            return None

        original_dimensions = get_image_dimensions(bgr)

        if max_dim is None:
            max_dim = self.max_image_dim

        t_step = time.monotonic()
        processed_bgr, scale = resize_image_to_max_dim(bgr, int(max_dim) if max_dim is not None else 0)
        logger.info("[extract_largest_face] resize=%.0fms scale=%.4f", (time.monotonic() - t_step) * 1000, scale)

        t_step = time.monotonic()
        rgb = bgr_to_rgb(processed_bgr)
        logger.info("[extract_largest_face] bgr2rgb=%.0fms", (time.monotonic() - t_step) * 1000)

        t_step = time.monotonic()
        faces = self.face_app.get(rgb)
        logger.info("[extract_largest_face] face_detection=%.0fms num_faces=%d", (time.monotonic() - t_step) * 1000, len(faces))
        if len(faces) == 0:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        x1, y1, x2, y2 = map(int, face.bbox[:4])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        if scale and scale != 1.0:
            bbox = [
                int(round(bbox[0] / scale)),
                int(round(bbox[1] / scale)),
                int(round(bbox[2] / scale)),
                int(round(bbox[3] / scale)),
            ]
        confidence = float(face.det_score) if hasattr(face, 'det_score') else None

        crop = crop_image_by_bbox(bgr, bbox)
        if crop is None:
            return None

        face_image_bytes = encode_image_to_bytes(crop)
        if face_image_bytes is None:
            return None

        return {
            "bbox": bbox,
            "confidence": confidence,
            "face_image_bytes": face_image_bytes,
            **original_dimensions,
        }
