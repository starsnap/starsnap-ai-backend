"""
등록(Enroll) 관련 라우트
"""
from flask import Blueprint, request, jsonify, send_file
import io
from app.services import EmbeddingService
from app.utils.jwt_utils import require_jwt, require_admin
from config import Config

# 블루프린트 생성
enroll_bp = Blueprint('enroll', __name__, url_prefix='/api')

# 임베딩 서비스 인스턴스 (전역)
embedding_service = EmbeddingService(
    providers=Config.ARCFACE_PROVIDERS,
    model_name=Config.ARCFACE_MODEL_NAME,
    det_size=(Config.ARCFACE_DET_SIZE, Config.ARCFACE_DET_SIZE),
    default_min_similarity=Config.MATCH_MIN_SIMILARITY,
)


@enroll_bp.route("/enroll", methods=["POST"])
@require_jwt
@require_admin
def enroll():
    """
    이미지 업로드 후 star.face_image_vector에 임베딩 저장
    
    Request:
        - file: 이미지 파일 (required)
        - star_id: 스타 ID (required)
        
    Returns:
        - status: ok
        - star_id: 스타 ID
        - embedding_dim: 임베딩 차원
    """
    # 파일 검증
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400
    
    # 입력 매개변수 추출
    star_id = request.form.get('star_id')
    if not star_id:
        return jsonify({"error": "star_id required"}), 400
    
    # 파일 읽기
    content = file.read()
    if not content:
        return jsonify({"error": "file is empty"}), 400
    
    # 임베딩 추출
    info = embedding_service.extract_face_embedding(content)
    if info is None:
        return jsonify({"error": "no face detected"}), 404

    # Star 테이블에 임베딩 저장
    updated = embedding_service.save_star_embedding(star_id=star_id, embedding=info['embedding'])
    if not updated:
        return jsonify({"error": "star not found"}), 404
    
    return jsonify({
        "status": "ok",
        "star_id": star_id,
        "embedding_dim": int(info['embedding'].shape[0])
    }), 201


@enroll_bp.route("/embedding/star/<string:star_id>", methods=["GET"])
def get_embedding(star_id):
    """
    Star ID로 임베딩 벡터 조회 (디버그용)
    
    Args:
        star_id (str): 스타 ID
        
    Returns:
        - star_id: 스타 ID
        - embedding_dim: 임베딩 차원
        - embedding_preview: 임베딩의 첫 10개 값
    """
    vec = embedding_service.get_star_embedding_vector(star_id)
    
    if vec is None:
        return jsonify({"error": "not found"}), 404
    
    return jsonify({
        "star_id": star_id,
        "embedding_dim": int(vec.shape[0]),
        "embedding_preview": vec[:10].tolist()  # 앞 10개만 보여줌
    }), 200


@enroll_bp.route("/match/star", methods=["POST"])
def match_star():
    """업로드한 얼굴과 가장 유사한 Star 정보를 반환한다.

    임계값 미만이어도 비교 가능한 Star가 있으면 최고 유사도 결과를 반환한다.
    """
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400

    content = file.read()
    if not content:
        return jsonify({"error": "file is empty"}), 400

    info = embedding_service.extract_face_embedding(content)
    if info is None:
        return jsonify({"error": "no face detected"}), 404

    min_similarity = Config.MATCH_MIN_SIMILARITY
    result = embedding_service.find_most_similar_star(
        info['embedding'],
        min_similarity=-1.0,
    )
    if result is None:
        return jsonify({"error": "no enrolled star embeddings to compare"}), 404

    similarity = float(result['similarity'])
    threshold_passed = similarity >= min_similarity

    return jsonify({
        "status": "ok",
        "threshold": {
            "min_similarity": float(min_similarity),
            "passed": threshold_passed,
        },
        "query": {
            "embedding_dim": int(info['embedding'].shape[0]),
            "bbox": info['bbox'],
            "confidence": info['confidence']
        },
        "match": {
            "star": result['star'],
            "similarity": similarity
        }
    }), 200


@enroll_bp.route("/test/largest-face", methods=["POST"])
def test_largest_face():
    """업로드 이미지에서 가장 큰 얼굴 1개를 잘라 파일로 반환한다."""
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400

    content = file.read()
    if not content:
        return jsonify({"error": "file is empty"}), 400

    result = embedding_service.extract_largest_face_for_test(content)
    if result is None:
        return jsonify({"error": "no face detected"}), 404

    response = send_file(
        io.BytesIO(result["face_image_bytes"]),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="largest-face.jpg",
    )
    response.headers["X-Face-Bbox"] = ",".join(map(str, result["bbox"]))
    response.headers["X-Face-Confidence"] = str(result["confidence"])
    response.headers["X-Source-Width"] = str(result["width"])
    response.headers["X-Source-Height"] = str(result["height"])
    return response
