import os
import io
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGBLOB, LONGTEXT
import cv2
from insightface.app import FaceAnalysis

# Load .env
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------------
# DB models
# -------------------------
class Person(db.Model):
    __tablename__ = 'person'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    person_id = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    person_id = db.Column(db.String(255), db.ForeignKey('person.person_id'), nullable=True)
    filename = db.Column(db.String(512))
    content_type = db.Column(db.String(64))
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    image_blob = db.Column(LONGBLOB)          # 원본 이미지 저장
    face_embedding = db.Column(LONGBLOB)      # float32.tobytes() 저장
    embedding_model = db.Column(db.String(128))
    embedding_dim = db.Column(db.Integer)
    normalized = db.Column(db.Boolean, default=True)
    face_bbox = db.Column(LONGTEXT)           # JSON string으로 저장
    face_confidence = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

# -------------------------
# 모델 초기화 (insightface)
# -------------------------
# ctx_id: GPU index (e.g., 0) or -1 for CPU. 여기선 CPU 기본
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640,640))  # ctx_id=-1로 바꾸면 CPU-only

# -------------------------
# 헬퍼 함수: 벡터 <-> bytes, 정규화
# -------------------------
def vec_to_bytes(v: np.ndarray) -> bytes:
    return v.astype('float32').tobytes()

def bytes_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype='float32')

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype('float32')
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

# -------------------------
# 이미지 -> 임베딩 추출 함수
# 반환: dict { embedding: np.array, bbox: [x,y,w,h], confidence: float }
# -------------------------
def extract_face_embedding_from_bytes(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = bgr[..., ::-1]  # BGR -> RGB
    faces = face_app.get(rgb)
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴 선택 (bbox 면적 기준)
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = face.embedding  # numpy array
    emb = l2_normalize(emb)
    # bbox -> (x,y,w,h)
    x1, y1, x2, y2 = map(int, face.bbox[:4])
    bbox = [x1, y1, x2-x1, y2-y1]
    conf = float(face.det_score) if hasattr(face, 'det_score') else None
    return {"embedding": emb, "bbox": bbox, "confidence": conf, "width": bgr.shape[1], "height": bgr.shape[0]}

# -------------------------
# DB 초기화 (앱 시작시)
# -------------------------

with app.app_context():
    db.create_all()

# -------------------------
# 엔드포인트: enroll - 이미지 업로드 후 임베딩 저장
# form-data: file=<image>, optional: person_id, embedding_model
# -------------------------
@app.route("/enroll", methods=["POST"])
def enroll():
    if 'file' not in request.files:
        return jsonify({"error":"file required"}), 400
    f = request.files['file']
    person_id = request.form.get('person_id')  # optional
    embedding_model = request.form.get('embedding_model', 'insightface-default')
    content = f.read()
    # 임베딩 추출
    info = extract_face_embedding_from_bytes(content)
    if info is None:
        return jsonify({"error":"no face detected"}), 404

    emb_bytes = vec_to_bytes(info['embedding'])
    # 저장
    img = Image(
        person_id = person_id,
        filename = f.filename,
        content_type = f.content_type,
        width = info.get('width'),
        height = info.get('height'),
        image_blob = content,
        face_embedding = emb_bytes,
        embedding_model = embedding_model,
        embedding_dim = int(info['embedding'].shape[0]),
        normalized = True,
        face_bbox = str(info['bbox']),
        face_confidence = info.get('confidence')
    )
    db.session.add(img)
    db.session.commit()
    return jsonify({"status":"ok", "image_id": img.id, "person_id": person_id}), 201

# -------------------------
# 엔드포인트: get embedding by image_id (디버그용)
# -------------------------
@app.route("/embedding/<int:image_id>", methods=["GET"])
def get_embedding(image_id):
    img = Image.query.get(image_id)
    if not img:
        return jsonify({"error":"not found"}), 404
    if img.face_embedding is None:
        return jsonify({"error":"no embedding stored"}), 404
    vec = bytes_to_vec(img.face_embedding)
    return jsonify({
        "image_id": img.id,
        "person_id": img.person_id,
        "embedding_dim": img.embedding_dim,
        "embedding_preview": vec[:10].tolist()  # 앞 10개만 보여줌
    })

# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
