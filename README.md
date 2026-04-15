# StarSnap AI Backend

Flask 기반 얼굴 임베딩 백엔드입니다. 업로드한 이미지에서 얼굴 임베딩을 추출하고 `star.face_image_vector`에 저장합니다.

## 프로젝트 구조

```
starsnap-ai-backend/
├── app.py
├── config.py
├── db.py
├── requirements.txt
├── dockerfile
└── app/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── star.py
    │   └── legacy/
    │       ├── image.py
    │       └── person.py
    ├── routes/
    │   └── enroll.py
    ├── services/
    │   └── embedding_service.py
    └── utils/
        ├── image_utils.py
        └── vector_utils.py
```

## 현재 동작 방식

- `/api/enroll`에서 얼굴 임베딩을 추출
- 추출된 벡터를 `star.face_image_vector (vector(512))`에 저장
- 별도 `images` 테이블 저장은 사용하지 않음
- 기존 `image.py`, `person.py` 모델은 `app/models/legacy/`로 이동

## JWT 인증/인가

- `POST /api/enroll`는 JWT 인증이 필요합니다.
- `POST /api/enroll`는 `ADMIN` 권한만 허용합니다.
- 요청 헤더는 반드시 `Authorization: Bearer <access_token>` 형식이어야 합니다.

토큰 검증 규칙:
- 알고리즘: `HS256`
- JWT 헤더: `JWT=access` (현재 서버 정책상 `typ`는 허용하지 않음)
- Payload 필드:
  - `jti`: 사용자 식별자
  - `authority`: 권한 (`ADMIN`, `USER`, `STAR`)

권한 정책:
- `authority=ADMIN`: `/api/enroll` 접근 가능
- `authority=USER` 또는 `authority=STAR`: `403 Admin only`

주요 인증 에러:
- `401 Authorization header missing or invalid`: Bearer 헤더 누락/형식 오류
- `401 Invalid token type: JWT=...`: Access 토큰 타입 불일치
- `401 Invalid token`: 서명 오류/토큰 형식 오류
- `401 Token expired`: 만료 토큰
- `403 Admin only`: 인증은 성공했지만 권한 부족

## API

### `POST /api/enroll`
이미지를 업로드해 `star.face_image_vector`를 갱신합니다.

접근 제어:
- 인증 필요 (`Authorization: Bearer <access_token>`)
- `ADMIN` 권한 필요

요청 form-data:
- `file` (required): 이미지 파일
- `star_id` (required): 스타 ID

응답 예시:
```json
{
  "status": "ok",
  "star_id": "star_001",
  "embedding_dim": 512
}
```

권한 부족 응답 예시:
```json
{
  "error": "Admin only"
}
```

### `GET /api/embedding/star/{star_id}`
스타의 임베딩 벡터 미리보기를 조회합니다.

응답 예시:
```json
{
  "star_id": "star_001",
  "embedding_dim": 512,
  "embedding_preview": [0.12, -0.03, 0.44]
}
```

### `GET /health`
헬스 체크 엔드포인트입니다.

### `POST /api/match/star`
업로드한 사진의 얼굴 임베딩을 추출하고, 등록된 `star.face_image_vector`와 비교해 가장 유사한 스타 1명을 반환합니다.

- 비교 가능한 Star 임베딩이 있으면 최고 유사도 1건을 항상 반환합니다.
- `threshold.passed`로 `MATCH_MIN_SIMILARITY` 통과 여부를 확인할 수 있습니다.
- 비교 대상 임베딩이 없을 때만 `404`를 반환합니다.

요청 form-data:
- `file` (required): 이미지 파일

Postman 설정:
- Method: `POST`
- URL: `http://localhost:8000/api/match/star`
- Body -> `form-data`
  - key: `file` (type: File)
  - value: 업로드할 이미지 파일 선택

응답 예시 (기본 임계값 0.45 이상일 때):
```json
{
  "status": "ok",
  "query": {
    "embedding_dim": 512,
    "bbox": [120, 80, 220, 220],
    "confidence": 0.99
  },
  "match": {
    "star": {
      "id": "star_001",
      "name": "...",
      "nickname": "...",
      "star_group_id": "group_001"
    },
    "similarity": 0.92
  }
}
```

응답 예시 (기본 임계값 0.45 미만일 때):
```json
{
  "status": "ok",
  "threshold": {
    "min_similarity": 0.45,
    "passed": false
  },
  "query": {
    "embedding_dim": 512,
    "bbox": [120, 80, 220, 220],
    "confidence": 0.99
  },
  "match": {
    "star": {
      "id": "star_001",
      "name": "...",
      "nickname": "...",
      "star_group_id": "group_001"
    },
    "similarity": 0.34
  }
}
```

### `POST /api/test/largest-face`
테스트용 API입니다. 업로드 이미지에서 **가장 큰 얼굴 1개만** 선택해 잘라낸 이미지를 **파일 다운로드**로 반환합니다.

요청 form-data:
- `file` (required): 이미지 파일

응답:
- Body: `largest-face.jpg` (image/jpeg)
- Header:
  - `X-Face-Bbox`: `x,y,w,h`
  - `X-Face-Confidence`: 얼굴 검출 신뢰도
  - `X-Source-Width`: 원본 이미지 너비
  - `X-Source-Height`: 원본 이미지 높이

## 빠른 테스트 플로우

1) starsnap-backend에서 Access Token 발급

2) 임베딩 등록
- `POST /api/enroll`
- Header: `Authorization: Bearer <access_token>`
- form-data: `file`, `star_id`

3) 유사도 검색
- `POST /api/match/star`
- form-data: `file`

## 환경 변수

`config.py` 기준 필수 키:

```env
DB_USER=starsnap
DB_PASSWORD=
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=starsnap
DB_SCHEME=starsnap
DEBUG=true

# ArcFace (InsightFace) 실행 옵션
ARCFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ARCFACE_MODEL_NAME=buffalo_l
ARCFACE_DET_SIZE=640

# /api/match/star 최소 유사도 임계값
MATCH_MIN_SIMILARITY=0.45

# JWT Access Token 검증 시크릿 (starsnap-backend와 동일해야 함)
JWT_ACCESS_SECRET=
```

`ARCFACE_PROVIDERS` 기본값은 GPU 우선 + CPU 폴백입니다.

## DB 준비

PostgreSQL + pgvector 확장을 사용합니다.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

`star` 테이블은 아래 스키마를 기준으로 사용합니다.
- `id` (PK)
- `face_image_vector vector(512)`
- 기타 메타 컬럼(`name`, `nickname`, `gender`, `created_at` 등)

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py
```

## Docker 실행

```bash
docker build -t starsnap-ai-backend -f dockerfile .
docker run --rm -p 8000:8000 --env-file .env starsnap-ai-backend
```
