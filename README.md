# StarSnap AI Backend

Flask 기반 얼굴 임베딩 백엔드입니다. 업로드한 이미지에서 얼굴 임베딩을 추출하고 `star.face_image_vector`에 저장합니다.

## 언어와 기술 스택

버전은 `requirements.txt`와 `dockerfile`의 현재 설정을 기준으로 합니다. Python minor 버전은 별도로 고정하지 않고 Ubuntu 22.04의 `python3` 패키지를 사용합니다.

| 구분 | 기술 | 설정 버전 |
|---|---|---:|
| 언어 | Python 3 | minor 미고정 |
| API | Flask / Flask-SQLAlchemy | 3.1.2 / 3.1.1 |
| 얼굴 모델 | InsightFace | 0.7.3 |
| 추론 | ONNX Runtime GPU | 1.23.2 |
| 영상 처리 | OpenCV headless | 4.12.0.88 |
| 수치 처리 | NumPy | 2.2.6 |
| 영속성 | SQLAlchemy / psycopg2 | 2.0.43 / 2.9.10 |
| 벡터 | pgvector | 0.3.6 |
| GPU 이미지 | CUDA cuDNN runtime / Ubuntu | 12.4.1 / 22.04 |

## 시스템 아키텍처

~~~mermaid
flowchart LR
    Client[관리자 또는 메인 서비스] -->|JWT / REST| Flask[Flask Routes]
    Flask --> Image[이미지 다운로드·검증]
    Image --> Insight[InsightFace / ArcFace]
    Insight --> Vector[512차원 임베딩]
    Vector --> DB[(메인 PostgreSQL + pgvector)]
    Flask --> S3[S3 presigned URL]
    Flask --> Hub[StarSnap Hub 로그]
~~~

AI 서버는 별도 데이터 저장소를 두지 않고 메인 서비스의 `star` 스키마를 공유합니다. 스키마와 JWT 설정을 메인 백엔드와 호환되게 유지해야 하며, GPU가 없을 때 사용할 provider 정책은 환경 설정으로 결정합니다.

상세 엔드포인트와 인증 조건은 [API_SPEC.md](API_SPEC.md)를 기준으로 합니다.

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

또한 `PHOTO_API_URL`이 `starsnap-backend`를 가리키면 presign 발급 + presigned URL 업로드 흐름을 함께 수행합니다.

접근 제어:
- 인증 필요 (`Authorization: Bearer <access_token>`)
- `ADMIN` 권한 필요

요청 form-data:
- `star_id` (required): 스타 ID
- `file` (optional): 이미지 파일
- `aiState` (optional): 업로드 메타데이터
- `dateTaken` (optional): 업로드 메타데이터
- `source` (optional): 업로드 메타데이터

동작:
- `file`가 있으면 얼굴 임베딩을 추출해 `star.face_image_vector`에 저장합니다.
- `file`가 없으면(또는 presign 전용 사용 시) 업스트림 presign 응답을 반환할 수 있습니다.

응답 예시(파일 업로드 + 임베딩 저장 성공):
```json
{
  "status": "ok",
  "star_id": "star_001",
  "embedding_dim": 512
}
```

응답 예시(presign 응답 전달):
```json
{
  "presignedUrl": "https://...",
  "requiredHeaders": {
    "x-amz-meta-ai-state": "false",
    "x-amz-meta-date-taken": "2026-05-13",
    "x-amz-meta-source": "internet"
  }
}
```

### Presigned URL 업로드 규칙

- `presignedUrl`로 업로드할 때는 `PUT` + raw body 업로드를 사용합니다.
- `FormData`(`-F`)를 사용하지 않습니다.
- 메타데이터(`aiState`, `dateTaken`, `source`)는 `x-amz-meta-*` 헤더로 전송해야 합니다.
- `requiredHeaders`가 있으면 반드시 그대로 포함해야 합니다.

`curl` 예시:

```bash
curl -X PUT "<presignedUrl>" \
  -H "x-amz-meta-ai-state:false" \
  -H "x-amz-meta-date-taken:2026-05-13" \
  -H "x-amz-meta-source:internet" \
  --upload-file "/path/to/sample.jpg"
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

### `GET /api/health`
헬스 체크 엔드포인트입니다.
- Docker 이미지 기준으로 컨테이너 내부 `HEALTHCHECK`가 `/api/health`를 **30분마다** 호출합니다.

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

    ### `POST /api/test/face-vector`
    테스트용 API입니다. 업로드한 인물 사진에서 얼굴 임베딩 벡터를 추출해 **JSON response body**로 반환합니다.

    요청 form-data:
    - `file` (required): 이미지 파일
        - `max_dim` (optional): 긴 변 기준 최대 픽셀 수. 기본값은 `ARCFACE_MAX_IMAGE_DIM`이며, 큰 이미지를 더 빠르게 처리하고 싶을 때 더 작은 값으로 낮출 수 있습니다.

    응답 예시:
    ```json
    {
      "status": "ok",
      "embedding_dim": 512,
      "embedding": [0.0123, -0.0456, ...],
      "bbox": [120, 80, 220, 220],
      "confidence": 0.99,
      "width": 1920,
      "height": 1080
    }
    ```

    curl 예시:
    ```bash
    curl -X POST "http://localhost:8000/api/test/face-vector" \
          -F "file=@/path/to/person.jpg" \
          -F "max_dim=1024"
    ```

## 빠른 테스트 플로우

1) starsnap-backend에서 Access Token 발급

2) 임베딩 등록 / presign 처리
- `POST /api/enroll`
- Header: `Authorization: Bearer <access_token>`
- form-data: `star_id`, (`file` 선택), (`aiState`, `dateTaken`, `source` 선택)

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
ARCFACE_MAX_IMAGE_DIM=1280

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

### GPU 모드

```bash
docker build -t starsnap-ai-backend -f dockerfile .
docker run --rm --gpus all -p 8000:8000 --env-file .env starsnap-ai-backend
```

헬스 체크 기본값(현재 `dockerfile` 기준):
- `HEALTHCHECK --interval=30m --timeout=10s --start-period=1m --retries=3`

실행 후 로그에서 아래처럼 provider 적용 상태를 확인하세요.
- 기대값: `Applied providers: ['CUDAExecutionProvider', ...]`
- CPU만 보이면 GPU 라이브러리를 로드하지 못한 상태입니다.

컨테이너 내부에서 빠른 확인:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

결과에 `CUDAExecutionProvider`가 포함되어야 합니다.

