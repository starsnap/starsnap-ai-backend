# StarSnap AI Backend

Flask 기반 얼굴 임베딩 백엔드입니다. 업로드한 이미지에서 얼굴 임베딩을 추출하고 `star.face_image_vector`에 저장하며, 스냅 사진의 다중 얼굴 벡터화와 스타 사전 식별을 위한 내부 API를 제공합니다.

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
| GPU 이미지 | CUDA cuDNN runtime / Ubuntu | 12.8.1 / 22.04 |

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
starsnap-ai-server/
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
- `/api/internal/v1/face-analysis`에서 얼굴을 최대 10개까지 안정적인 순서로 추출하고 512차원 L2-normalized 벡터와 임계값을 통과한 스타 매칭을 반환
- 내부 분석 요청의 등록 스타 벡터는 한 번만 조회하고 모든 얼굴을 NumPy 행렬 연산으로 비교
- 별도 `images` 테이블 저장은 사용하지 않음
- 기존 `image.py`, `person.py` 모델은 `app/models/legacy/`로 이동

## JWT 인증/인가

- `POST /api/enroll`는 JWT 인증이 필요합니다.
- `POST /api/enroll`는 `ADMIN` 권한만 허용합니다.
- 로그인 시 발급된 HttpOnly `access-token` 쿠키가 필요하며, Bearer 헤더는 인증에 사용하지 않습니다.
- 상태 변경 요청 전 `GET /api/csrf-token`에서 토큰을 받은 뒤 `X-CSRFToken` 헤더로 전달해야 합니다.

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
- `401 Access token cookie missing or invalid`: `access-token` 쿠키 누락/형식 오류
- `401 Invalid token type: JWT=...`: Access 토큰 타입 불일치
- `401 Invalid token`: 서명 오류/토큰 형식 오류
- `401 Token expired`: 만료 토큰
- `403 Admin only`: 인증은 성공했지만 권한 부족

내부 얼굴 분석 API는 사용자 JWT와 분리된 `AI_INTERNAL_TOKEN`을 사용합니다. 값은 직접 환경변수 또는 `AI_INTERNAL_TOKEN_FILE`의 secret 파일에서 로드할 수 있습니다. 메인 백엔드는 `Authorization: Bearer <resolved-token>`을 전송해야 하며, 이 API의 이미지·embedding 응답 본문은 access log로 전달되지 않습니다.

## API

### `POST /api/internal/v1/face-analysis`

메인 백엔드 전용 다중 얼굴 분석 API입니다.

- Header: `Authorization: Bearer <AI_INTERNAL_TOKEN>`
- Header: `X-Request-Id` (optional, 없으면 UUID 생성)
- form-data: `file` (required), `maxFaces` (optional, 1부터 설정 상한까지)
- 얼굴 정렬: bbox 면적 내림차순, 이후 좌표 순
- 얼굴 없음: `200` + `faces: []`
- `bestMatch`: `MATCH_MIN_SIMILARITY` 이상인 스타만 반환하며 나머지는 `null`

```json
{
  "schemaVersion": "1",
  "requestId": "snap-request-123",
  "image": {"width": 1920, "height": 1080},
  "model": {
    "name": "buffalo_l",
    "version": "insightface-0.7.3",
    "embeddingDimension": 512,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "detectedFaceCount": 1,
  "processedFaceCount": 1,
  "truncated": false,
  "faces": [
    {
      "faceIndex": 0,
      "bbox": [120, 80, 220, 220],
      "detectionConfidence": 0.99,
      "embedding": [0.0123, -0.0456],
      "bestMatch": {"starId": "star_001", "similarity": 0.92}
    }
  ]
}
```

전체 오류·계약은 [API_SPEC.md](API_SPEC.md)를 참고하세요.

### `POST /api/enroll`
이미지를 업로드해 `star.face_image_vector`를 갱신합니다.

또한 `PHOTO_API_URL`의 path가 `/api/file/photo`로 끝나면 hostname과 관계없이 presign 발급 + presigned URL 업로드 흐름을 함께 수행합니다.

접근 제어:
- 인증 필요 (HttpOnly `access-token` 쿠키)
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
- Docker 이미지 기준으로 컨테이너 내부 `HEALTHCHECK`가 `/api/health`를 **30초마다** 호출합니다.

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
- `GET /api/csrf-token` 응답의 `csrfToken`을 같은 쿠키 세션에서 준비
- `POST /api/enroll`
- Cookie: `access-token=<access_token>` (로그인된 클라이언트에서는 자동 첨부)
- Header: `X-CSRFToken: <csrfToken>`
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

# 메인 백엔드 -> AI 백엔드 내부 API 전용 공유 토큰 (필수)
AI_INTERNAL_TOKEN=
# 파일 기반 secret을 사용할 때 설정 (AI_INTERNAL_TOKEN이 비어 있을 때만 사용)
AI_INTERNAL_TOKEN_FILE=

# 내부 얼굴 분석 제한/모델 계약 메타데이터 (선택, 아래는 기본값)
AI_FACE_ANALYSIS_MAX_IMAGE_BYTES=15728640
AI_FACE_ANALYSIS_MAX_PIXELS=60000000
AI_FACE_ANALYSIS_MAX_FACES=10
AI_FACE_ANALYSIS_MATCH_STARS=true
AI_FACE_MODEL_VERSION=insightface-0.7.3

# JWT Access Token 검증 시크릿 (starsnap-backend와 동일해야 함)
JWT_ACCESS_SECRET=

# 브라우저 쿠키 인증 API의 CSRF 서명 키 (선택, 미설정 시 JWT_ACCESS_SECRET 사용)
CSRF_SECRET_KEY=
# 운영 HTTPS에서는 true, 로컬 HTTP 디버그에서만 false
SESSION_COOKIE_SECURE=true
```

`AI_INTERNAL_TOKEN`은 사용자 JWT secret과 다른 충분히 긴 임의 값으로 설정하고 로그나 저장소에 커밋하지 않습니다. Docker/Swarm secret을 사용할 때는 `AI_INTERNAL_TOKEN`을 비우고 `AI_INTERNAL_TOKEN_FILE`에 마운트된 파일 경로를 지정할 수 있습니다. 두 값이 모두 있으면 직접 환경변수가 우선이며, 파일 값은 앞뒤 공백과 줄바꿈을 제거해 사용합니다. 어느 쪽도 없거나 파일이 비어 있으면 앱이 설정 로드 단계에서 즉시 실패합니다.

`AI_FACE_ANALYSIS_MATCH_STARS=false`로 실행하면 내부 얼굴 분석 API는 Star DB 인덱스를 읽지 않습니다. 얼굴 임베딩은 동일하게 반환하지만 모든 `bestMatch`는 `null`이므로, 운영 DB를 보유한 메인 백엔드가 벡터 매칭을 담당할 수 있습니다. 기본값 `true`는 기존 AI 백엔드의 직접 매칭 동작을 유지합니다.

## 단위 테스트

실제 얼굴 모델 없이 fake detector/service로 내부 계약을 검증합니다.

```bash
python -m unittest test_embedding_service_unit test_face_analysis_route_unit test_image_utils_unit -v
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
- `HEALTHCHECK --interval=30s --timeout=10s --start-period=2m --retries=3`

실행 후 로그에서 아래처럼 provider 적용 상태를 확인하세요.
- 기대값: `Applied providers: ['CUDAExecutionProvider', ...]`
- CPU만 보이면 GPU 라이브러리를 로드하지 못한 상태입니다.

컨테이너 내부에서 빠른 확인:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

결과에 `CUDAExecutionProvider`가 포함되어야 합니다.

