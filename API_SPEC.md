# StarSnap AI 백엔드 API 명세

> 서버 루트: `starsnap-main/starsnap-ai-backend`  
> Server origin: `http://localhost:8000`  
> 엔드포인트: 6개

## 1. 공통 규칙

| 항목 | 코드 기준 |
|---|---|
| Bind | `0.0.0.0:8000` |
| Blueprint prefix | `/api` |
| 기본 응답 | 명시적 API 응답은 JSON. 얼굴 crop API만 JPEG binary |
| CORS | 앱 설정과 `flask-cors` 의존성이 없어 미지원 |
| 인증 | `/enroll`만 Bearer JWT + `ADMIN`; 나머지는 공개 |
| DB | 메인 PostgreSQL의 `star` table과 `vector(512)` column 공유 |
| 자동 method | Flask가 `OPTIONS`, GET route에는 `HEAD`도 제공하지만 아래에는 업무 method만 기재 |

등록되지 않은 path/method의 `404`, `405`와 잡히지 않은 `500`은 Flask 기본 HTML 응답일 수 있다. 모든 요청은 access-log 설정이 활성화된 경우 Hub `POST /api/server-logs`로 비동기 전송된다. multipart의 파일 내용은 로그에서 제외되며, 비밀번호·토큰·서명 query와 `Authorization`/`Cookie`/`Set-Cookie` 등 민감 header는 `[REDACTED]`로 치환된다.

소스: [app.py](../../starsnap-main/starsnap-ai-backend/app.py), [app/__init__.py](../../starsnap-main/starsnap-ai-backend/app/__init__.py), [enroll.py](../../starsnap-main/starsnap-ai-backend/app/routes/enroll.py)

## 2. 엔드포인트 요약

| 구분 | Method | Path | 인증 | 성공 |
|---|---|---|---|---|
| 운영 | `GET` | `/api/health` | 공개 | `200 {"status":"ok"}` |
| 운영 | `POST` | `/api/enroll` | Bearer JWT + `ADMIN` | `201` 또는 upstream status |
| 디버그 | `GET` | `/api/embedding/star/{star_id}` | 공개 | `200` |
| 운영 | `POST` | `/api/match/star` | 공개 | `200` |
| 테스트 | `POST` | `/api/test/largest-face` | 공개 | `200 image/jpeg` |
| 테스트 | `POST` | `/api/test/face-vector` | 공개 | `200 JSON` |

## 3. 인증

`POST /api/enroll`만 다음 header가 필요하다.

```http
Authorization: Bearer <access-token>
```

- 서명 algorithm: `HS256`.
- JWT header의 `JWT` 값이 정확히 `access`여야 한다.
- payload `authority`가 정확히 `ADMIN`이어야 한다.
- `exp`, `iat`, `nbf`는 claim이 있을 때 PyJWT 기본 규칙으로 검증하지만 필수 claim으로 강제하지 않는다.
- `sub`, `jti`가 있으면 문자열 타입 검증도 적용된다. 서버가 audience 값을 넘기지 않으므로 비어 있지 않은 `aud` claim이 든 token은 invalid audience로 거부된다.
- PyJWT의 header `typ` 검증은 사용하지 않지만, 애플리케이션이 별도 header claim `JWT=access`를 직접 강제한다. 위 표에 따로 노출하지 않은 PyJWT 검증 실패도 `401 {"error":"Invalid token"}`으로 합쳐진다.

| HTTP | 응답 |
|---|---|
| `401` | `{"error":"Authorization header missing or invalid"}` |
| `401` | `{"error":"Token expired"}` |
| `401` | `{"error":"Invalid token"}` |
| `401` | `{"error":"Invalid token type: JWT=<value>"}` |
| `403` | `{"error":"Admin only"}` |

근거: [jwt_utils.py](../../starsnap-main/starsnap-ai-backend/app/utils/jwt_utils.py)

## 4. 상세 API

### `GET /api/health`

입력과 인증이 없다.

```json
{
  "status": "ok"
}
```

Docker health check도 이 path를 사용한다.

### `POST /api/enroll`

얼굴 embedding을 추출해 `star.face_image_vector`에 저장하고 사진 API로 원본 파일을 전달한다.

Request: 파일을 보내면 `multipart/form-data`. 파일 없는 presign-only 흐름은 `application/x-www-form-urlencoded` 또는 multipart form도 처리한다.

| Field | 타입 | 필수 | 처리 |
|---|---|---:|---|
| `star_id` | string | 예 | 빈 문자열만 거부. 공백·길이·ID 형식 검증 없음 |
| `file` | binary | 조건부 | filename과 내용이 비면 거부. MIME·확장자·크기 제한 없음 |
| `aiState` | string/boolean-like | 아니요 | `1,true,yes,on`→true, `0,false,no,off`→false, 그 외 값은 그대로 upstream 전달 |
| `dateTaken` | string | 아니요 | 형식 검증 없음 |
| `source` | string | 아니요 | 형식 검증 없음 |

`file` 생략은 `PHOTO_API_URL`이 `starsnap-backend`를 포함해 presign 흐름을 사용할 때만 코드상 허용된다. 그 외에는 `400 no file provided`다.

일반 성공:

```http
201 Created
Content-Type: application/json
```

```json
{
  "status": "ok",
  "star_id": "star_001",
  "embedding_dim": 512
}
```

파일 없는 presign-only 흐름은 upstream JSON object와 HTTP status를 그대로 반환하므로 고정 schema가 아니다.

처리 순서와 부수효과:

1. presign POST와 얼굴 추출을 병렬 실행한다.
2. 가장 큰 얼굴을 선택하고 L2-normalized 512차원 embedding을 만든다.
3. 기존 star row의 `face_image_vector`를 갱신하고 commit한다.
4. 현재 메인 서버 구성에서는 presigned URL로 raw `PUT`, 다른 구성에서는 multipart `POST`한다.

DB commit이 사진 업로드보다 먼저이므로 이후 presign/PUT 실패 시 embedding 변경은 rollback되지 않는다.

| HTTP | 조건 / body |
|---|---|
| `400` | `star_id required`, `filename is empty`, `file is empty`; presign 요청 자체가 없는 파일 미첨부 흐름의 `no file provided` |
| `404` | `no face detected` — 이미지 decode 실패 포함 |
| `404` | `star not found` |
| `500` | `failed to save face_image_vector`; `reason=invalid_embedding_dim:<n>` 또는 `db_error:<ExceptionName>` |
| `502` | `presignedUrl missing from upstream response`, `no response from upstream`, `no response from presigned upload` |
| upstream status | 파일 없는 presign-only 응답은 upstream status/body를 그대로 전달. upstream이 성공 status와 비-object/빈 body를 주면 같은 status로 `no file provided`가 반환될 수 있음 |

파일이 있는 흐름에서 presign이 실패하거나 JSON object를 주지 않으면 원 status/body 대신 `502 {"error":"presignedUrl missing from upstream response"}`를 반환한다. Presigned PUT 실패는 upstream status를 유지하되 body를 `{ "error": "presigned upload failed", "upstream_body": ... }`로 감싼다.

ML 예외, DB 조회 자체의 예외, async future 예외 등은 Flask 기본 `500`으로 떨어질 수 있다.

### `GET /api/embedding/star/{star_id}`

인증 없이 embedding 일부를 노출하는 디버그 API다. `star_id`는 `/`를 제외한 string이고 별도 형식 검증은 없다.

```json
{
  "star_id": "star_001",
  "embedding_dim": 512,
  "embedding_preview": [0.012, -0.034, 0.056]
}
```

- `embedding_preview`: 최대 첫 10개 float.
- star가 없거나 embedding이 `NULL`이면 모두 `404 {"error":"not found"}`.
- DB read만 수행한다.

### `POST /api/match/star`

업로드 이미지의 가장 큰 얼굴과 등록된 모든 star embedding을 비교해 최고 유사도 1건을 반환한다.

Request: `multipart/form-data`, binary field `file` 필수. filename/non-empty만 검사한다.

```json
{
  "status": "ok",
  "threshold": {
    "min_similarity": 0.45,
    "passed": true
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
      "gender": "...",
      "birthday": "2000-01-01",
      "explanation": "...",
      "image_key": "...",
      "star_group_id": "group_001",
      "created_at": "2026-08-20T12:34:56"
    },
    "similarity": 0.92
  }
}
```

- `bbox`: 원본 좌표의 `[x,y,width,height]` integer 배열.
- 임계값 미만이어도 후보가 있으면 최고 후보와 `passed=false`를 `200`으로 반환한다.
- 현재 checkout의 `MATCH_MIN_SIMILARITY` 값은 `0.45`지만 환경 설정값이다.
- non-null embedding 전체를 애플리케이션에서 순회 비교하며 DB write는 없다.

오류:

- `400`: `file required`, `filename is empty`, `file is empty`.
- `404`: `no face detected`, `no enrolled star embeddings to compare`.

### `POST /api/test/largest-face`

공개 테스트 API다. Request는 `multipart/form-data`, binary field `file` 필수다.

성공:

```http
200 OK
Content-Type: image/jpeg
Content-Disposition: attachment; filename=largest-face.jpg
X-Face-Bbox: x,y,width,height
X-Face-Confidence: <float-or-None>
X-Source-Width: <integer>
X-Source-Height: <integer>
```

Body는 가장 큰 얼굴 하나를 원본 이미지에서 crop하고 JPEG로 재인코딩한 binary다. `400`은 파일 누락/빈 파일, `404`는 얼굴 없음 또는 decode/crop/encode 실패다. 인증과 DB 접근은 없다.

### `POST /api/test/face-vector`

전체 얼굴 vector와 실행 provider 정보를 반환하는 공개 테스트 API다.

Request: `multipart/form-data`

| Field | 타입 | 필수 | 처리 |
|---|---|---:|---|
| `file` | binary | 예 | filename/non-empty만 검사 |
| `max_dim` | integer string | 아니요 | 기본 `ARCFACE_MAX_IMAGE_DIM`(현재 1280); 정수 변환만 검사, 0 이하이면 resize 비활성화, 상한 없음 |

```json
{
  "status": "ok",
  "device": "GPU",
  "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "embedding_dim": 512,
  "embedding": [0.0123, -0.0456],
  "bbox": [120, 80, 220, 220],
  "confidence": 0.99,
  "width": 1920,
  "height": 1080,
  "max_dim": 1280
}
```

오류는 파일 관련 `400`, 잘못된 `max_dim`의 `400`, 얼굴 없음의 `404`다. DB 접근은 없다.

## 5. 주요 환경 설정

필수:

- DB: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_SCHEME`.
- 실행: `DEBUG`.
- ML: `ARCFACE_PROVIDERS`, `ARCFACE_MODEL_NAME`, `ARCFACE_DET_SIZE`, `MATCH_MIN_SIMILARITY`.
- 인증: `JWT_ACCESS_SECRET`.
- upstream: `PHOTO_API_URL`, `PHOTO_API_TIMEOUT_SECONDS`.

선택 기본값:

- `ARCFACE_MAX_IMAGE_DIM=1280`.
- `ACCESS_LOG_ENABLED=true`.
- `ACCESS_LOG_SERVICE_NAME=starsnap-ai-backend`.
- 코드의 기본 `ACCESS_LOG_URL`은 `http://host.docker.internal:7070/api/server-logs`인데 현재 Hub server port `8081`과 불일치한다.

`PHOTO_API_URL`과 timeout은 환경변수가 없으면 `services.yaml` 값을 사용할 수 있다. `DB_SCHEME`은 필수로 읽지만 실제 URI는 `postgresql://`로 고정되는 구현 불일치가 있다. 앱 시작 시 `db.create_all()`도 시도한다.

## 6. 구현 주의사항

1. `/embedding/star/*`, `/test/largest-face`, `/test/face-vector`가 인증 없이 공개된다.
2. 업로드 파일의 크기, MIME, 확장자, 이미지 해상도 제한과 rate limit이 없다.
3. CORS가 없어 browser cross-origin 호출은 reverse proxy 설정 없이는 허용되지 않는다.
4. presign-only 성공 계약은 upstream에 종속된다.
5. HTTP API 자동화 테스트가 없다. 루트 `test.py`는 API test가 아니라 로컬 이미지 비교 script다.
6. 일부 기본 오류는 JSON이 아니라 Flask HTML일 수 있다.

핵심 구현 근거: [enroll.py](../../starsnap-main/starsnap-ai-backend/app/routes/enroll.py), [embedding_service.py](../../starsnap-main/starsnap-ai-backend/app/services/embedding_service.py), [http_forward.py](../../starsnap-main/starsnap-ai-backend/app/utils/http_forward.py), [config.py](../../starsnap-main/starsnap-ai-backend/config.py).
