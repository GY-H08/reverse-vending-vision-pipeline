# 🥤 Reverse Vending Machine Vision Pipeline

> 일회용컵 무인 반납기를 위한 컴퓨터 비전 판별 시스템
> 듀얼 카메라(상단/측면) 기반 컵 종류·이물질·뚜껑·바코드 판별 로직을 설계 단계부터 단독으로 구축하고, 인증 시험 결과를 반영해 반복적으로 고도화한 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-MobileNetV3-005CED?logo=onnx&logoColor=white)
![asyncio](https://img.shields.io/badge/asyncio-concurrent-yellow)

> ⚠️ 본 저장소는 실제 운영 코드 중 알고리즘 핵심 로직만 발췌하여 일반화한 버전입니다. 사내 보안 정책상 실제 IP, API 엔드포인트, 인증 정보는 포함하지 않습니다.

---

## 📌 Overview

- **What**: 자원순환 보증금 일회용컵 반납기의 비전 판별 파이프라인 (카메라 구성 → 통신 프로토콜 → 판정 알고리즘 → 2차 검증 모델까지 전체)
- **My Role**: 시스템 아키텍처 설계, 통신 프로토콜 구현, 판정 알고리즘 개발, 보조 딥러닝 모델 학습·연동, 현장 검증까지 단독 수행
- **Why it's hard**: 일반적인 산업용 비전 검사와 달리, 투명 액체 감지·기구적 제약·실시간 처리라는 세 가지 제약이 동시에 걸린 환경

---

## 🧨 Problem & Constraints

### 인증 시험 결과 분석
초기 버전은 외부 인증 시험에서 FAIL(90%대) 판정을 받았습니다. 오판 케이스를 직접 분류·분석한 결과:

| 오판 유형 | 비중 | 원인 |
|---|---|---|
| 액상 이물질(물) | 다수 | 정지 이미지 기반 모델이 투명 액체를 구분 못함 |
| 바코드 위치/소형컵 | 다수 | 단일 ROI로 컵 위치 편차 대응 불가 |
| 이중 바코드(훼손 컵) | 일부 | 검증 로직 부재 |

### 하드웨어 기구적 제약
회전판 규격이 일반 비전 검사 장비보다 훨씬 크게 설계되어, 컵이 정중앙이 아닌 위치에 투입되는 경우가 실사용에서 빈번했습니다. 이로 인해:

1. **투명 액체 미감지** — 정지 상태에서는 반사·굴절 차이가 거의 없어 빈 컵과 구분 불가
2. **바코드 시야 이탈** — 컵 위치에 따라 바코드가 카메라 경계에서 잘림
3. **컵 전도** — 리프트 승하강 중 측면에 위치한 컵이 부딪혀 넘어짐

비전 모델 정확도만으로는 해결되지 않는, 하드웨어-소프트웨어 복합 문제였습니다.

---

## 🏗️ Architecture

```
                 ┌─────────────────────┐
                 │     User / QR       │
                 └─────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │   Admission API (REST) │
              └────────────┬───────────┘
                           ▼
         ┌──────────────────────────────────┐
         │   Return Orchestrator (FastAPI)  │
         │  - Motor / IO control            │
         │  - Vision pipeline trigger       │
         └───────────┬──────────────────────┘
                      ▼
     ┌────────────────────────────────────────┐
     │            Vision Pipeline             │
     │                                         │
     │  [Side Camera]         [Top Camera]    │
     │  - Barcode (3-ROI)     - Cup type      │
     │  - Holder detection    - Foreign object│
     │  - Duplicate barcode   - Lid detection │
     │    rejection           - Liquid (ripple)│
     │                                         │
     │           ▼                ▼           │
     │     ┌──────────────────────────┐       │
     │     │  Rule-based Judgement    │       │
     │     │  (weighted voting)       │       │
     │     └────────────┬─────────────┘       │
     │                  ▼                     │
     │     ┌──────────────────────────┐       │
     │     │  ONNX 2nd-stage Verifier │       │
     │     │  (MobileNetV3-Small)     │       │
     │     └────────────┬─────────────┘       │
     └──────────────────┬──────────────────────┘
                         ▼
              ┌────────────────────┐
              │  Approve / Reject  │
              └────────────────────┘
```

통신은 TCP 소켓 기반으로 직접 구현했으며, 비전 카메라가 보내는 프레임을 실시간 파싱해 투표(voting) 방식으로 판정합니다.

---

## 💡 Key Engineering Decisions

### Ripple Pulse — 비접촉 액체 감지 기법 (직접 고안)
정지 이미지로는 투명 액체를 감지할 수 없다는 점에서 출발해, 회전판을 짧게 cw→ccw로 반복 구동시켜 컵 내부에 인위적인 파동을 만드는 방식을 설계했습니다.

- 정지 8프레임(이물질/뚜껑 판별) → ripple 8프레임(액상류 판별)으로 구간 분리 수집
- `asyncio.gather`로 모터 구동과 프레임 수집을 동시 실행, 목표 프레임 도달 시 즉시 정지
- 현장에서 진폭·속도를 튜닝하며 "감지율"과 "컵 안정성" 사이의 균형점을 탐색

### Classification 구조 축소 — OR 합산의 함정
판별 항목을 세분화할수록 정밀해 보였지만, OR 합산 구조에서는 항목이 늘어날수록 정상 컵까지 오거부되는 역효과가 발생함을 직접 데이터로 확인했습니다. 클래스 수를 줄이고 학습 데이터를 클래스당 대폭 강화하는 방향으로 재설계해 정확도와 안정성을 동시에 개선했습니다.

### 바코드 ROI 3분할
단일 중앙 ROI의 한계(좌우 투입 시 오판)를 해결하기 위해 인식 영역을 중앙+좌+우 3개로 분할하고, 좌우 영역에서 바코드가 인식되면 위치 이상으로 즉시 거부하는 안전 로직을 추가했습니다.

### 이중(보조) 모델 검증 구조
1차 규칙 기반 판정 외에, 직접 이미지를 수집·학습시킨 ONNX(MobileNetV3-Small) 4-class 모델을 2차 검증으로 연동했습니다. 1차 판정이 놓치는 유형(예: 정상 컵의 미세 오탐)을 2차에서 재확인하는 이중 검증 구조입니다.

---

## 🔬 Example: Inference Result

### ✅ 정상 승인 사례

실제 운영 환경에서 캡처된 정상 컵 판별 사례입니다.  
EasyVS 1차 판별과 ONNX 2차 검증이 모두 일치(`camera_model_agreed`)하여 최종 승인된 로그입니다.

!<img width="2368" height="1792" alt="top_20260629_161900_294366" src="https://github.com/user-attachments/assets/68c27438-917d-46b8-a3a0-1d63bc84ae0c" />


```json
{
  "file_id": "",
  "image_file": "",
  "camera": "top",
  "barcode_data": "",
  "cup_type": "플라스틱컵",
  "cup_type_score": 92.875,
  "holder": "",
  "holder_score": 0,
  "foreign_material": "이물질 없음",
  "foreign_score": 92.875,
  "whipping_cream": "플라스틱컵 휘핑크림 없음",
  "whipping_cream_score": 95.875,
  "water": "플라스틱 물 없음",
  "water_score": 90.25,
  "top_lid": "플라스틱컵 뚜껑 없음",
  "top_lid_score": 94.125,
  "passed": true,
  "result": "",
  "reason": null,
  "reject_reasons": [],
  "elapsed_sec": 7.992,
  "timestamp": "2026-06-29 16:19:00",
  "model_classification": {
    "enabled": true,
    "camera": "top",
    "image_path": "images\\top\\approved\\images\\top_20260629_161900_294366.jpg",
    "status": "ok",
    "label": "normal_plastic",
    "class_index": 0,
    "confidence": 0.976546,
    "probabilities": {
      "normal_plastic": 0.976546,
      "normal_paper": 0.015409,
      "abnormal": 0.001382,
      "paper_whipping": 0.006663
    },
    "passed": true,
    "reason": null
  },
  "model_fusion": {
    "mode": "agree_required",
    "action": "camera_model_agreed",
    "uncertain_action": "reject",
    "camera_passed": true,
    "camera_reason": null,
    "camera_reject_reasons": [],
    "model_passed": true,
    "model_label": "normal_plastic",
    "model_confidence": 0.976546,
    "whipping_ai_recheck": false,
    "whipping_ai_model_override": false,
    "final_passed": true,
    "final_reason": null,
    "changed": false
  }
}

````
### ❌ 반납 거부 사례 — 투명 액체(물) 감지

정지 이미지로는 투명한 물을 빈 컵과 거의 구분할 수 없습니다.  
회전판을 짧게 좌우로 반복 구동시켜 컵 내부에 인위적인 파동(ripple)을 만들고,  
그 구간에서만 별도로 촬영하여 액면의 출렁임을 포착하는 방식으로 감지합니다.

<img width="2368" height="1792" alt="top_20260629_143034_307099" src="https://github.com/user-attachments/assets/099f3e10-d3d8-48f2-ab77-8afe44217432" />
```json
{
  "file_id": "",
  "image_file": "",
  "camera": "top",
  "barcode_data": "",
  "cup_type": "플라스틱컵",
  "cup_type_score": 93.75,
  "holder": "",
  "holder_score": 0,
  "foreign_material": "이물질 있음",
  "foreign_score": 94.875,
  "whipping_cream": "플라스틱컵 휘핑크림 없음",
  "whipping_cream_score": 95.0,
  "water": "플라스틱 물 있음",
  "water_score": 94.5,
  "top_lid": "플라스틱컵 뚜껑 없음",
  "top_lid_score": 92.4,
  "passed": false,
  "result": "",
  "reason": "foreign_material_detected",
  "reject_reasons": [
    "foreign_material_detected"
  ],
  "elapsed_sec": 8.057,
  "timestamp": "2026-06-29 14:30:34"
}
''''
---

## 📊 Results

| 항목 | 개선 전 | 개선 후 |
|---|---|---|
| Classification 항목 수 | 14개 | 9개 (학습 데이터 대폭 강화) |
| 인증 시험 결과 | FAIL (90%대) | 이물질 전유형 100% 거부 확인 |
| 바코드 인식 ROI | 1개 (중앙) | 3개 (중앙+좌우, 위치 이상 즉시 거부) |
| 이중 바코드 대응 | 미감지 | 다중 프레임 내 검출 시 즉시 거부 |
| 정상 컵 검증 | - | 다위치 투입 반복 테스트로 안정성 확인 |

---


## 🛠️ Tech Stack

`Python` `FastAPI` `asyncio` `ONNX Runtime` `MobileNetV3` `TCP Socket Programming` `Computer Vision (Rule-based + DL hybrid)`

---

## 📁 Code (Sanitized Snippets)

핵심 알고리즘만 발췌했습니다. 전체 운영 코드, 실제 네트워크 정보, 사내 API 연동 부분은 보안상 비공개입니다.

```
github_portfolio
├── README.md
├── docs/
│   └── before_after.png        # 개선 전후 구조 비교 다이어그램
└── snippets/
    ├── ripple_pulse.py          # 비접촉 액체 감지를 위한 회전판 파동 생성 로직
    ├── judge_top_frames.py      # 상단 카메라 다중 프레임 가중 투표 판정 로직
    └── judge_side_frames.py     # 측면 카메라 바코드 ROI 분할 및 이중 바코드 거부 로직
```

- [`snippets/ripple_pulse.py`](./snippets/ripple_pulse.py)
- [`snippets/judge_top_frames.py`](./snippets/judge_top_frames.py)
- [`snippets/judge_side_frames.py`](./snippets/judge_side_frames.py)

---

## License

이 저장소는 포트폴리오 목적의 발췌 코드입니다. 실제 프로덕션 코드 및 사내 인프라 정보는 포함되어 있지 않습니다.
