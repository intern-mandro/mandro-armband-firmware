# 펌웨어 프로토콜 명세

> 소스 펌웨어: `firmware/exo_armband_hybrid/`  
> 이 문서는 이 펌웨어와 BLE로 통신하는 클라이언트(암밴드 제어 앱)를 만드는 데
> 필요한 모든 스펙(GATT 프로파일, 패킷 포맷, 가중치 전송 프로토콜)을 정리한다.

---

## 목차

1. [기기 식별](#1-기기-식별)
2. [BLE GATT 프로파일](#2-ble-gatt-프로파일)
3. [Characteristic 1 — raw EMG 스트림](#3-characteristic-1--raw-emg-스트림)
4. [Characteristic 2 — 추론 결과](#4-characteristic-2--추론-결과)
4-1. [Characteristic 3 — 가중치 수신](#4-1-characteristic-3--가중치-수신-로컬-학습용)
5. [타이밍](#5-타이밍)
6. [연결 파라미터](#6-연결-파라미터)
7. [채널-핀 매핑](#7-채널-핀-매핑)
8. [연결 끊김 동작](#8-연결-끊김-동작)
9. [모델 플래시 방식](#9-모델-플래시-방식)
10. [펌웨어 내부 추론 파이프라인](#10-펌웨어-내부-추론-파이프라인)

---

## 1. 기기 식별

| 항목 | 값 |
|------|----|
| BLE 광고 이름 | `ESP32S3_FAST_BLE` |
| 제조사 VID (USB 감지용) | `0x303A` (Espressif) |
| 칩 | ESP32-S3 |

---

## 2. BLE GATT 프로파일

| 항목 | 값 |
|------|----|
| Service UUID | `12345678-1234-1234-1234-1234567890ab` |
| Characteristic UUID (raw EMG) | `abcd1234-5678-1234-5678-abcdef123456` |
| Characteristic UUID (추론 결과) | `abcd1234-5678-1234-5678-abcdef123457` |
| Characteristic UUID (가중치 수신) | `abcd1234-5678-1234-5678-abcdef123458` |

> UUID 마지막 두 자리만 다름: `...56` = EMG, `...57` = 예측, `...58` = 가중치 수신(§4-1)

**Characteristic 속성:**

| Characteristic | 속성 | CCCD |
|---|---|---|
| raw EMG (`...56`) | NOTIFY only | BLE2902 포함, `0x0001`/`0x0000`로 구독 켜고 끌 수 있음(전력 절약, §3 참고) |
| 추론 결과 (`...57`) | READ + NOTIFY | BLE2902 포함, `0x0001` 쓰기로 활성화, 항상 구독 유지 |
| 가중치 수신 (`...58`) | WRITE + NOTIFY | BLE2902 포함, `0x0001` 쓰기로 응답 NOTIFY 활성화 (§4-1) |

---

## 3. Characteristic 1 — raw EMG 스트림

### 패킷 크기

```
SAMPLES_PER_PACKET = 20
N_CHANNELS         = 8
PACKET_SIZE        = 20 × 8 = 160 bytes
```

### 레이아웃 (row-major)

```
byte[0..7]    → sample 0, CH0~CH7
byte[8..15]   → sample 1, CH0~CH7
...
byte[152..159]→ sample 19, CH0~CH7

인덱스 공식: byte[샘플번호 × 8 + 채널번호]
```

### 값 타입 및 범위

| 항목 | 내용 |
|------|------|
| 타입 | `uint8` (부호 없는 1바이트) |
| 원본 ADC | 10-bit (0~1023) |
| 펌웨어 변환 | `val = ADC_read - 250`, 클리핑 `[0, 255]` |
| 유효 범위 | 0~255 |
| 엔디안 | 해당 없음 (1바이트) |

### Android 파싱 예시 (Kotlin)

```kotlin
fun parseEmgPacket(bytes: ByteArray): Array<IntArray> {
    // returns [20][8] — [sample_index][channel_index]
    val result = Array(20) { IntArray(8) }
    for (s in 0 until 20) {
        for (ch in 0 until 8) {
            result[s][ch] = bytes[s * 8 + ch].toInt() and 0xFF
        }
    }
    return result
}
```

> `and 0xFF` 필수 — Kotlin `Byte`는 signed (-128~127)이므로 uint8로 재해석해야 함.

### 용도

- 실시간 파형 시각화 (신호 탭 8채널 파형)
- 학습용 데이터 녹화(로컬 학습 — 서버로 업로드하지 않음, §9 참고)

### 전력 절약 — 이 Characteristic만 구독 껐다 켤 수 있음

라이브러리(`BLEDevice.h`/`BLE2902.h`, Bluedroid 기반)가 표준으로 지원하는
CCCD 동작을 그대로 이용: 클라이언트가 이 Characteristic의 CCCD를 `0x0000`으로
다시 쓰면(구독 해제), 라이브러리의 `BLECharacteristic::notify()`가 실제
무선 송신 함수(`esp_ble_gatts_send_indicate`) 호출 전에 CCCD 상태를 확인해서
그 자리에서 리턴한다 — **펌웨어 코드는 전혀 안 바뀌고**, 클라이언트가 CCCD를
다시 켜면(`0x0001`) 그대로 재개된다. 추론 결과(`...57`)는 이 전력 절약 대상이
아니라 항상 구독 유지가 기본 — 인식 기능 자체가 이 스트림에 의존하기 때문.

**동작 원리 — CCCD 값이 저장되고 확인되는 경로**:

1. **구독 설정 방향 (클라이언트 → ESP32)**: 클라이언트가 CCCD에 Write하면
   ESP32 라이브러리(`BLEDescriptor.cpp::handleGATTServerEvent()`)가
   `ESP_GATTS_WRITE_EVT`를 받아서 `setValue(...)` 호출 → `BLE2902` 객체
   (`addDescriptor(new BLE2902())`로 붙여둔 것)의 2바이트 메모리에 값을
   저장함. `BLE2902`는 `BLEDescriptor`를 상속한 CCCD 전용 클래스로,
   `byte[0]`의 bit 0이 notification on/off를 뜻함. Characteristic마다(raw
   EMG/추론/가중치) 독립된 `BLE2902` 인스턴스가 있어서 구독 상태가 서로
   안 섞임.
2. **데이터 알림 방향 (ESP32 → 클라이언트)**: `notify()`가 호출될 때마다
   `p2902->getNotifications()`로 위에서 저장된 값을 읽고, 꺼져있으면
   `esp_ble_gatts_send_indicate` 호출 전에 `return`함.

이 저장/확인 로직 전부 ESP32 BLE 라이브러리(`arduino-cli core install
esp32:esp32`로 설치되는 서드파티 코드, 이 폴더 안에는 없음 — README `## 실행
방법 0번` 참고) 안에서 일어난다.

---

## 4. Characteristic 2 — 추론 결과

### 포맷

```
"classname|l0|l1|l2|l3|l4|l5"
```

| 필드 | 설명 | 예시 |
|------|------|------|
| classname | 최종 예측 제스처명 (3-vote cascade 적용) | `flexion` |
| l0~l5 | 각 클래스 softmax 확률 (소수점 3자리) | `0.020` |

### 실제 패킷 예시

```
"flexion|0.020|0.910|0.030|0.015|0.015|0.010"
```

### 클래스 인덱스 매핑

| 인덱스 | 클래스명 | 동작 |
|--------|----------|------|
| 0 | `rest` | 휴식 |
| 1 | `flexion` | 손목 아래로 구부리기 |
| 2 | `extension` | 손목 위로 젖히기 |
| 3 | `close` | 주먹 쥐기 |
| 4 | `supination` | 손바닥 위로 돌리기 |
| 5 | `pronation` | 손바닥 아래로 돌리기 |

### Android 파싱 예시 (Kotlin)

```kotlin
fun parsePrediction(bytes: ByteArray): Pair<String, FloatArray> {
    val msg = String(bytes, Charsets.UTF_8)
    val parts = msg.split("|")
    val className = parts[0]
    val logits = FloatArray(6) { i -> parts[i + 1].toFloat() }
    return Pair(className, logits)
}
```

### 발행 주기

64샘플마다 추론 1회 → **약 20Hz** (64 / 1281 Hz ≈ 50ms)

### 3-vote cascade

연속 3번의 추론 중 다수결로 최종 예측 결정. 노이즈성 순간 오분류 방지.

```
vote_threshold = 0.34 (= 1/3보다 조금 높음)
threshold 미달 시 직전 예측 유지
```

---

## 4-1. Characteristic 3 — 가중치 수신 (로컬 학습용)

클라이언트(폰)에서 로컬 학습으로 얻은 새 가중치를 암밴드에 전달하기 위한
Characteristic. **구현·실기기 검증 완료** — 재학습마다 펌웨어 전체를
재컴파일/재플래시할 필요 없이, 가중치 파일(53,304 bytes)만 교체하는 방식.

### 설계 배경

신경망의 구조(토폴로지 — 몇 층, 층마다 뉴런 몇 개)는 재학습해도 안 바뀌고
학습으로 얻은 숫자 값(가중치)만 매번 바뀐다는 점에 착안해, 구조는 펌웨어에
고정 컴파일해두고(`MODEL.h`의 `MODEL_TOPOLOGY`) 가중치만 별도 파일
(`/weights.bin`)로 분리해서 전송한다. 펌웨어는 최초 1회만 플래시하면 되고,
이후 재학습 결과는 파일 교체만으로 반영된다(§9 참고).

- **저장소로 LittleFS 사용**: SPIFFS는 Espressif가 deprecated 처리했고,
  전원이 갑자기 꺼져도 데이터가 덜 깨지는 wear-leveling 구조라 LittleFS를
  권장하기 때문.
- **전송 채널은 BLE가 기본, USB는 폴백(현재 실질적으로 미사용)**: USB
  Serial 경로(`receiveWeightsUsb()`)도 코드상 존재하지만, ESP32-S3 네이티브
  USB CDC(HWCDC)로 대용량 연속 스트림을 받을 때 멈추는 코어 버그
  ([espressif/arduino-esp32#10836](https://github.com/espressif/arduino-esp32/issues/10836))가
  있어 현재는 BLE만 실사용됨. BLE는 청크(244B)마다 Write 응답을 기다리는
  구조라 자연스럽게 페이싱되어 이 문제가 없음.
- **임시 파일 → CRC 검증 → rename 순서로 저장**: 전송 도중 연결이 끊겨도
  기존에 쓰던 가중치 파일이 반쯤 덮어써진 채로 남지 않도록, 항상
  `/weights.tmp`에 먼저 쓰고 CRC32 검증까지 통과했을 때만 `/weights.bin`으로
  교체(rename)한다. 검증 실패 시 임시 파일만 지우고 기존 `/weights.bin`은
  그대로 둔다.

### 식별

| 항목 | 값 |
|------|----|
| Characteristic UUID | `abcd1234-5678-1234-5678-abcdef123458` |
| 속성 | **WRITE**(클라이언트 → 암밴드) + **NOTIFY**(암밴드 → 클라이언트, 응답용) |
| CCCD | 포함, `0x0001` 쓰기로 NOTIFY 활성화 |

### 패킷 포맷

```
[4 bytes] MAGIC   : 0xDEADBEEF (little-endian)
[4 bytes] LENGTH  : payload 바이트 수 = 53,304 (uint32 little-endian)
[N bytes] PAYLOAD : W0 b0 W1 b1 W2 b2 means stds (float32)
[4 bytes] CRC32   : PAYLOAD 체크섬 (little-endian)
```

- 총 패킷 크기: 53,316 bytes
- BLE Write 청크 크기: **244 bytes**(MTU 247 − ATT 오버헤드 3)
- 필요한 Write 횟수: `⌈53316 / 244⌉ = 219`회, 순차 Write Request(응답 대기 후 다음 전송)
- USB 경로는 동일 페이로드를 연속 스트림으로 받음(115200 baud) — 위 코어 버그로
  현재 비권장

### 암밴드 응답 (Characteristic 3의 NOTIFY로 전송)

| 응답(UTF-8 텍스트) | 의미 |
|---|---|
| `OK:WEIGHTS` | 수신 완료, CRC 일치, `/weights.tmp` → `/weights.bin` 교체 성공 → 재부팅 |
| `ERR:SIZE` | LENGTH 필드와 실제 수신 바이트 수 불일치(`/weights.tmp` 삭제, 기존 파일 유지) |
| `ERR:CRC` | CRC32 불일치(`/weights.tmp` 삭제, 기존 파일 유지) |

클라이언트는 마지막 청크 전송 후 이 NOTIFY를 최대 10초 대기하며,
타임아웃/오류 시 실패 처리.

> **구현 시 주의(실기기에서 겪은 버그, §9 이력 참고)**: BLE `onWrite()` 콜백
> 안에서 `notify()` 직후 곧바로 `delay()`나 `ESP.restart()` 같은 블로킹
> 호출을 하면, 그 write에 대한 ATT 확인 응답이 클라이언트에게 영원히 안 감
> (콜백이 return해야 스택이 확인 응답을 실제로 내보낼 수 있음). 재부팅처럼
> 지연이 필요한 작업은 플래그만 세팅하고 `loop()` 밖에서 처리할 것.

---

## 5. 타이밍

| 항목 | 값 |
|------|----|
| 샘플링 주기 | 781µs → **약 1281 Hz** |
| raw EMG Notify 주기 | 20샘플 × 781µs = 15.6ms → **약 64 Hz** |
| 추론 Notify 주기 | 64샘플 × 781µs = 50ms → **약 20 Hz** |
| MTU 요청 | 247 bytes (160바이트 패킷 여유 있음) |

---

## 6. 연결 파라미터

| 항목 | 값 |
|------|----|
| Connection interval min | `0x06` → 7.5ms |
| Connection interval max | `0x0C` → 15ms |
| Slave latency | 0 |

Android에서 `requestConnectionPriority(CONNECTION_PRIORITY_HIGH)` 호출 권장.

---

## 7. 채널-핀 매핑

| 채널 | ADC 핀 |
|------|--------|
| CH0 | GPIO 9 |
| CH1 | GPIO 10 |
| CH2 | GPIO 7 |
| CH3 | GPIO 8 |
| CH4 | GPIO 11 |
| CH5 | GPIO 12 |
| CH6 | GPIO 17 |
| CH7 | GPIO 18 |

패킷 내 CH0~CH7 순서는 위 핀 순서와 동일.

---

## 8. 연결 끊김 동작

- 펌웨어가 연결 끊김 시 `BLEDevice::startAdvertising()` 자동 재호출
- 클라이언트 쪽 정책(참고, 펌웨어와 무관): 이 프로젝트의 Android 앱은 자동
  재연결을 시도하지 않고 사용자가 다시 스캔하도록 BLE 스캔 화면으로 이동함

---

## 9. 모델 플래시 방식

> 2026-07-20 갱신 — 아래 §4-1(BLE 가중치 수신)이 실제로 구현·검증된 현재 방식임.
> 이전엔 이 섹션이 "학습마다 펌웨어 전체를 재컴파일해서 USB로 재플래시"하던
> 구시대 흐름을 그대로 기록하고 있었는데, 실제로는 그 방식으로 간 적 없이
> `LOCAL_MIGRATION.md`가 설계한 대로 처음부터 BLE 가중치 전송으로 구현됨 — 내용을
> 현재 코드(`MODEL.h`, `nn.cpp::loadFromLittleFS()`) 기준으로 다시 씀.

### 개요

TFLite도, "모델을 통째로 C 헤더 파일로 변환해서 매번 재플래시"하는 방식도 안 씀.
**신경망 구조(토폴로지)는 컴파일 시점에 고정**되고, **재학습마다 바뀌는 가중치만**
BLE로 전송해서 LittleFS에 저장 → 재부팅 시 로드하는 방식.

### MODEL.h — 이제 고정 파일, 재생성 안 함

`firmware/exo_armband_hybrid/MODEL.h`는 토폴로지/활성화 함수 상수만 담고
있고, **재학습해도 이 파일은 안 바뀜**:
```c
const int MODEL_N_LAYERS = 4;
const int MODEL_TOPOLOGY[4] = { 132, 64, 64, 6 };
const int MODEL_ACTIVATIONS[3] = { 0, 0, 1 };  // RELU, RELU, SOFTMAX
```
가중치 배열은 여기 없음 — 부팅 시 `NeuralNet::loadFromLittleFS()`가
`/weights.bin`에서 읽어옴(§4-1). **`means.h`/`stds.h` 같은 별도 헤더 파일도
더 이상 없음** — StandardScaler의 mean/std는 `weights.bin` 페이로드 안에
가중치와 함께 이어붙어서 옴(`W0 b0 W1 b1 W2 b2 means stds` 순서, §4-1 참고).

### 실제 펌웨어 플래시(USB, arduino-cli)가 필요한 시점 — 최초 1회뿐

```
[최초 셋업, 새 암밴드마다 딱 1회]
arduino-cli로 exo_armband_hybrid 컴파일(FQBN에 CDCOnBoot=cdc 필수)
    ↓ USB
ESP32-S3에 업로드
    ↓
[암밴드] 부팅 → LittleFS.begin(true)로 빈 파일시스템 자동 포맷
    ↓ /weights.bin 아직 없음
[암밴드] "no weights — inference disabled" — 부팅/raw EMG 송신은 정상,
         추론 결과만 비활성
    ↓
[이후, 재학습할 때마다] Android가 §4-1 프로토콜로 가중치(53,304B)만 BLE 전송
    ↓
[암밴드] /weights.tmp에 받고 CRC 통과 시 /weights.bin으로 교체 → 재부팅
    ↓
[암밴드] 새 가중치로 추론 시작 (재플래시 없음)
```

**재플래시가 다시 필요한 경우는 딱 3가지뿐**: (1) 새 암밴드 최초 셋업, (2)
`MODEL_TOPOLOGY` 자체를 바꿀 때(예: 132→64→64→6이 아닌 다른 신경망 구조로
설계 변경), (3) 펌웨어 로직(전처리, BLE 프로토콜 등) 자체를 수정할 때. 그냥
"유저가 재학습했다"는 여기 해당 안 됨 — §4-1의 BLE 가중치 전송만으로 끝남.

---

## 10. 펌웨어 내부 추론 파이프라인

암밴드가 BLE로 추론 결과를 쏘기 위해 내부적으로 수행하는 과정.

### 상수

```c
WINDOW_SIZE     = 128       // 추론 윈도우 샘플 수
INFER_HOP       = 64        // 추론 간격 (50% overlap)
N_CHANNEL       = 8
N_FEATURES      = 132       // 피처 벡터 차원
SAMPLING_FREQ   = 1200 Hz
ENVELOPE_KERNEL = 12        // moving average 커널 크기
```

### 처리 순서

```
getEMG() → circular buffer (128×8) 누적
    64샘플마다 runInference() 호출
        ↓
1. applyBandpass()
   - Butterworth 4차 BP 필터 (35~300 Hz)
   - Direct Form II Transposed (causal, Python lfilter와 동일)
   - 필터 계수: preprocessor.cpp의 bp_b[], bp_a[]
   - 초기 상태: lfilter_zi(b, a) * x[0]

2. applyRectify()
   - abs() 취하기

3. applyEnvelope()
   - 이동평균 (kernel=12), circular buffer 방식
   - 윈도우 시작마다 상태 리셋 (Python np.convolve와 동일)

4. extractClassicFeatures()  → 88차원
   [시간영역, 피처 기준 묶음, 인덱스 0..47]
     MAV   × 8ch (idx 0..7)
     MaxAV × 8ch (idx 8..15)
     STD   × 8ch (idx 16..23)
     RMS   × 8ch (idx 24..31)
     WL    × 8ch (idx 32..39)
     SSC   × 8ch (idx 40..47)
   [주파수영역, 채널 기준 묶음, 인덱스 48..87]
     ch0: MeanPow, TotalPow, MeanFreq, MedianFreq, PeakFreq (idx 48..52)
     ch1: 위와 동일 (idx 53..57)
     ...
     ch7: 위와 동일 (idx 83..87)

5. extractTSDFeatures()  → 44차원
   [공분산 상삼각, 인덱스 88..123]
     (0,0),(0,1),(0,2),...,(0,7),
     (1,1),(1,2),...,(1,7),
     ...
     (7,7) → 총 36개
   [채널별 에너지, 인덱스 124..131]
     ch0~ch7 → 8개
   TSD 서브윈도우: win=96샘플, inc=48샘플, lam=0.1
   np.cov 기준 ddof=1 (n-1로 나눔)

6. standardize()
   - (x / std) - (mean / std) 형태로 계산 (수치 안정성)
   - mean/std는 `/weights.bin`에서 로드되어 메모리에 유지됨(§4-1, §9 —
     별도 헤더 파일(means.h/stds.h)로 존재하지 않음)

7. nn.predict()
   - Dense(132→64, ReLU) → Dense(64→64, ReLU) → Dense(64→6, Softmax)
   - 가중치도 `/weights.bin`에서 로드되어 메모리에 유지됨(§4-1, §9 —
     MODEL.h에는 토폴로지 상수만 있고 가중치 배열은 없음)

8. 3-vote cascade → BLE Notify (Characteristic ...57)
```

---

## 부록 — 피처 벡터 132개 상세 인덱스

| 인덱스 | 피처 | 채널 |
|--------|------|------|
| 0~7 | MAV | ch0~ch7 |
| 8~15 | MaxAV | ch0~ch7 |
| 16~23 | STD | ch0~ch7 |
| 24~31 | RMS | ch0~ch7 |
| 32~39 | WL | ch0~ch7 |
| 40~47 | SSC | ch0~ch7 |
| 48~52 | MeanPow, TotalPow, MeanFreq, MedianFreq, PeakFreq | ch0 |
| 53~57 | 위와 동일 | ch1 |
| 58~62 | 위와 동일 | ch2 |
| 63~67 | 위와 동일 | ch3 |
| 68~72 | 위와 동일 | ch4 |
| 73~77 | 위와 동일 | ch5 |
| 78~82 | 위와 동일 | ch6 |
| 83~87 | 위와 동일 | ch7 |
| 88~123 | 공분산 상삼각 (i,j), i≤j | ch0~ch7 |
| 124~131 | 채널별 에너지 | ch0~ch7 |
