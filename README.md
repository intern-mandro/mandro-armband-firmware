# mandro 펌웨어 (단독 폴더)

`mandro-backend`에서 펌웨어 관련 파일만 뽑아낸 폴더입니다. 서버(FastAPI)/DB/
Docker 등 앱 관련 파일은 전부 제외되어 있어서, 이 폴더만으로 바로 컴파일+플래시가
가능합니다.

## 구성

- `firmware/exo_armband_hybrid/` — ESP32-S3 암밴드 펌웨어 소스 (arduino-cli 프로젝트).
  빌드/업로드 방법은 바로 아래 "실행 방법" 참고.
- `docs/FIRMWARE_PROTOCOL.md` — BLE 프로토콜 상세 스펙(Characteristic UUID, 패킷 포맷,
  가중치 전송 프로토콜, 펌웨어 내부 추론 파이프라인 등 전부)

## 실행 방법 (펌웨어 컴파일 + 업로드)

모든 명령은 이 폴더(`mandro-firmware/`) 루트에서 실행한다 (경로가 전부 상대경로임).

### 0. 최초 1회만 — arduino-cli + ESP32 보드 코어 설치

```bash
# arduino-cli 설치: https://arduino.github.io/arduino-cli/ 참고 후 PATH 등록

arduino-cli config init
arduino-cli config set board_manager.additional_urls https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
```

### 1. 컴파일

```bash
arduino-cli compile \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc" \
  --build-path firmware/exo_armband_hybrid/build \
  firmware/exo_armband_hybrid
```

- `--fqbn`의 `CDCOnBoot=cdc`는 필수 — 이게 없으면(기본값 Disabled) USB로 시리얼
  로그가 전혀 안 나온다(이 보드는 별도 USB-UART 브릿지 없이 네이티브 USB만 씀).
- `--build-path`를 꼭 지정할 것 — 안 주면 arduino-cli가 사용자 프로필 하위
  기본 캐시 경로를 쓰는데, 사용자명에 한글 등 비ASCII 문자가 있으면 링커가 그
  경로를 못 읽어 빌드가 실패할 수 있다(`collect2.exe: error: ld returned 1
  exit status`). `build/`는 `.gitignore`에 이미 포함되어 있어 커밋 안 됨.

### 2. 보드에 업로드(플래시)

암밴드를 USB로 연결한 뒤 포트 확인:
```bash
arduino-cli board list
```
(Windows는 보통 `COM3` 같은 이름, macOS/Linux는 `/dev/tty.usbmodem*` 등)

컴파일 산출물을 그대로 업로드(`--input-dir`는 위 컴파일 단계의 `--build-path`와
동일해야 함):
```bash
arduino-cli upload \
  -p <포트> \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc" \
  --input-dir firmware/exo_armband_hybrid/build \
  firmware/exo_armband_hybrid
```

### 3. (선택) 시리얼 로그 확인

```bash
arduino-cli monitor -p <포트> -c baudrate=115200
```
(`Ctrl+C`로 종료. `Serial.println()` 로그는 USB 케이블이 꽂혀 있을 때만 보임 —
BLE 연결 여부와는 무관)

### 참고

이 펌웨어는 **최초 1회만** USB로 플래시하면 되고, 이후 재학습한 모델 가중치
교체는 클라이언트 앱이 BLE(또는 USB)로 `weights.bin`만 전송하는 방식이라
전체 재플래시가 필요 없다 — 자세한 프로토콜은
[`docs/FIRMWARE_PROTOCOL.md`](docs/FIRMWARE_PROTOCOL.md) §4-1/§9 참고.
토폴로지나 펌웨어 로직 자체를 바꿀 때만 위 컴파일+업로드 과정을 다시 하면 됨.
