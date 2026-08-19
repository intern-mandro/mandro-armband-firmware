# mandro 펌웨어 (단독 폴더)

`mandro-backend`에서 펌웨어 관련 파일만 뽑아낸 폴더입니다. 서버(FastAPI)/DB/
Docker 등 앱 관련 파일은 전부 제외되어 있어서, 이 폴더만으로 바로 컴파일+플래시가
가능합니다.

## 구성

- `firmware/exo_armband_hybrid/` — ESP32-S3 암밴드 펌웨어 소스 (arduino-cli 프로젝트).
  빌드/업로드 방법은 바로 아래 "실행 방법" 참고.
- `docs/FIRMWARE_PROTOCOL.md` — BLE 프로토콜 상세 스펙(Characteristic UUID, 패킷 포맷,
  가중치 전송 프로토콜, 펌웨어 내부 추론 파이프라인 등 전부)

## 실행 방법

Windows PowerShell 기준. **명령어는 한 줄씩** 실행한다.
0~1단계는 최초 1회만, 이후 코드를 고칠 때는 2~4단계만 반복한다.

### 0. arduino-cli 설치

```powershell
winget install ArduinoSA.CLI
```

설치 후 **PowerShell 창을 닫고 새로 연다.** `arduino-cli version`이 찍히면 성공.

### 1. 초기 설정 + 코어·라이브러리 설치

```powershell
arduino-cli config init
arduino-cli config set directories.data C:\arduino-cli-data
arduino-cli config set directories.downloads C:\arduino-cli-data\staging
arduino-cli config set directories.user C:\arduino-cli-data\user
arduino-cli config set board_manager.additional_urls https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "Adafruit NeoPixel"
arduino-cli lib install "Adafruit BNO055"
```

설치에 수 분 걸린다. `directories.*`를 영문 경로로 지정하는 이유는 아래 참고.

### 1-1. NimBLE BLEClient 패치 (필수, 코어 새로 설치할 때마다)

```powershell
powershell -File patches\apply-bleclient-patch.ps1
```

방금 설치한 esp32 코어의 `BLEClient.cpp`는 스톡 상태라 MASTER 모드에서 로봇 의수에
BLE 클라이언트로 접속할 때 조용히 실패하는 문제가 있다 (NimBLE이 연결 직후 MTU
교환을 자체적으로 먼저 끝내버리면 `BLE_HS_EALREADY`가 반환되는데, 스톡 코드는
이걸 다른 에러와 똑같이 취급해 정상 연결도 실패로 처리한다). 이 스크립트가 코어
설치 경로를 자동으로 찾아 `patches/BLEClient.ealready-mtu.patch`를 적용한다.
이미 적용돼 있으면 아무것도 안 하고 넘어간다 — 자세한 내용은
[`patches/BLEClient.ealready-mtu.patch`](patches/BLEClient.ealready-mtu.patch) 상단 주석 참고.

### 2. 컴파일

```powershell
cd C:\Intern\mandro-final_Armband_Android\mandro-firmware
arduino-cli compile --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc" --build-path firmware/exo_armband_hybrid/build firmware/exo_armband_hybrid
```

- `cd` 경로 → **저장소를 클론한 실제 경로**
- 첫 빌드는 5~10분 걸리고 그동안 출력이 없다. 두 번째부터는 20~30초.
- 성공하면 `Sketch uses 1234567 bytes (xx%) ...`가 나온다.

### 3. 업로드

암밴드를 USB로 연결하고:

```powershell
arduino-cli board list
arduino-cli upload -p COM4 --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc" --input-dir firmware/exo_armband_hybrid/build firmware/exo_armband_hybrid
```

- `COM4` → **`board list`로 확인한 실제 포트 번호**

### 4. 동작 확인

```powershell
arduino-cli monitor -p COM4 -c baudrate=115200
```

- `COM4` → **실제 포트 번호**
- 부팅 로그에 `BLE Advertising Started`가 나오면 정상. `Ctrl+C`로 종료.

---

## 이런 경우엔

### 계정명에 한글이 있는 경우

**1단계의 `directories.*` 설정이 이걸 위한 것이다.** 지정하지 않으면 코어가
`C:\Users\<한글계정>\AppData\Local\Arduino15\`에 깔리는데, 링커(`ld.exe`)가
그 경로를 읽지 못해 빌드가 **링크 단계에서** 실패한다.

```
ld.exe: cannot find -lespnow: No such file or directory
ld.exe: cannot find -lstdc++: No such file or directory
collect2.exe: error: ld returned 1 exit status
```

`error:`로 시작하는 C++ 에러 없이 `cannot find -l...`만 나오면 이 경우다.
**코드 문제가 아니라 경로 문제다.**

이미 한글 경로에 설치해버렸다면 이 순서로 복구한다.

```powershell
arduino-cli config set directories.data C:\arduino-cli-data
arduino-cli config set directories.downloads C:\arduino-cli-data\staging
arduino-cli config set directories.user C:\arduino-cli-data\user
arduino-cli config dump
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "Adafruit NeoPixel"
arduino-cli lib install "Adafruit BNO055"
Remove-Item -Recurse -Force firmware\exo_armband_hybrid\build
```

`config dump`에서 `directories` 3개가 전부 `C:\arduino-cli-data\...`로 바뀌었는지
확인한 뒤 2단계(컴파일)를 다시 한다. 코어를 새로 받으므로 수 분 걸린다.
`build/`를 지우는 이유는 예전 산출물에 읽지 못하는 경로가 박혀 있기 때문이다.

**예전 설치 폴더 삭제는 선택 사항이다.** 경로만 바꾸면 빌드는 정상 동작하고,
옛 폴더는 디스크(1~2GB)만 차지한다. 정리하려면:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Arduino15"
Remove-Item -Recurse -Force "$env:USERPROFILE\Documents\Arduino\libraries"
```

- `$env:` 변수를 쓰면 한글 계정명을 직접 입력하지 않아도 된다.
- **Arduino IDE도 함께 쓴다면 지우지 말 것.** IDE는 `Arduino15`를 공유하므로
  삭제하면 IDE 쪽 코어·라이브러리를 다시 받아야 한다.

컴파일 시 `--build-path`를 주는 것도 같은 이유다. 둘 다 필요하다.

### MASTER 모드에서 로봇 의수 연결이 계속 실패하는 경우

시리얼 모니터에 `MTU exchange error` 로그가 찍히면서 의수(칩센)에 연결이 안 되면
1-1단계의 BLEClient 패치가 안 적용된 것이다. 코어를 새로 설치했거나(`arduino-cli
core install esp32:esp32`을 다시 돌렸거나) 다른 PC에서 처음 세팅하는 경우 흔히
빠뜨린다. `powershell -File patches\apply-bleclient-patch.ps1`을 실행하고
(컴파일이 아니라 코어 파일 자체를 고치는 것이므로 재컴파일은 필요 없다) 다시
시도한다.

### `fatal error: Adafruit_NeoPixel.h: No such file or directory`

1단계의 `lib install`을 건너뛴 경우다. 펌웨어가 LED(NeoPixel)와 IMU(BNO055)를
쓰기 때문에 필요하다. `Adafruit BNO055`를 깔면 의존성인 `Adafruit Unified Sensor`,
`Adafruit BusIO`가 함께 설치된다.

### `Error finding build artifacts: could not find a valid build artifact`

업로드할 산출물이 없다는 뜻이다. 둘 중 하나다.

- `--input-dir`가 컴파일 때의 `--build-path`와 다름 → 경로를 맞춘다
- **컴파일이 실패했는데 업로드를 시도함** → 2단계부터 다시 한다.
  `build/`에 `.bootloader.bin`만 있고 `.ino.bin`·`.elf`가 없으면 이 경우다.

### 컴파일이 멈춘 것 같은 경우

첫 빌드는 ESP32 코어와 BLE 라이브러리를 전부 컴파일하므로 **5~10분 동안 출력이
없다.** 정상이다. 확인하려면 다른 터미널에서 `build/` 아래 `.o` 파일이 늘어나는지
보거나, 작업 관리자에서 `xtensa-esp32s3-elf-g++` 프로세스를 확인한다.
진행 상황을 보려면 `-v`를 붙인다(출력이 매우 길어짐).

### 첫 부팅에 `nvs_open failed`가 뜨는 경우

페어링 정보를 저장할 NVS 네임스페이스가 아직 없어서 나는 로그다. **무해하다.**
페어링을 한 번 수행하면 이후에는 나오지 않는다.

---

## 참고

이 펌웨어는 **최초 1회만** USB로 플래시하면 되고, 이후 재학습한 모델 가중치
교체는 클라이언트 앱이 BLE(또는 USB)로 `weights.bin`만 전송하는 방식이라
전체 재플래시가 필요 없다 — 자세한 프로토콜은
[`docs/FIRMWARE_PROTOCOL.md`](docs/FIRMWARE_PROTOCOL.md) §4-1/§9 참고.
토폴로지나 펌웨어 로직 자체를 바꿀 때만 위 컴파일+업로드 과정을 다시 하면 됨.
