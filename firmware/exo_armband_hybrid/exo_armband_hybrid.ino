/*
 * exo_armband_hybrid.ino
 * ──────────────────────
 * Hybrid ESP32S3 sketch: original recorder + inline live inference.
 *
 * STARTS FROM: Exo_Armband_ESP32S3_8EMG_BLE_260516.ino (the recorder that
 * generated the training CSVs at ~1280 Hz, working hardware init).
 *
 * ADDS:
 *   - Circular window buffer of 128 samples x 8 channels (fed by getEMG)
 *   - Every 64 new samples (= 50% overlap), runs:
 *       preproc.process() -> nn.predict() -> argmax + 3-vote cascade
 *   - Prints prediction on Serial alongside the BLE notify
 *
 * EVERYTHING ELSE IS UNCHANGED FROM THE RECORDER:
 *   - Same hardware init (TR_ENABLE_PIN=37 HIGH, BNO055 wait loop, NeoPixel)
 *   - Same getEMG() with analogRead 10-bit + (raw - 250) + clamp [0, 255]
 *   - Same BLE service / characteristic / notify timing (1280 Hz target)
 *   - Same power switch handling
 *
 * The model is the 6-class concat132 model trained at fs=1200 Hz on the
 * same hardware. Class names: rest, flexion, extension, close, supination,
 * pronation (see CLASS_NAMES / MODEL_TOPOLOGY — 4-class was the old 4ch model).
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <LittleFS.h>
#include <Preferences.h>    // 로봇 의수 MAC 주소를 ESP32의 NVS에 저장
#include <esp_system.h>     // esp_reset_reason() — 부팅 직후 리셋 원인 로깅용

// connId는 있는데 isConnected=false인 반쪽짜리 링크(=BLEClient가 놓친 상태)를
// forceDisconnectChipsen()에서 ble_gap_terminate()로 직접 끊기 위해 필요.
#if defined(CONFIG_NIMBLE_ENABLED)
#include <host/ble_hs.h>
#include <host/ble_gap.h>
#include <host/ble_att.h>   // dumpConnDesc()에서 ble_att_mtu()로 실제 MTU를 읽기 위해 필요
#include <nimble/ble.h>
#endif

#include "preprocessor.h"
#include "nn.h"

#define WEIGHTS_PATH     "/weights.bin"
#define WEIGHTS_TMP_PATH "/weights.tmp"

#define PAIR_HDR         0xE0
#define PAIR_PACKET_LEN  8       // 헤더 1B + MAC 6B + 체크섬 1B = 8B
#define PAIR_HDR_CLEAR   0xE1    // 저장된 마스터 MAC 삭제 명령 (1바이트 단독 write)

// =========================
// 설정 (UNCHANGED FROM RECORDER)
// =========================

#define SAMPLES_PER_PACKET 20
#define SAMPLE_SIZE        8
#define PACKET_SIZE        (SAMPLES_PER_PACKET * SAMPLE_SIZE)

// 가중치 수신 프로토콜 (docs/FIRMWARE_PROTOCOL.md 4-1)
#define WEIGHT_MAGIC        0xDEADBEEFUL
#define WEIGHTS_TOTAL_BYTES 53304UL  // W0 b0 W1 b1 W2 b2 means stds (float32)


static const char* SERVICE_UUID              = "12345678-1234-1234-1234-1234567890ab";
static const char* CHARACTERISTIC_UUID       = "abcd1234-5678-1234-5678-abcdef123456";  // raw samples
static const char* CHARACTERISTIC_UUID_PRED  = "abcd1234-5678-1234-5678-abcdef123457";  // predictions
static const char* CHARACTERISTIC_UUID_WEIGHTS = "abcd1234-5678-1234-5678-abcdef123458"; // 가중치 수신
static const char* CHARACTERISTIC_UUID_PAIR = "abcd1234-5678-1234-5678-abcdef123459";   // 페어링용 

// MARK7(로봇 의수) BLE 모듈. MAC: 5C:F2:86:49:3A:AA (AT 명령어로 실측 확인됨 —
// 예전엔 3C:A5:51:99:BB:5F로 적혀있었는데 그건 다른 개체/오기였다.)
// 이 MAC은 NVS(pairing/master_mac)에 등록해서 쓴다 — 하드코딩 fallback 없음.
//
// 실물 GATT 구조 — reference/BLE_Scanner.png (기기명 CHIPSEN, NOT BONDED):

//   0000FFF0  service (Primary)
//     ├ 0000FFF1  NOTIFY (+ CCCD 0x2902)     ← 의수 → 암밴드 (STATUS/ACK)
//     └ 0000FFF2  WRITE, WRITE NO RESPONSE   ← 암밴드 → 의수 (CMD)

static const char* MARK7_SERVICE_UUID = "0000fff0-0000-1000-8000-00805f9b34fb";
static const char* MARK7_CMD_UUID     = "0000fff2-0000-1000-8000-00805f9b34fb";  // write
static const char* MARK7_STATUS_UUID  = "0000fff1-0000-1000-8000-00805f9b34fb";  // notify

// ── 의수 명령 프로토콜 (hand.py:1-8, build_cmd() 기준) ──────────────────
// CMD 12byte: HDR(0xFF) + finger_sel + speed + current + pos[6] + dir + XOR
//   XOR 범위는 byte 1~10 (헤더 제외), 결과를 byte 11에 넣는다.
//   payload 12B는 MTU 안에 들어가므로 한 번의 write로 전송된다.
//
// finger_sel 비트마스크: bit5=F1(엄지) bit4=F2 bit3=F3 bit2=F4 bit1=F5 bit0=F6(엄지외전)
//   0x1E = F2~F5,  0x3F = 전체
// dir: 1=GRASP, 2=RELEASE, 4=RESET_POWER
//
// ※ 현재 runInference()는 1바이트만 보내고 있어서 의수가 절대 해석하지 못한다.
//   12byte 패킷 생성기로 교체해야 한다 (아래 TODO).
#define MARK7_HDR_CMD      0xFF
#define MARK7_CMD_LEN      12
#define MARK7_DIR_GRASP    1
#define MARK7_DIR_RELEASE  2
#define MARK7_DIR_RESET    4
#define MARK7_STATUS_LEN   36   // 의수 → 암밴드 STATUS (hand.py::STATUS_BUF_SIZE)

// ── 벤치 테스트용 스윕 ──────────────────────────────────────────────────
// 1로 두면 추론 결과와 무관하게 rest/close CMD를 주기적으로 번갈아 보낸다.
// 용도: CHIPSEN 모듈을 PC USB에 물려놓고 시리얼 툴로 12바이트가 실제로 나오는지
// 확인할 때. runInference()는 /weights.bin이 있어야만 돌기 때문에(loop()의
// nn.isLoaded() 조건), 가중치를 안 올린 상태에서는 제스처 경로로 아무것도 안
// 나간다 — 그때 이걸 켜면 BLE 경로만 따로 검증된다.
#define MARK7_TEST_SWEEP     0
#define MARK7_TEST_SWEEP_MS  1500

// 모드 전환 타이밍
#define SCAN_SECONDS           2      // 스캔 1회 지속 (loop 블로킹)
#define MASTER_ENTRY_DELAY_MS  1000   // MASTER 진입 후 첫 스캔까지 유예
#define HAND_RETRY_MS_MIN      1000
#define HAND_RETRY_MS_MAX      30000
#define HAND_KEEPALIVE_MS      500
#define HAND_STATUS_TIMEOUT_MS 3000   // 이 시간 동안 STATUS(FFF1) notify가 한 번도 안 오면 경고

// =========================
// NEW: 가중치 수신 — 공통 로직 (USB/BLE 공용)
// =========================
// CRC 검증 + 임시파일 저장 + 실제 파일 교체까지 여기서 처리. USB(Serial
// 스트림)와 BLE(청크 재조립) 둘 다 PAYLOAD+CRC를 다 모으고 나면 이 함수를
// 호출한다 — "다 모인 뒤 저장하는 방식"은 두 채널이 동일하고, "어떻게
// 모으는지"만 다르기 때문에 이 부분만 공유.

enum WeightSaveResult { WSAVE_OK, WSAVE_ERR_CRC, WSAVE_ERR_WRITE };

// 표준 CRC-32 (poly 0xEDB88320, init/최종 XOR 0xFFFFFFFF) — zlib, PNG,
// Android의 java.util.zip.CRC32와 동일한 결과를 낸다 (BleManager.kt,
// UsbRepositoryImpl.kt에서 이 클래스로 계산한 값과 맞춰야 하므로 ESP32
// ROM의 crc32_le()는 쓰지 않고 직접 구현 — ROM 함수는 초기/최종 XOR
// 컨벤션이 애매해서 값이 안 맞을 위험이 있음).
uint32_t crc32_calc(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) {
      crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320UL : (crc >> 1);
    }
  }
  return crc ^ 0xFFFFFFFFUL;
}

// CRC 검증 후 /weights.tmp에 썼다가 /weights.bin으로 교체(rename)한다.
// 실패하면 /weights.tmp만 지우고 기존 /weights.bin은 그대로 둔다 (안전장치).
// 재부팅은 호출자가 한다 (USB는 Serial.println 후, BLE는 notify 후 —
// 응답을 실제로 다 보낸 뒤에 재부팅해야 상대방이 결과를 받을 수 있음).
WeightSaveResult saveWeightsIfCrcOk(const uint8_t* payload, uint32_t totalBytes, uint32_t crcRecv) {
  uint32_t t0 = millis();
  uint32_t crcCalc = crc32_calc(payload, totalBytes);
  Serial.printf("# WSAVE: CRC 계산 %lu ms\n", (unsigned long)(millis() - t0));

  if (crcCalc != crcRecv) {
    Serial.printf("# 가중치 CRC 불일치 (calc=%08X recv=%08X)\n", crcCalc, crcRecv);
    return WSAVE_ERR_CRC;
  }

  t0 = millis();
  File f = LittleFS.open(WEIGHTS_TMP_PATH, "w");
  if (!f) {
    Serial.println("# WSAVE: LittleFS.open 실패");
    return WSAVE_ERR_WRITE;
  }
  size_t written = f.write(payload, totalBytes);
  f.close();
  Serial.printf("# WSAVE: flash write %u/%u bytes, %lu ms\n",
                (unsigned)written, totalBytes, (unsigned long)(millis() - t0));
  if (written != totalBytes) {
    LittleFS.remove(WEIGHTS_TMP_PATH);
    return WSAVE_ERR_WRITE;
  }

  LittleFS.remove(WEIGHTS_PATH);  // 있으면 지움 (없어도 무해)
  if (!LittleFS.rename(WEIGHTS_TMP_PATH, WEIGHTS_PATH)) {
    LittleFS.remove(WEIGHTS_TMP_PATH);
    return WSAVE_ERR_WRITE;
  }
  return WSAVE_OK;
}

const char* weightSaveResultStr(WeightSaveResult r) {
  switch (r) {
    case WSAVE_OK:        return "OK:WEIGHTS";
    case WSAVE_ERR_CRC:   return "ERR:CRC";
    case WSAVE_ERR_WRITE: return "ERR:WRITE";
  }
  return "ERR:WRITE";
}


// =========================
// BLE - 슬레이브(Server) 측
// =========================

BLECharacteristic* pCharacteristic;         // raw samples
BLECharacteristic* pCharacteristicPred;     // predictions
BLECharacteristic* pCharacteristicWeights;  // 가중치 수신
BLECharacteristic* pCharacteristicPair;     // 페어링용 MAC 수신 

volatile bool deviceConnected = false;

// ── Pairing: 로봇 의수 MAC 저장 ──────────────────
Preferences pairPrefs;
uint8_t masterMac[6] = {0};
bool hasMasterMac = false;

// PAYLOAD 검증(형식+체크섬)까지만 onWrite()에서 동기로 하고, 실제 NVS 쓰기/삭제는
// loop()로 미룸 — WeightsCharCallbacks/pendingWeightSave와 같은 이유(아래 그
// 선언부 주석 참고): onWrite()는 BLE 스택 태스크 컨텍스트라 그 안에서 동기 flash
// 쓰기를 하면 연결 인터벌을 놓치거나 브라운아웃/워치독 재부팅까지 갈 수 있음.
// 페어링 앱(안드로이드/PC) 쪽에서 "OK:PAIR"/"OK:CLEAR" notify가 가끔 늦거나
// 아예 안 오는 증상이 있었는데, 원인이 이거였을 가능성이 높음(2026-08-22).
static volatile bool pendingPairSave = false;
static uint8_t pendingMasterMac[6] = {0};
static volatile bool pendingPairClear = false;

// ESP32에 저장된 로봇 의수 MAC을 불러오고, 저장 여부를 확인하는 함수.
// NVS(pairing/master_mac)에 저장된 값만 쓴다 — 하드코딩 fallback 없음.
// 저장된 게 없으면 hasMasterMac이 false로 남고, handLinkTick()은
// PC/폰 페어링 앱으로 MAC이 등록될 때까지 아무 것도 하지 않는다.
void loadMasterMac() {
  pairPrefs.begin("pairing", true);

  size_t n = pairPrefs.getBytes(
    "master_mac",
    masterMac,
    sizeof(masterMac)
  );

  pairPrefs.end();

  hasMasterMac = (n == sizeof(masterMac));

  if (hasMasterMac) {
    Serial.printf(
      "# PAIR: 저장된 마스터(NVS) %02X:%02X:%02X:%02X:%02X:%02X\n",
      masterMac[0],
      masterMac[1],
      masterMac[2],
      masterMac[3],
      masterMac[4],
      masterMac[5]
    );
  } else {
    Serial.println("# PAIR: 저장된 마스터 없음 — 페어링 앱으로 MAC을 등록해야 함");
  }
}

// PC에게 Pairing 결과를 보내는 함수
void notifyPairResult(const char* msg) {
  if (!deviceConnected) {
    return;
  }

  pCharacteristicPair->setValue(
    (uint8_t*)msg, 
    strlen(msg)
  );

  pCharacteristicPair->notify();
}

// ── BLE 가중치 수신 상태 ──────────────────────────────────────────────
// Android(BleManager.kt::sendWeights())가 244바이트씩 순차 Write Request로
// 보내는 걸 여기에 순서대로 이어붙인다. MAGIC+LENGTH+PAYLOAD+CRC32가 다
// 모이면 saveWeightsIfCrcOk()를 호출.
#define BLE_WEIGHT_BUF_MAX (4 + 4 + WEIGHTS_TOTAL_BYTES + 4)
static uint8_t bleWeightBuf[BLE_WEIGHT_BUF_MAX];
static size_t  bleWeightBufLen = 0;

// 마지막 청크 도착 시각. 연결은 살아 있는데 앱이 전송을 중단하면(앱 크래시,
// 사용자 이탈) 버퍼가 영구히 남고, 다음 전송의 첫 청크가 거기에 이어붙는다.
// 이때 버퍼 앞의 **옛 매직넘버**가 그대로 남아 있어서 검사를 통과해버리므로
// 다음 전송이 한 번은 반드시 실패한다. 일정 시간 청크가 없으면 버린다.
#define BLE_WEIGHT_IDLE_TIMEOUT_MS 5000
static uint32_t bleWeightLastChunkMs = 0;

// PAYLOAD+CRC32가 다 모였다는 표시만 하는 플래그. 실제 CRC 검증과 53KB
// LittleFS.write()는 onWrite()가 아니라 loop()에서 한다 — onWrite()는 BLE
// 스택 자체의 태스크 컨텍스트에서 도는데, 그 안에서 53KB 동기 flash 쓰기를
// 하면 BLE 스택이 그동안 멈춰서 연결 인터벌(7.5~15ms)을 여러 번 놓치고,
// 브라운아웃/워치독으로 재부팅까지 발생한 사례가 있었다. loop()는 Arduino
// 메인 태스크라 여기서 블로킹돼도 BLE 스택 자체의 타이밍에는 영향이 없다
// (재부팅을 loop()로 미루는 pendingRestart와 같은 이유).
static volatile bool pendingWeightSave = false;

// characteristic 3의 NOTIFY로 결과 문자열 전송
void notifyWeightsResult(const char* msg) {
  if (!deviceConnected) return;
  pCharacteristicWeights->setValue((uint8_t*)msg, strlen(msg));
  pCharacteristicWeights->notify();
}

// BLE 가중치 수신 상태 초기화 (연결 끊기거나 오류 시 호출)
void resetBleWeightReceive() {
  bleWeightBufLen = 0;
  pendingWeightSave = false;   // 저장 대기 중이던 것도 함께 취소
}


// BLE ATT의 "write 확인 응답"은 onWrite() 콜백이 return해야 스택이 실제로
// 내보낼 수 있음 — 콜백 안에서 delay()+ESP.restart()로 블로킹해버리면 안드로이드가
// 그 write의 확인 응답을 영원히 못 받고 연결 전체 타임아웃(5초)까지 기다리다
// 실패 처리됨 (실기기 로그로 확인됨). 그래서 재부팅은 콜백 밖(loop())에서
// 예약 처리하고, 콜백 자체는 바로 return하게 함.
static bool pendingRestart = false;
static unsigned long restartAtMs = 0;

class WeightsCharCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override { // Android가 244바이트씩 Write 할 때마다 이 onWrite()가 호출됨.
    // 이미 저장 대기 중이면(직전 write로 다 모여서 loop()가 처리를 기다리는
    // 중) 버퍼를 더 건드리지 않는다 — 여기서 더 받아버리면 overflow 검사에
    // 걸려 resetBleWeightReceive()가 돌면서 loop()가 처리하기 전에 데이터가
    // 날아간다.
    if (pendingWeightSave) return;

    String chunk = c->getValue();
    const uint8_t* data = (const uint8_t*)chunk.c_str();
    size_t len = chunk.length();

    uint32_t now = millis();
    Serial.printf("# WCHUNK len=%u bufLen=%u dt=%lu heap=%u\n",
                   (unsigned)len, (unsigned)bleWeightBufLen,
                   (unsigned long)(now - bleWeightLastChunkMs), ESP.getFreeHeap());

    bleWeightLastChunkMs = now;   // 수신 타임아웃 감시용 (loop()에서 판정)

    if (bleWeightBufLen + len > BLE_WEIGHT_BUF_MAX) {
      Serial.println("# BLE: 가중치 수신 버퍼 초과 — 리셋");
      resetBleWeightReceive();
      notifyWeightsResult("ERR:SIZE");
      return;
    }
    // 들어온 청크를 bleWeightBuf에 이어붙임
    memcpy(bleWeightBuf + bleWeightBufLen, data, len);
    bleWeightBufLen += len;

    // 매직넘버: 처음 4바이트가 모이자마자 확인
    if (bleWeightBufLen >= 4) {
      uint32_t magic;
      memcpy(&magic, bleWeightBuf, 4);
      if (magic != WEIGHT_MAGIC) {
        Serial.println("# BLE: 매직넘버 불일치 — 리셋");
        resetBleWeightReceive();
        notifyWeightsResult("ERR:SIZE");
        return;
      }
    }

    // LENGTH: 8바이트가 모이면 총 기대 크기 계산 가능
    if (bleWeightBufLen < 8) return;

    uint32_t declaredLen;
    memcpy(&declaredLen, bleWeightBuf + 4, 4);
    if (declaredLen != WEIGHTS_TOTAL_BYTES) {
      Serial.println("# BLE: LENGTH 불일치 — 리셋");
      resetBleWeightReceive();
      notifyWeightsResult("ERR:SIZE");
      return;
    }

    size_t expectedTotal = 4 + 4 + declaredLen + 4;
    if (bleWeightBufLen < expectedTotal) return;  // 아직 다 안 옴, 다음 청크 기다림

    // PAYLOAD + CRC32까지 다 도착했다. CRC 검증과 53KB flash 쓰기는 여기서
    // 하지 않고 loop()에 넘긴다 — bleWeightBuf는 그대로 두고 표시만 세운다.
    // 이 콜백은 바로 return해서 이번 write의 ATT 확인 응답이 즉시 나가게 함.
    pendingWeightSave = true;
  }
};

class PairCharCallbacks : public BLECharacteristicCallbacks  {

  void onRead(BLECharacteristic* c) override {
    if (hasMasterMac) {
      c->setValue(masterMac, 6);
    } else {
      uint8_t empty = 0;
      c->setValue(&empty, 0);
    }
  }

  void onWrite(BLECharacteristic* c) override {

    const uint8_t* d = c->getData();
    size_t len = c->getLength();

    // 0. CLEAR 명령: 1바이트(0xE1)만 오면 저장된 마스터 MAC을 지운다.
    //    NVS 자체에서 키를 지워야 hasMasterMac이 정직하게 false가 되고
    //    onRead()가 "비어있음"을 돌려준다 — 00:00:.. 같은 가짜 MAC을
    //    저장하는 방식은 "페어링 안 됨"과 "그 MAC으로 페어링됨"을 구분 못
    //    하게 만들어서 쓰지 않는다.
    //    실제 NVS 삭제는 loop()에서 처리(위 pendingPairClear 선언부 주석 참고) —
    //    여기선 플래그만 세우고 바로 return, notify도 loop()가 지운 뒤에 보냄.
    if (len == 1 && d[0] == PAIR_HDR_CLEAR) {
      if (pendingPairSave || pendingPairClear) {
        Serial.println("# PAIR: 이전 요청 처리 대기 중 — CLEAR 무시");
        return;
      }
      pendingPairClear = true;
      return;
    }

    // 1. 길이 + 헤더 검사 (동기, 빠름 — flash 접근 없음)
    if (len != PAIR_PACKET_LEN || d[0] != PAIR_HDR) {

      Serial.printf(
        "# PAIR: 형식 불일치 (len=%u, hdr=%02X)\n",
        (unsigned)len,
        len ? d[0] : 0
      );

      notifyPairResult("ERR:PAIR_FORMAT");

      return;
    }

    // 2. XOR checksum 검사 (동기, 빠름)
    uint8_t chk = 0;

    for (int i = 1; i < PAIR_PACKET_LEN - 1; ++i) {
      chk ^= d[i];
    }

    if (chk != d[PAIR_PACKET_LEN - 1]) {

      Serial.println("# PAIR: 체크섬 불일치");

      notifyPairResult("ERR:PAIR_CRC");

      return;
    }

    // 3. MAC 6byte를 대기 버퍼에 복사만 해두고, 실제 NVS 저장/notify는 loop()로
    //    미룸(위 pendingPairSave 선언부 주석 참고).
    if (pendingPairSave || pendingPairClear) {
      Serial.println("# PAIR: 이전 요청 처리 대기 중 — PAIR 무시");
      return;
    }
    memcpy(pendingMasterMac, d + 1, 6);
    pendingPairSave = true;
  }
};

uint8_t txBuffer[PACKET_SIZE];
volatile uint32_t sampleCounter = 0;

// 모드 전환은 여기서 하지 않음 
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE Connected");
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE Disconnected");
    resetBleWeightReceive();  // 전송 도중 끊겼을 수 있으니 다음 연결은 깨끗한 상태로 시작
    BLEDevice::startAdvertising();
  }
};

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
Adafruit_NeoPixel strip = Adafruit_NeoPixel(4, 14, NEO_GRB + NEO_KHZ800);

const int periodMicro = 781;  // 1280 Hz

#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#define SDA_PIN 34
#define SCL_PIN 35
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29);

int q_int[4] = {0, 0, 0, 10000};
int calib = 0;

const int N_SAMPLE = 20;
const int N_SENSOR = 8;

int trEnablePin    = 37;
int powerSwitchPin = 36;
int voltagePin     = 13;

int sensorPin[N_SENSOR] = {9, 10, 7, 8, 11, 12, 17, 18};
int emg[N_SAMPLE][N_SENSOR] = {{0,},};
int emg_diff[N_SENSOR] = {0,};
const int led_order[N_SENSOR] = {1, 2, 3, 0};

int getEMG();
void getQuatFromBNO055();
void shutdownOnSwitch();


// ─────────────────────────────────────────────────────────────────────
// NEW: inference state
// ─────────────────────────────────────────────────────────────────────

Preprocessor preproc;
NeuralNet    nn;

static const int N_CLASSES  = 6;
static const int INFER_HOP  = 64;
static const int N_VOTE     = 3;
static const float VOTE_THRESHOLD = 0.34f;
static const char* CLASS_NAMES[N_CLASSES] = { "rest", "flexion", "extension", "close", "supination", "pronation" };

// Circular sample buffer (mirrors what BLE sends, AFTER the same -250+clamp)
static int16_t infer_buffer[WINDOW_SIZE][N_CHANNEL];
static int     infer_write_idx = 0;
static uint32_t infer_total_samples = 0;
static uint32_t infer_samples_since_last = 0;

// Vote ring
static int vote_buf[N_VOTE] = {0, 0, 0};
static int vote_count = 0;

// LATENCY tracking
static uint32_t infer_count = 0;
static uint32_t t_preproc_sum = 0, t_nn_sum = 0;
static uint32_t t_preproc_max = 0, t_nn_max = 0;
static int      infer_latency_count = 0;
static const int N_TIMING_INF = 20;

// 디버그 출력 스위치 (1로 바꾸면 켜짐). 이전엔 `if (false & <조건>)` 형태로
// 껐는데, `&`가 `==`/`>=`보다 우선순위가 낮아서 실제로는 `false & (조건)`이
// 아니라 컴파일러 경고 없이 항상 거짓인 죽은 코드였다. 의도를 명시적으로
// 드러내려고 상수 플래그로 바꿨다 (0이면 최적화 단계에서 통째로 빠진다).
#define DEBUG_ADC_STATS      1
#define DEBUG_INFER_LATENCY  0

// 모드 전환 (슬레이브 ↔ 마스터)
enum ArmbandMode { MODE_SLAVE, MODE_MASTER };
enum HandLinkState { HAND_IDLE, HAND_CONNECTING, HAND_READY };

static ArmbandMode mode = MODE_SLAVE;
static HandLinkState handState = HAND_IDLE;

static BLEClient* pHandClient = nullptr;
static BLERemoteCharacteristic* pHandCmd    = nullptr;  // CMD 송신 (FFF2 또는 FFE1)
static BLERemoteCharacteristic* pHandStatus = nullptr;  // STATUS 수신 (FFF1 또는 FFE1)
static BLEAdvertisedDevice* pHandFound = nullptr;

static volatile bool handConnected = false;
static uint32_t handNextTryMs = 0;
static uint32_t handRetryMs = HAND_RETRY_MS_MIN;
static uint32_t handReadyMs = 0;   // HAND_READY 진입 시각 — STATUS 무응답 워치독 기준점

void setStatusLed(uint8_t r, uint8_t g, uint8_t b) {
  for (int j = 0; j < 4; ++j) strip.setPixelColor(j, r, g, b);
  strip.show();
}

// GAP 이벤트 핸들러/connect() 디버그 로그가 공통으로 쓰는 헬퍼 — connection
// handle 하나로 저수준 NimBLE이 실제로 들고 있는 연결 상태(interval, MTU 등)를
// 조회해서 찍는다. BLEClient 래퍼의 ok/isConnected()와 무관하게, "진짜로 링크가
// 있는지"를 ble_gap_conn_find()로 직접 확인하기 위한 용도.
#if defined(CONFIG_NIMBLE_ENABLED)
void dumpConnDesc(uint16_t connHandle, const char* tag) {
  struct ble_gap_conn_desc desc;
  int rc = ble_gap_conn_find(connHandle, &desc);
  if (rc != 0) {
    Serial.printf(
      "### CONN_DESC [%s] handle=%u — ble_gap_conn_find rc=%d (%s) ###\n",
      tag, connHandle, rc, BLEUtils::returnCodeToString(rc)
    );
    return;
  }
  BLEAddress peer(desc.peer_ota_addr);
  Serial.printf(
    "### CONN_DESC [%s] handle=%u peer=%s itvl=%u latency=%u timeout=%u role=%u mtu=%u ###\n",
    tag, connHandle, peer.toString().c_str(),
    desc.conn_itvl, desc.conn_latency, desc.supervision_timeout, desc.role,
    ble_att_mtu(connHandle)
  );
}
#endif

// BLEDevice::setCustomGapHandler()로 등록하는 원시 GAP 이벤트 리스너 — BLEClient의
// 내부 처리를 대체하는 게 아니라(ble_gap_event_listener_register()는 리스너를
// 여러 개 등록 가능), 그 위에 관찰용으로 얹는 것. connect()가 FAIL인데 connId는
// 남는 상황을 status/handle 단위로 직접 보기 위함.
#if defined(CONFIG_NIMBLE_ENABLED)
int bleGapDebug(struct ble_gap_event* event, void* arg) {
  Serial.printf("### GAP EVENT type=%u ###\n", (unsigned)event->type);

  switch (event->type) {

    case BLE_GAP_EVENT_CONNECT:
      Serial.printf(
        "### GAP CONNECT: status=%d handle=%u ###\n",
        event->connect.status,
        event->connect.conn_handle
      );
      break;

    case BLE_GAP_EVENT_MTU:
      Serial.printf(
        "### GAP MTU: handle=%u mtu=%u ###\n",
        event->mtu.conn_handle,
        event->mtu.value
      );
      break;

    // ─────────────────────────────
    // 4. 연결 파라미터 변경 요청
    // ─────────────────────────────
    case BLE_GAP_EVENT_CONN_UPDATE_REQ:
      Serial.printf(
        "### GAP CONN_UPDATE_REQ: handle=%u"
        " min=%u max=%u latency=%u timeout=%u ###\n",
        event->conn_update_req.conn_handle,
        event->conn_update_req.peer_params->itvl_min,
        event->conn_update_req.peer_params->itvl_max,
        event->conn_update_req.peer_params->latency,
        event->conn_update_req.peer_params->supervision_timeout
      );
      break;

    // ─────────────────────────────
    // 5. 암호화 / 보안 상태 변화
    // ─────────────────────────────
    case BLE_GAP_EVENT_ENC_CHANGE:
      Serial.printf(
        "### GAP ENC_CHANGE: handle=%u status=%d (%s) ###\n",
        event->enc_change.conn_handle,
        event->enc_change.status,
        BLEUtils::returnCodeToString(event->enc_change.status)
      );
      dumpConnDesc(event->enc_change.conn_handle, "ENC_CHANGE");
      break;

    // ─────────────────────────────
    // 6. Passkey / Pairing 요청
    // ─────────────────────────────
    case BLE_GAP_EVENT_PASSKEY_ACTION:
      Serial.printf(
        "### GAP PASSKEY_ACTION: handle=%u action=%u ###\n",
        event->passkey.conn_handle,
        event->passkey.params.action
      );
      break;

    // ─────────────────────────────
    // 7. 연결 종료
    // ─────────────────────────────
    case BLE_GAP_EVENT_DISCONNECT:
      Serial.printf(
        "### GAP DISCONNECT: handle=%u reason=%d (%s) ###\n",
        event->disconnect.conn.conn_handle,
        event->disconnect.reason,
        BLEUtils::returnCodeToString(event->disconnect.reason)
      );
      break;

    // ─────────────────────────────
    // 8. 연결 종료 자체가 실패
    // ─────────────────────────────
    case BLE_GAP_EVENT_TERM_FAILURE:
      Serial.printf(
        "### GAP TERM_FAILURE: handle=%u status=%d (%s) ###\n",
        event->term_failure.conn_handle,
        event->term_failure.status,
        BLEUtils::returnCodeToString(event->term_failure.status)
      );
      break;

    default:
      // 위에서 첫 줄에 type 번호는 이미 찍힘
      break;
  }

  return 0;
}
#endif

// handState는 loop()만 쓰게 하고, 콜백은 handConnected만 건드린다. (writer를 하나로 유지 → 상태 경합 없음)
class HandClientCallbacks : public BLEClientCallbacks {
  void onConnect(BLEClient*) override {
    handConnected = true;
    Serial.println("### ARMBAND -> CHIPSEN CLIENT CONNECTED ###");
  }
  void onDisconnect(BLEClient*) override {
    handConnected = false;   // 재탐색 전이는 handLinkTick()이 감지해서 처리
    Serial.println("### ARMBAND -> CHIPSEN CLIENT DISCONNECTED ###");
  }
};
static HandClientCallbacks handClientCb;

// isConnected()==false인데 connId만 남아있는 반쪽짜리 링크는 BLEClient::disconnect()가
// (isConnected() 체크를 거치는 상위 래퍼라면) 놓칠 수 있다. ble_gap_terminate()는
// connection handle만 있으면 그 상태와 무관하게 무조건 종료를 시도하는 저수준
// NimBLE 호출이라, 여기서 그걸 직접 쓴다.
void forceDisconnectChipsen() {
  if (!pHandClient) return;

  uint16_t connId = pHandClient->getConnId();

#if defined(CONFIG_NIMBLE_ENABLED)
  if (connId != BLE_HS_CONN_HANDLE_NONE) {
    int rc = ble_gap_terminate(connId, BLE_ERR_REM_USER_CONN_TERM);
    Serial.printf("# CHIPSEN: force disconnect connId=%u rc=%d\n", (unsigned)connId, rc);
  }
#else
  if (pHandClient->isConnected()) {
    pHandClient->disconnect();
  }
#endif
}

// ── NimBLE 주소 바이트 순서 ──────────────────────────────────────────────
// 이 코어(esp32 3.3.11)는 build/sdkconfig에 CONFIG_NIMBLE_ENABLED=y로 빌드된다
// (Bluedroid 아님 — BLEDevice.h 등 API 이름은 같아도 내부는 NimBLE 래퍼다).
//
// NimBLE 분기에서 BLEAddress 내부 저장은 사람이 읽는 순서와 **반대**다:
//   - BLEScan.cpp:643  `BLEAddress advertisedAddress(disc.addr)` 가
//     `ble_addr_t.val`(wire order, LSB 옥텟이 먼저)을 그대로 memcpy
//   - BLEAddress.cpp   `toString()`이 NimBLE에서는 `m_address[5..0]` 순서로
//     출력해서 사람이 읽는 표기와 맞춰준다
//   → getNative()[i] == (사람이 읽는 순서의 MAC)[5-i]
//
// masterMac은 항상 사람이 읽는 순서(§4-2 "정순")로 들고
// 있으므로, getNative()와 그대로 memcmp하면 실제 기기와 절대 일치하지 않는다
// (MAC이 우연히 팔린드롬이 아닌 이상 100% 실패). 반드시 이 함수로 비교할 것.
bool nativeAddrMatchesHandMac(const uint8_t* native) {
  for (int i = 0; i < 6; i++) {
    if (native[i] != masterMac[5 - i]) return false;
  }
  return true;
}

// 저장된 MAC과 일치하는 광고만 잡는다.
static uint32_t handScanSeenCount = 0;   // 이번 스캔에서 (MAC 무관) 관측한 광고 총 개수 — "안 보임"과 "MAC 안 맞음"을 구분하는 용도

class HandScanCallbacks : public BLEAdvertisedDeviceCallbacks {
  // 코어 3.3.11의 시그니처는 **값 전달**이다 (BLEAdvertisedDevice.h:215
  // `virtual void onResult(BLEAdvertisedDevice advertisedDevice) = 0;`).
  // 포인터로 쓰면 override가 안 붙어 클래스가 추상 타입으로 남는다.
  void onResult(BLEAdvertisedDevice dev) override {
    handScanSeenCount++;
    if (!hasMasterMac) return;
    if (!nativeAddrMatchesHandMac(dev.getAddress().getNative())) return;
    if (pHandFound) delete pHandFound;
    pHandFound = new BLEAdvertisedDevice(dev);
    BLEDevice::getScan()->stop();
  }
};
static HandScanCallbacks handScanCb;

void backoffHand() {
  handNextTryMs = millis() + handRetryMs;
  if (handRetryMs < HAND_RETRY_MS_MAX) handRetryMs *=2;
  if (handRetryMs > HAND_RETRY_MS_MAX) handRetryMs = HAND_RETRY_MS_MAX;
}

// ── 의수 명령 패킷 만들기 ───────────────────────────────────────────────
// hand.py::build_cmd()의 C++ 대응. PC 프로그램과 **바이트 단위로 같아야** 한다.
//
// 테스트 벡터 — PC 코드(ble/hand.py::build_cmd)를 직접 실행해 얻은 기준값이다.
// 이 함수를 고친 뒤에는 아래와 같은 바이트열이 나오는지 확인할 것:
//
//   rest  build_cmd(0x1E,0x70,0x70,{0,0x90,0x90,0x90,0x90,0}, RELEASE=2)
//         → FF 1E 70 70 00 90 90 90 90 00 02 1C
//   close build_cmd(0x1E,0x70,0x70,{0,0x90,0x90,0x90,0x90,0}, GRASP=1)
//         → FF 1E 70 70 00 90 90 90 90 00 01 1F
//   reset build_cmd(0x3F,0x00,0x00,{0,0,0,0,0,0},        RESET=4)
//         → FF 3F 00 00 00 00 00 00 00 00 04 3B
//
// 앞 두 개는 send_cmd.py의 'F2~F5 GRASP/RELEASE' 프리셋, 세 번째는
// '전체 RESET_POWER' 프리셋과 같다 (test_protocol.py:72도 같은 벡터를 쓴다).
void buildMark7Cmd(uint8_t out[MARK7_CMD_LEN],
                   uint8_t finger_sel, uint8_t speed, uint8_t current,
                   const uint8_t pos[6], uint8_t dir) {
  out[0] = MARK7_HDR_CMD;
  out[1] = finger_sel;
  out[2] = speed;
  out[3] = current;
  for (int i = 0; i < 6; ++i) out[4 + i] = pos[i];
  out[10] = dir;

  uint8_t chk = 0;
  for (int i = 1; i <= 10; ++i) chk ^= out[i];   // 헤더 제외, byte 1~10
  out[MARK7_CMD_LEN - 1] = chk;
}

static const uint8_t MARK7_POS_GRASP[6] = {0, 0x90, 0x90, 0x90, 0x90, 0};
static const uint8_t MARK7_POS_ZERO[6]  = {0, 0, 0, 0, 0, 0};

// 전체 RESET_POWER — send_cmd.py의 '전체 RESET_POWER' 프리셋과 동일
void buildMark7Reset(uint8_t out[MARK7_CMD_LEN]) {
  buildMark7Cmd(out, 0x3F, 0x00, 0x00, MARK7_POS_ZERO, MARK7_DIR_RESET);
}

// ── 제스처 → 의수 명령 ──────────────────────────────────────────────────
// ※ 여기는 **제품 결정이 필요한 지점**이다. 암밴드는 6개 제스처를 인식하지만
//   MARK7은 손가락 6 DOF(F1~F5 + 엄지외전)만 있고 **손목 관절이 없다.** 그래서
//   손목 계열 제스처(flexion/extension/supination/pronation)는 대응할 모터가
//   아예 없다. 임의로 매핑하면 의수가 의도와 무관하게 움직이므로, 확정되기
//   전까지는 보내지 않는다(false 반환).
//
// 반환값: 보낼 명령이 있으면 true.
bool buildGestureCmd(int pred, uint8_t out[MARK7_CMD_LEN]) {
  switch (pred) {
    case 0:  // rest — 손 펴기
      buildMark7Cmd(out, 0x1E, 0x70, 0x70, MARK7_POS_GRASP, MARK7_DIR_RELEASE);
      return true;
    case 3:  // close — 손 쥐기
      buildMark7Cmd(out, 0x1E, 0x70, 0x70, MARK7_POS_GRASP, MARK7_DIR_GRASP);
      return true;
    default: // 1 flexion, 2 extension, 4 supination, 5 pronation — TODO 미정
      return false;
  }
}


// 의수로 보낸 12바이트를 시리얼에 남긴다. CHIPSEN 모듈을 PC USB에 물려놓고
// 테스트할 때, 암밴드가 보낸 바이트와 PC 시리얼 툴에 도착한 바이트를 직접
// 대조할 수 있다(양쪽이 같아야 BLE 구간이 정상).
//
// ok는 writeValue()의 반환값을 그대로 받는다 — no-response write라 상대가
// 실제로 받았는지까지는 이 값도 보장 못 하지만(전송 확인 자체가 없는 모드),
// 최소한 "로컬 BLE 스택이 이 write를 실제로 큐에 넣었는지"는 구분해준다.
// 이게 없으면 "# HAND TX: ..."가 매번 찍혀서 실패해도 성공한 것처럼 보인다.
void logHandTx(const uint8_t* cmd, const char* tag, bool ok) {
  Serial.print(ok ? "# HAND TX:" : "# HAND TX 실패:");
  for (int i = 0; i < MARK7_CMD_LEN; i++) Serial.printf(" %02X", cmd[i]);
  Serial.printf("  (%s)\n", tag);
}


// ── 의수 수신 (STATUS / ACK) ────────────────────────────────────────────
// 프레임 형식 (hand.py:1-8, parse_status()):
//   STATUS 36B : 온도[6] + 전류평균[6×2 BE] + 회전량[6×2 BE signed] + EMG[2×2 BE] + XOR
//   ACK     5B : "SETok"  (SET 수신 응답. CMD에는 ACK가 없다)
//
// ★ 조각 재조립이 필요하다. 의수 모듈은 HM-10/CC254x 계열로 MTU 23에 묶여
//   있고 협상을 하지 않으므로 notify 페이로드가 **20바이트**로 잘린다. 36B
//   STATUS는 실제로 20B + 16B **두 번**에 나눠 도착한다
//   (상향흐름_모터에서_PC까지.md ⑤). 잘리는 위치는 값 한복판일 수 있어서
//   조각을 그대로 해석하면 안 되고, 이어붙인 뒤 해석해야 한다.
//
//   같은 이유로 우리가 **보내는** CMD도 20바이트를 넘으면 쪼개진다. 12바이트라
//   한 번에 들어가지만, MTU는 양쪽의 최솟값이므로 암밴드의 setMTU(247)은
//   이 링크에서는 효력이 없다.
//
// ★ 프레임에 헤더도 길이 표시도 없어서 경계를 길이로만 잡는다(hand.py와 동일).
//   상향 경로에는 재동기 장치가 없으므로(같은 문서 표), 조각이 한동안 끊기면
//   잔여물을 버려 다음 프레임이 어긋나지 않게 한다.
#define HAND_RX_BUF_MAX  96
#define HAND_RX_IDLE_MS  100   // STATUS는 20ms 주기라 이보다 긴 공백은 단절이다

static uint8_t  handRxBuf[HAND_RX_BUF_MAX];
static size_t   handRxLen = 0;
static uint32_t handRxLastMs = 0;
static uint32_t handStatusCount = 0;

// STATUS는 20ms 루프마다 올라와 약 50Hz다. 매 프레임 찍으면 시리얼이 넘쳐
// 로그를 읽을 수 없으므로 1초에 한 번만 찍고, 그 사이 수신 건수(fps)를 함께
// 보여준다. fps가 50 근처면 링크가 정상 속도로 살아 있다는 뜻이다.
void logHandStatus(const uint8_t* s) {
  static uint32_t lastLogMs = 0;
  static uint32_t lastCount = 0;
  uint32_t now = millis();
  if (now - lastLogMs < 1000) return;

  unsigned fps = (unsigned)(handStatusCount - lastCount);
  lastCount = handStatusCount;
  lastLogMs = now;

  // mturn(회전량)은 부호 있는 16bit이고, 구동기가 실제로 움직였는지 보여준다.
  // CMD가 먹었는지 확인할 때 온도보다 이게 결정적이다 — 명령 전후로 값이
  // 변하면 전달이 확인된 것이다.
  int16_t turn[6];
  for (int i = 0; i < 6; i++) {
    turn[i] = (int16_t)((s[18 + i * 2] << 8) | s[19 + i * 2]);
  }

  Serial.printf("# HAND RX: STATUS %ufps  temp=%u,%u,%u,%u,%u,%u"
                "  turn=%d,%d,%d,%d,%d,%d  emg=%u,%u\n",
                fps, s[0], s[1], s[2], s[3], s[4], s[5],
                turn[0], turn[1], turn[2], turn[3], turn[4], turn[5],
                (unsigned)((s[30] << 8) | s[31]),
                (unsigned)((s[32] << 8) | s[33]));
}

// BLE 스택 태스크에서 호출되므로 짧게만 유지한다 — 재연결·파일IO 같은 무거운
// 처리를 여기서 하면 스택이 막힌다(pendingRestart 패턴과 같은 이유).
void onHandNotify(BLERemoteCharacteristic*, uint8_t* d, size_t len, bool) {
  uint32_t now = millis();

  if (handRxLen > 0 && now - handRxLastMs > HAND_RX_IDLE_MS) {
    handRxLen = 0;   // 단절 후 잔여물 폐기 (재동기)
  }
  handRxLastMs = now;

  if (handRxLen + len > HAND_RX_BUF_MAX) handRxLen = 0;
  memcpy(handRxBuf + handRxLen, d, len);
  handRxLen += len;

  // 떼어낼 수 있는 프레임을 모두 처리
  for (;;) {
    if (handRxLen >= 5 && memcmp(handRxBuf, "SETok", 5) == 0) {
      Serial.println("# HAND RX: ACK(SETok)");
      memmove(handRxBuf, handRxBuf + 5, handRxLen - 5);
      handRxLen -= 5;
      continue;
    }
    if (handRxLen >= MARK7_STATUS_LEN) {
      handStatusCount++;
      logHandStatus(handRxBuf);
      memmove(handRxBuf, handRxBuf + MARK7_STATUS_LEN, handRxLen - MARK7_STATUS_LEN);
      handRxLen -= MARK7_STATUS_LEN;
      continue;
    }
    break;
  }
}


// ── 의수 GATT 해석 ──────────────────────────────────────────────────────

// 연결된 의수의 service/characteristic을 전부 시리얼로 출력한다. 후보 UUID가
// 하나도 안 맞을 때 실물이 무엇을 갖고 있는지 확인할 유일한 창구다.
void dumpHandGatt() {
  Serial.println("# ── HAND GATT ──────────────────");
  std::map<std::string, BLERemoteService*>* services = pHandClient->getServices();
  if (!services || services->empty()) {
    Serial.println("#   (service 없음)");
    return;
  }
  for (auto& sp : *services) {
    BLERemoteService* svc = sp.second;
    Serial.printf("#  service %s\n", svc->getUUID().toString().c_str());
    std::map<std::string, BLERemoteCharacteristic*>* chars = svc->getCharacteristics();
    if (!chars) continue;
    for (auto& cp : *chars) {
      BLERemoteCharacteristic* ch = cp.second;
      Serial.printf("#    char %s  [%s%s%s%s]\n",
                    ch->getUUID().toString().c_str(),
                    ch->canRead()            ? "R" : "",
                    ch->canWrite()           ? "W" : "",
                    ch->canWriteNoResponse() ? "w" : "",
                    ch->canNotify()          ? "N" : "");
    }
  }
  Serial.println("# ───────────────────────────────");
}

// FFF2(송신)와 FFF1(수신)을 잡고 STATUS 구독까지 처리한다. 구독(CCCD write)은
// 클라이언트(암밴드)가 서버(의수)에게 하는 동작이고, reference/pairing.png의
// "데이터 구독 설정" 화살표가 열어주는 채널이 이것이다.
bool resolveHandChars() {
  pHandCmd    = nullptr;
  pHandStatus = nullptr;

  BLERemoteService* svc = pHandClient->getService(MARK7_SERVICE_UUID);
  if (!svc) {
    Serial.println("# HAND: service FFF0 없음 — 위 GATT 덤프 확인");
    return false;
  }

  pHandCmd = svc->getCharacteristic(MARK7_CMD_UUID);
  if (!pHandCmd) {
    Serial.println("# HAND: characteristic FFF2 없음 — 위 GATT 덤프 확인");
    return false;
  }

  pHandStatus = svc->getCharacteristic(MARK7_STATUS_UUID);
  if (pHandStatus && pHandStatus->canNotify()) {
    pHandStatus->registerForNotify(onHandNotify);  // CCCD 0x0001 write까지 함께 처리됨
    Serial.println("# HAND: STATUS(FFF1) 구독 완료");
  } else {
    pHandStatus = nullptr;
    Serial.println("# HAND: STATUS(FFF1) 없음 — 수신 없이 명령만 전송");
  }
  return true;
}

// MASTER → SLAVE 모드 전환 (폰에 붙은 경우)
void enterSlaveMode() {
  Serial.println("# MODE: MASTER -> SLAVE (폰 연결)");

  // 의수가 마지막 제스처 자세로 굳어있지 않도록 정지 명령 먼저 전송 
  if (handConnected && pHandCmd) {
    uint8_t cmd[MARK7_CMD_LEN];
    buildMark7Reset(cmd);
    bool ok = pHandCmd->writeValue(cmd, MARK7_CMD_LEN, false);
    logHandTx(cmd, "RESET (모드전환)", ok);
  }

  forceDisconnectChipsen();   // 반쪽짜리(연결 끊겼는데 connId만 남은) 링크까지 확실히 정리

  // 로봇손 관련 상태 초기화
  handConnected = false;
  pHandCmd    = nullptr;
  pHandStatus = nullptr;
  handRxLen   = 0;          // 수신 버퍼 초기화 (다음 연결 프레임 밀림 방지)
  handState = HAND_IDLE;
  handRetryMs = HAND_RETRY_MS_MIN;
  if (pHandFound) { delete pHandFound; pHandFound = nullptr; }

  // 폰 연결 중이라 광고는 꺼진 상태가 정상 
  setStatusLed(0, 0, 16);   // 파랑 = SLAVE
}

// SLAVE → MASTER 모드 전환 (폰이 끊긴 경우 MARK7 탐색 시작)
void enterMasterMode() {
  Serial.println("# MODE: SLAVE -> MASTER (폰 끊김)"); 
  
  handState = HAND_IDLE;                      // 로봇손 연결 상태 머신을 '대기' 상태로 리셋
  handRetryMs = HAND_RETRY_MS_MIN;            // 재시도 간격을 최소값으로 초기화 (다음 실패 시부터 백오프 다시 시작)
  
  handNextTryMs = millis() + MASTER_ENTRY_DELAY_MS;
  // 지금 당장 스캔 시작 x, MASTER_ENTRY_DELAY_MS(ms)만큼 지난 뒤에야 첫 탐색 시도 허용 

  setStatusLed(16, 8, 0);   // 노랑 = MARK7 탐색 대기 (아직 스캔 시작 전)
}

// HAND_READY인데 STATUS characteristic가 없을 경우 리턴 
void handStatusWatchdog() {
  if (!pHandStatus) return;   // STATUS characteristic 없는 기기면 체크 대상 아님 

  static uint32_t warnedAtMs = 0;
  uint32_t lastActivity = (handRxLastMs > handReadyMs) ? handRxLastMs : handReadyMs;
  uint32_t silentMs = millis() - lastActivity;    // 침묵 지속 시간 

  if (silentMs < HAND_STATUS_TIMEOUT_MS) {
    warnedAtMs = 0;   // 정상 수신 중 → 경고 상태 리셋
    return;
  }

  if (warnedAtMs != 0 && millis() - warnedAtMs < HAND_STATUS_TIMEOUT_MS) return;  // 같은 침묵 구간 중복 경고 방지 
  warnedAtMs = millis();

  // 무응답 경고 출력 
  Serial.printf("# HAND: STATUS(FFF1) %lums째 무응답 — 링크는 연결 상태지만 무선 구간 또는 "
                "칩센 쪽 브릿지에서 끊겼을 수 있음 (칩센 자체 시리얼에 CMD 원문이 보이는지 대조)\n",
                (unsigned long)silentMs);
}

// MASTER 모드에서만 호출 (스캔 → connect → service discovery)
void handLinkTick() {
  // READY 상태였는데 실제 연결이 끊긴 경우 → 재탐색으로 상태 되돌림 
  if (handState == HAND_READY && !handConnected) {
    Serial.println("# HAND: 링크 끊김 — 재탐색");
    pHandCmd    = nullptr;
    pHandStatus = nullptr;
    handRxLen = 0;            // 수신 버퍼 초기화 
    handState = HAND_IDLE;
    setStatusLed(16, 8, 0);
    backoffHand();            // 재시도 간격(백오프) 적용 
  }

  // 이미 연결된 상태면 워치독만 체크하고 리턴 
  if (handState == HAND_READY) {
    handStatusWatchdog();
    return;
  }
  if (!hasMasterMac) return;              // 로봇손 MAC 페어링 안 됐으면 아무것도 안함 
  if (millis() < handNextTryMs) return;   // 재시도 대기시간 안지났으면 스킵 

  switch (handState) {

    // 1. 스캔 단계
    case HAND_IDLE: {
      if (pHandFound) { delete pHandFound; pHandFound = nullptr; }
      handScanSeenCount = 0;

      BLEDevice::stopAdvertising();    // 스캔 중엔 Advertising 꺼서 라디오 경합 방지

      // 실제 BLE 스캔 실행 (SCAN_SECONDS 동안, 발견 시 handScanCb 콜백 호출)
      BLEScan* s = BLEDevice::getScan();
      s->setAdvertisedDeviceCallbacks(&handScanCb, false);
      s->setActiveScan(true);
      s->start(SCAN_SECONDS, false);
      s->stop();
      s->clearResults();

      // 스캔 도중 폰이 연결되면 → 다음 modeTick()에서 SLAVE로 전환될 것이므로 여기선 중단 
      if (deviceConnected) return;

      if (pHandFound) {
        handState = HAND_CONNECTING;    // 로봇손 발견 → 연결 시도 단계로 
      } else {

        // 못 찾은 이유 구분: 광고 자체를 못 봄 vs MAC 안 맞음 → 원인 진단용 로그
        if (handScanSeenCount == 0) {
          Serial.printf("# HAND: MARK7(%02X:%02X:%02X:%02X:%02X:%02X) 못 찾음 — 이번 스캔(%ds)에서 광고 자체를 하나도 못 봄 (칩센 전원/거리 확인)\n",
                        masterMac[0], masterMac[1], masterMac[2], masterMac[3], masterMac[4], masterMac[5],
                        SCAN_SECONDS);
        } else {
          Serial.printf("# HAND: MARK7(%02X:%02X:%02X:%02X:%02X:%02X) 못 찾음 — 광고 %lu건 봤지만 이 MAC과 일치하는 건 없음 (NVS에 등록된 MAC 확인)\n",
                        masterMac[0], masterMac[1], masterMac[2], masterMac[3], masterMac[4], masterMac[5],
                        (unsigned long)handScanSeenCount);
        }

        BLEDevice::startAdvertising();    // 재시도 대기 중엔 폰이 연결할 수 있게 광고 재개
        backoffHand();
      }
      break;
    }

    // 2. 연결 시도 단계 
    case HAND_CONNECTING: {
      if (!pHandClient) {
        pHandClient = BLEDevice::createClient();    // 암밴드가 로봇손에 대해 GATT Client 역할 생성 
        pHandClient->setClientCallbacks(&handClientCb);
      }

      Serial.println("##################################################");
      Serial.println("### CHIPSEN CONNECT DEBUG START");
      Serial.println("##################################################");

#if defined(CONFIG_NIMBLE_ENABLED)
      ble_addr_t targetAddr;
      targetAddr.type = pHandFound->getAddressType();
      memcpy(targetAddr.val, pHandFound->getAddress().getNative(), 6);

      struct ble_gap_conn_desc existingDesc;
      int existingRc = ble_gap_conn_find_by_addr(&targetAddr, &existingDesc);

      Serial.printf(
        "### PRECHECK: existing connection rc=%d (%s) ###\n",
        existingRc,
        BLEUtils::returnCodeToString(existingRc)
      );

      if (existingRc == 0) {
        Serial.printf(
          "### PRECHECK: 이미 링크 존재! handle=%u ###\n",
          existingDesc.conn_handle
        );
        dumpConnDesc(existingDesc.conn_handle, "BEFORE connect()");
      }
#endif

      uint32_t connectStartMs = millis();
      Serial.printf("### [%lu ms] pHandClient->connect() CALL ###\n", (unsigned long)connectStartMs);

      bool ok = pHandClient->connect(pHandFound);   // 실제 연결 시도 (블로킹)

      uint32_t connectEndMs = millis();
      uint16_t connId = pHandClient->getConnId();

      Serial.printf(
        "### [%lu ms] pHandClient->connect() RETURN elapsed=%lu ms ###\n",
        (unsigned long)connectEndMs, (unsigned long)(connectEndMs - connectStartMs)
      );

      Serial.printf(
        "### RESULT: ok=%d isConnected=%d handConnected=%d connId=%u ###\n",
        ok, pHandClient->isConnected(), handConnected, (unsigned)connId
      );

#if defined(CONFIG_NIMBLE_ENABLED)

      if (connId != BLE_HS_CONN_HANDLE_NONE) {
        dumpConnDesc(connId, "AFTER connect() RETURN");
      } else {
        Serial.println("### AFTER connect(): connId 자체가 없음 ###");
      }
#endif

      Serial.println("##################################################");
      Serial.println("### CHIPSEN CONNECT DEBUG END");
      Serial.println("##################################################");
      Serial.println();

      // 연결 시도 중 폰이 붙어버리면 → 로봇손 링크는 강제 종료하고 IDLE로 
      if (deviceConnected) {
        Serial.println("# PHONE 연결됨 → CHIPSEN 링크 강제 종료");
        forceDisconnectChipsen();
        handState = HAND_IDLE;
        return;
      }

      if (ok) {
        dumpHandGatt();   // 연결 성공 시 GATT 구조 로그로 남김 (디버깅용)

        if (resolveHandChars()) {      // 필요한 characteristic(FFF0/1/2 등) 찾기 성공하면 
          handState = HAND_READY;
          handRetryMs = HAND_RETRY_MS_MIN;
          handReadyMs = millis();           // STATUS 무응답 워치독 기준점 리셋
          BLEDevice::startAdvertising();    // 폰 연결 경로도 계속 열어둠 
          setStatusLed(0, 16, 0);           // 초록 = MASTER 정상 동작 중
          Serial.println("# HAND: ready - MASTER 모드 동작");
          break;
        }
        
        // characteristic을 못 찾으면 연결 무의미 → 끊기 
        Serial.println("# HAND: 후보 UUID 중 맞는 것 없음 — 위 GATT 덤프 참고");
        pHandClient->disconnect();
      } else {
        // 연결 실패 처리: 반쪽짜리 링크 남아있으면 명시적으로 정리 
        uint16_t connId = pHandClient->getConnId();

        Serial.printf(
          "# HAND: connect 실패 / isConnected=%d / connId=%u\n",
          pHandClient->isConnected(),
          (unsigned)connId
        );

        if (connId != 0xFFFF) {
          Serial.println("# CHIPSEN: 잔여 BLE 링크 정리 시도");

          int rc = pHandClient->disconnect();

          Serial.printf("# CHIPSEN: cleanup disconnect rc=%d\n", rc);

          delay(1000);  // 상대(칩센)가 disconnect 처리하고 재광고할 시간 확보 
        }

        // Client 객체는 재사용을 위해 delete 안함 
      }

      BLEDevice::startAdvertising();   // 실패했어도 광고는 반드시 재개 
      handState = HAND_IDLE;
      backoffHand();
      break;
    }

    default: break;
  }
}

#if MARK7_TEST_SWEEP
// 추론·가중치 없이 CMD 송신 경로만 검증한다. MARK7_TEST_SWEEP_MS마다
// rest(RELEASE) ↔ close(GRASP)를 번갈아 보낸다.
void handTestSweep() {
  if (handState != HAND_READY || !pHandCmd) return;

  static uint32_t lastMs = 0;
  static bool grasp = false;
  if (millis() - lastMs < MARK7_TEST_SWEEP_MS) return;
  lastMs = millis();

  uint8_t cmd[MARK7_CMD_LEN];
  buildGestureCmd(grasp ? 3 : 0, cmd);   // 3=close, 0=rest
  bool ok = pHandCmd->writeValue(cmd, MARK7_CMD_LEN, false);
  logHandTx(cmd, grasp ? "SWEEP close" : "SWEEP rest", ok);
  grasp = !grasp;
}
#endif

// loop()에서 매번 호출 (모드 판정은 deciceConnected 하나로 끝냄)
void modeTick() {
  // 폰 연결돼 있으면 SLAVE, 아니면 MASTER
  ArmbandMode want = deviceConnected ? MODE_SLAVE : MODE_MASTER;

  // 모드가 바뀌는 시점에만 진입 함수 1번 호출 
  if (want != mode) {
    if (want == MODE_SLAVE) enterSlaveMode();
    else                    enterMasterMode();
    mode = want;
  }

  // MASTER 모드(폰 미연결)면 로봇손과의 연결/통신 유지 
  if (mode == MODE_MASTER) {
    handLinkTick();
#if MARK7_TEST_SWEEP
    handTestSweep();     // 테스트 빌드에서만: 손가락 스윕 디버그 동작 
#endif
  }
}

// =========================
// SETUP (IDENTICAL TO RECORDER except final NN init)
// =========================
void setup() {
  Serial.begin(115200);
  delay(100);
  // 리셋 원인 로깅 (esp_reset_reason_t 값) —
  // 0=UNKNOWN, 1=POWERON, 2=EXT, 3=SW(esp_restart 호출, 정상 재부팅),
  // 4=PANIC, 5=INT_WDT, 6=TASK_WDT, 7=WDT, 8=DEEPSLEEP, 9=BROWNOUT, 10=SDIO
  Serial.printf("# BOOT: reset_reason=%d\n", (int)esp_reset_reason());
  pinMode(powerSwitchPin, INPUT_PULLUP);
  pinMode(trEnablePin, OUTPUT);

  // NeoPixel LED 부팅 
  strip.begin();
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, i * 2, i * 2, 0);
    }
    strip.show();
    delay(50);
  }

  digitalWrite(trEnablePin, HIGH);    // TR 활성화 (센서/모듈 전원 켬)

  Wire.begin(SDA_PIN, SCL_PIN);       // I2C 통신 시작 (BNO055 센서용)

  // BNO055(IMU 센서) 초기화 - 될 때까지 재시도, LED로 진행 표시 
  Serial.println("# SETUP: BNO055 init 시작 (bno.begin() 대기 — 여기서 안 넘어가면 센서 배선/전원 확인)");
  uint32_t bnoRetries = 0;
  while (!bno.begin()) {
    bnoRetries++;
    Serial.printf("# SETUP: bno.begin() 실패, 재시도 중 (%lu회째)\n", (unsigned long)bnoRetries);
    for (int i = 0; i < 15; ++i) {
      for (int j = 0; j < 4; ++j) {
        strip.setPixelColor(j, i * 4, 0, i * 4);   // 보라색 깜빡임 (실패/재시도 표시)
      }
      strip.show();
      delay(10);
    }
  }
  Serial.printf("# SETUP: BNO055 init 완료 (재시도 %lu회)\n", (unsigned long)bnoRetries);

  // LED 애니메니션 (청록색, 초기화 완료 표시용)
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  analogReadResolution(10);   // ADC(아날로그 입력) 해상도를 10비트로 설정 (EMG 센서 읽기용)

  // LED 애니메이션 (청록색, 점점 어두워짐)
  for (int i = 10; i >= 0; --i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  bno.setExtCrystalUse(true);     // BNO055 외부 크리스탈 사용 설정 (정밀도 향상)
  calib = 1;

  // BLE 초기화 및 서버(암밴드→폰 통신) 설정 시작 
  BLEDevice::init("ESP32S3_FAST_BLE");       // BLE 스택 초기화, 기기 이름 설정 

#if defined(CONFIG_NIMBLE_ENABLED)
  BLEDevice::setCustomGapHandler(bleGapDebug);     // GAP 이벤트 디버그 로거 등록(연결 관련 이벤트 전부 시리얼에 찍힘)
#endif

  BLEDevice::setMTU(247);    // MTU(한 번에 주고받는 데이터 크기) 247바이트로 설정 요청 

  // 암밴드를 BLE 서버(GATT Server)로 만듦 (폰이 여기 연결하러 옴)
  BLEServer* pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());    // 폰 연결/해제 시 호출될 콜백 등록 

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());

  // Second characteristic: live prediction (string format)
  pCharacteristicPred = pService->createCharacteristic(
    CHARACTERISTIC_UUID_PRED,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristicPred->addDescriptor(new BLE2902());

  // NEW: third characteristic — weight upload (Android -> armband) +
  // OK:WEIGHTS/ERR:* response (armband -> Android). docs/FIRMWARE_PROTOCOL.md 4-1.
  pCharacteristicWeights = pService->createCharacteristic(
    CHARACTERISTIC_UUID_WEIGHTS,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristicWeights->addDescriptor(new BLE2902());
  pCharacteristicWeights->setCallbacks(new WeightsCharCallbacks());

  // 페어링용 Characteristic 만들기 
  pCharacteristicPair = pService->createCharacteristic(
    CHARACTERISTIC_UUID_PAIR, 
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );
  pCharacteristicPair->addDescriptor(new BLE2902());
  pCharacteristicPair->setCallbacks(new PairCharCallbacks());

  loadMasterMac();

  if (hasMasterMac) {
    pCharacteristicPair->setValue(masterMac, 6);
  }

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMaxPreferred(0x0C);

  BLEDevice::startAdvertising();

  Serial.println("BLE Advertising Started");

  // 부팅 직후엔 폰이 없으므로 첫 modeTick()에서 MASTER로 넘어감 
  mode = MODE_SLAVE;
  setStatusLed(0, 0, 16);

  delay(100);
  for (int j = 0; j < 4; ++j) {
    strip.setPixelColor(j, 8, 8, 8);
  }
  strip.show();

  // NEW: inference state init
  for (int n = 0; n < WINDOW_SIZE; n++) {
    for (int c = 0; c < N_CHANNEL; c++) {
      infer_buffer[n][c] = 0;
    }
  }
  Serial.println("# hybrid sketch: recorder + inference 6-class");
  Serial.printf("# inference window=%d, hop=%d, vote=%d\n",
                WINDOW_SIZE, INFER_HOP, N_VOTE);

  // NEW: load NN weights + scaler from LittleFS (local-training migration,
  // see docs/FIRMWARE_PROTOCOL.md 4-1). Inference stays disabled until a
  // valid /weights.bin shows up (Android sends one over USB/BLE).
  if (!LittleFS.begin(true)) {
    Serial.println("# LittleFS mount failed — inference disabled");
  } else if (nn.loadFromLittleFS(WEIGHTS_PATH, preproc)) {
    Serial.println("# weights loaded — inference enabled");
  } else {
    Serial.println("# no weights — inference disabled");
  }
}


// =========================
// getEMG — IDENTICAL TO RECORDER + feeds inference buffer
// =========================
int getEMG() {
  static int cursor = 0;
  for (int i = 0; i < N_SENSOR; ++i) {
    static int tmp = 0;
    tmp = analogRead(sensorPin[i]);
    tmp = 0;
    tmp = analogRead(sensorPin[i]);
    tmp -= 250;
    if (tmp > 255) {
      tmp = 255;
    } else if (tmp < 0) {
      tmp = 0;
    }
    txBuffer[cursor * N_SENSOR + i] = tmp;

    // NEW: feed the inference circular buffer with the same processed value
    infer_buffer[infer_write_idx][i] = (int16_t)tmp;
  }

  // Advance circular buffer
  infer_write_idx = (infer_write_idx + 1) % WINDOW_SIZE;
  infer_total_samples++;
  infer_samples_since_last++;

  cursor++;
  if (cursor >= N_SAMPLE) cursor = 0;
  return cursor;
}


// =========================
// NEW: Inference routine
// =========================
int getMostFrequent() {
  int counts[N_CLASSES] = {0};
  int n_seen = (vote_count < N_VOTE) ? vote_count : N_VOTE;
  for (int i = 0; i < n_seen; i++) {
    int p = vote_buf[i];
    if (p >= 0 && p < N_CLASSES) counts[p]++;
  }
  int best = 0, best_count = counts[0];
  for (int i = 1; i < N_CLASSES; i++) {
    if (counts[i] > best_count) {
      best_count = counts[i];
      best = i;
    }
  }
  if (best_count < VOTE_THRESHOLD * N_VOTE) {
    return vote_buf[(vote_count - 1) % N_VOTE];
  }
  return best;
}


static int16_t snapshot[WINDOW_SIZE][N_CHANNEL];
static float   features[N_FEATURES];
static float   logits[N_CLASSES];


void runInference() {
  // Copy circular buffer into a contiguous snapshot (oldest first)
  int start = infer_write_idx;  // oldest sample lives here (about to be overwritten)
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int src = (start + i) % WINDOW_SIZE;
    for (int ch = 0; ch < N_CHANNEL; ch++) {
      snapshot[i][ch] = infer_buffer[src][ch];
    }
  }

  uint32_t t0 = micros();
  preproc.process(snapshot, features);
  uint32_t t_preproc = micros() - t0;

  t0 = micros();
  nn.predict(features, logits);
  int raw_pred = nn.argmax(logits, N_CLASSES);
  uint32_t t_nn = micros() - t0;

  // Vote cascade
  for (int i = 0; i < N_VOTE - 1; i++) vote_buf[i] = vote_buf[i + 1];
  vote_buf[N_VOTE - 1] = raw_pred;
  if (vote_count < N_VOTE) vote_count++;
  int final_pred = getMostFrequent();

  // Debug ADC stats every 5 inferences
  static int adc_dbg = 0;
  if (DEBUG_ADC_STATS && (++adc_dbg % 5) == 0) {
    Serial.print("[ADC] ");
    for (int ch = 0; ch < N_CHANNEL; ch++) {
      int mn = 999, mx = -999;
      long sum = 0;
      for (int i = 0; i < WINDOW_SIZE; i++) {
        int v = snapshot[i][ch];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
      }
      Serial.printf("ch%d:%d-%d(avg%ld) ", ch, mn, mx, sum/WINDOW_SIZE);
    }
    Serial.println();
  }

  // Per-window prediction print
  Serial.print(final_pred);
  Serial.print(" ");
  Serial.print(CLASS_NAMES[final_pred]);
  Serial.print("  raw=");
  Serial.print(raw_pred);
  Serial.print("  logits: ");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(logits[i], 3);
    if (i < N_CLASSES - 1) Serial.print(" ");
  }
  Serial.println();

  // BLE notify the prediction (format: "classname|l0|l1|l2|l3")
  if (mode == MODE_SLAVE && deviceConnected) {

    // SLAVE: 폰으로 notify
    char pred_msg[80];
    snprintf(pred_msg, sizeof(pred_msg), "%s|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f",
             CLASS_NAMES[final_pred],
             logits[0], logits[1], logits[2], logits[3], logits[4], logits[5]);
    pCharacteristicPred->setValue((uint8_t*)pred_msg, strlen(pred_msg));
    pCharacteristicPred->notify();

  } else if (mode == MODE_MASTER && handState == HAND_READY && pHandCmd) {
    // MASTER: MARK7로 제스처 인덱스 전송 (포맷: "<index>", 0~5, CLASS_NAMES 순서와 동일)
    // 예전엔 buildGestureCmd()로 12바이트 바이너리 모터 명령을 만들어 보냈지만,
    // MARK7 수신측 프로토콜이 인덱스를 직접 받는 방식으로 바뀌어서 그 변환 없이
    // 예측 인덱스를 그대로 전송한다. rest/close 외 제스처(손목 계열)에 대한
    // 매핑 여부는 이제 MARK7(수신측)이 결정한다.
    static int        lastSentPred = -1;
    static uint32_t   lastSendMs   = 0;
    if (final_pred != lastSentPred || millis() - lastSendMs > HAND_KEEPALIVE_MS) {
      char idx_msg[4];
      snprintf(idx_msg, sizeof(idx_msg), "%d", final_pred);
      // 20Hz 경로이므로 no-response. with-response는 connection interval만큼
      // 블로킹해서 781µs 샘플링 루프를 깨뜨린다.
      bool ok = pHandCmd->writeValue((uint8_t*)idx_msg, strlen(idx_msg), false);
      Serial.printf("# HAND TX%s: %s (%s)\n", ok ? "" : " 실패", idx_msg, CLASS_NAMES[final_pred]);
      lastSentPred = final_pred;
      lastSendMs = millis();
    }
  } else if (mode == MODE_MASTER) {
    // MASTER 모드이긴 한데 아직 MARK7에 연결되지 않은 상태 — 추론은 계속
    // 돌지만 보낼 곳이 없다. 20Hz마다 찍으면 스팸이라 상태가 바뀔 때만 남김.
    static bool warned = false;
    if (!warned) {
      Serial.println("# HAND: MASTER 모드지만 아직 HAND_READY 아님 — 추론 결과 전송 안 됨");
      warned = true;
    }
    if (handState == HAND_READY) warned = false;  // 다음에 다시 끊기면 재경고
  }

  // Aggregate stats
  t_preproc_sum += t_preproc;
  t_nn_sum      += t_nn;
  if (t_preproc > t_preproc_max) t_preproc_max = t_preproc;
  if (t_nn      > t_nn_max)      t_nn_max      = t_nn;
  infer_latency_count++;
  infer_count++;

  if (DEBUG_INFER_LATENCY && infer_latency_count >= N_TIMING_INF) {
    Serial.print("LATENCY us [");
    Serial.print(infer_latency_count);
    Serial.print(" inf] | mean: preproc=");
    Serial.print(t_preproc_sum / infer_latency_count);
    Serial.print(" nn=");
    Serial.print(t_nn_sum / infer_latency_count);
    Serial.print(" | max: preproc=");
    Serial.print(t_preproc_max);
    Serial.print(" nn=");
    Serial.println(t_nn_max);
    t_preproc_sum = t_nn_sum = 0;
    t_preproc_max = t_nn_max = 0;
    infer_latency_count = 0;
  }
}


// =========================
// NEW: 가중치 수신 — USB Serial
// =========================
// CRC 검증/저장은 saveWeightsIfCrcOk() 공용 함수를 그대로 씀 (BLE 쪽과
// 동일 로직) — 여기서는 Serial 스트림에서 LENGTH/PAYLOAD/CRC32를 순서대로
// 읽어 모으는 것만 담당.

// LENGTH(4B) + PAYLOAD(LENGTH바이트) + CRC32(4B)를 수신한다. 매직넘버는
// 이미 소비된 상태에서 호출됨.
void receiveWeightsUsb() {
  // 기본 1000ms는 53KB짜리 전송 중 USB CDC가 순간적으로 밀리는 구간에서
  // 너무 타이트함 (실기기 테스트에서 확인됨) — 이 함수 안에서만 넉넉하게.
  unsigned long prevTimeout = Serial.getTimeout();
  Serial.setTimeout(5000);

  uint32_t totalBytes = 0;
  if (Serial.readBytes((uint8_t*)&totalBytes, sizeof(totalBytes)) != sizeof(totalBytes)) {
    Serial.println("ERR:SIZE");
    Serial.setTimeout(prevTimeout);
    return;
  }
  if (totalBytes != WEIGHTS_TOTAL_BYTES) {
    Serial.printf("ERR:SIZE (got %u, expected %lu)\n", totalBytes, WEIGHTS_TOTAL_BYTES);
    Serial.setTimeout(prevTimeout);
    return;
  }

  uint8_t* payload = (uint8_t*)malloc(totalBytes);
  if (!payload) {
    Serial.println("ERR:MEM");
    Serial.setTimeout(prevTimeout);
    return;
  }

  uint32_t received = 0;
  const size_t CHUNK = 256;
  while (received < totalBytes) {
    size_t want = min((uint32_t)CHUNK, totalBytes - received);
    size_t got = Serial.readBytes(payload + received, want);
    if (got == 0) {
      Serial.printf("ERR:TIMEOUT (received %u / %lu bytes)\n", received, totalBytes);
      free(payload);
      Serial.setTimeout(prevTimeout);
      return;
    }
    received += got;
  }
  Serial.printf("# USB: PAYLOAD 수신 완료 (%u / %lu bytes)\n", received, totalBytes);

  uint32_t crcRecv = 0;
  if (Serial.readBytes((uint8_t*)&crcRecv, sizeof(crcRecv)) != sizeof(crcRecv)) {
    Serial.println("ERR:SIZE");
    free(payload);
    Serial.setTimeout(prevTimeout);
    return;
  }
  Serial.setTimeout(prevTimeout);

  WeightSaveResult result = saveWeightsIfCrcOk(payload, totalBytes, crcRecv);
  free(payload);

  Serial.println(weightSaveResultStr(result));
  if (result == WSAVE_OK) {
    delay(100);
    ESP.restart();  // 재부팅 후 setup()에서 새 가중치 자동 로드
  }
}

// loop()마다 호출. Serial에 4바이트 이상 쌓여있으면 읽어서 매직넘버인지
// 확인한다. 맞으면 receiveWeightsUsb()로 LENGTH/PAYLOAD 수신을 이어간다.
void checkWeightReceiveUsb() {
  if (Serial.available() < 4) return;

  uint32_t magic = 0;
  Serial.readBytes((uint8_t*)&magic, sizeof(magic));

  if (magic != WEIGHT_MAGIC) {
    Serial.println("# USB: 매직넘버 아닌 4바이트 수신 — 무시");
    return;
  }

  Serial.println("# USB: 가중치 수신 매직넘버 감지");
  receiveWeightsUsb();
}


// =========================
// LOOP — recorder loop + inference hook
// =========================
void loop() {
  // 예약된 재부팅 실행 
  if (pendingRestart && millis() >= restartAtMs) {
    ESP.restart();
  }

  // 예약된 가중치 저장 처리 (CRC 검증 후 파일 저장)
  if (pendingWeightSave) {
    pendingWeightSave = false;
    Serial.printf("# WSAVE: 시작 heap=%u\n", ESP.getFreeHeap());
    uint32_t tSaveStart = millis();

    uint32_t declaredLen;
    memcpy(&declaredLen, bleWeightBuf + 4, 4);
    const uint8_t* payload = bleWeightBuf + 8;
    uint32_t crcRecv;
    memcpy(&crcRecv, bleWeightBuf + 8 + declaredLen, 4);

    Serial.printf("# WSAVE: declaredLen=%u, CRC 계산 시작\n", declaredLen);

    WeightSaveResult result = saveWeightsIfCrcOk(payload, declaredLen, crcRecv);

    Serial.printf("# WSAVE: 완료 result=%s elapsed=%lu ms heap=%u\n",
                  weightSaveResultStr(result),
                  (unsigned long)(millis() - tSaveStart), ESP.getFreeHeap());

    resetBleWeightReceive();

    notifyWeightsResult(weightSaveResultStr(result));     // 결과 폰에 알림
    if (result == WSAVE_OK) {
      pendingRestart = true;
      restartAtMs = millis() + 300;  // NOTIFY 나갈 시간 확보 후 재부팅
    }
  }

  // 예약된 로봇 의수 MAC 저장/삭제 처리 (PairCharCallbacks::onWrite() 참고 —
  // NVS 쓰기를 BLE 콜백 밖 loop()로 미뤄서 WeightsCharCallbacks와 같은 이유로
  // BLE 스택 타이밍 문제를 피함).
  if (pendingPairSave) {
    pendingPairSave = false;
    memcpy(masterMac, pendingMasterMac, 6);

    pairPrefs.begin("pairing", false);
    pairPrefs.putBytes("master_mac", masterMac, 6);
    pairPrefs.end();

    hasMasterMac = true;

    notifyPairResult("OK:PAIR");

    Serial.printf(
      "# PAIR: 저장 완료 %02X:%02X:%02X:%02X:%02X:%02X\n",
      masterMac[0], masterMac[1], masterMac[2],
      masterMac[3], masterMac[4], masterMac[5]
    );
  }

  if (pendingPairClear) {
    pendingPairClear = false;

    pairPrefs.begin("pairing", false);
    pairPrefs.remove("master_mac");
    pairPrefs.end();

    memset(masterMac, 0, sizeof(masterMac));
    hasMasterMac = false;

    notifyPairResult("OK:CLEAR");
    Serial.println("# PAIR: 저장된 마스터 MAC 삭제됨");
  }

  static uint32_t tick = 0;
  tick = micros();      // 주기 계산용 시작 시간 

  checkWeightReceiveUsb();       // USB로도 가중치 수신 체크

  // 가중치 수신 중단된 채 방치되면 타임아웃 처리 
  if (!pendingWeightSave && bleWeightBufLen > 0 &&
      millis() - bleWeightLastChunkMs > BLE_WEIGHT_IDLE_TIMEOUT_MS) {
    Serial.println("# BLE: 가중치 수신 타임아웃 — 버퍼 폐기");
    resetBleWeightReceive();
    notifyWeightsResult("ERR:SIZE");
  }

  shutdownOnSwitch();     // 전원 스위치 확인 

  modeTick();   // 폰 연결 여부로 SLAVE/MASTER 판정, MASTER면 로봇손 링크 진행 

  int pos = getEMG();      // EMG 샘플링 

  // ─── 20샘플마다 EMG 데이터 BLE notify ─────────────────────────
  if (pos == 0) {
    if (deviceConnected) {
      pCharacteristic->setValue(txBuffer, PACKET_SIZE);
      pCharacteristic->notify();
    }
  }

  // ─── 추론 윈도우 다 차면 신경망 추론 실행 ─────────────────────
  if (nn.isLoaded() &&
      infer_total_samples >= WINDOW_SIZE &&
      infer_samples_since_last >= (uint32_t)INFER_HOP) {
    infer_samples_since_last = 0;
    runInference();
  }

  // 일정 주기 유지를 위한 딜레이 
  int delayMicro = periodMicro - micros() + tick;
  if (delayMicro > 0 && delayMicro < periodMicro) {
    delayMicroseconds(delayMicro);
  }
}


void shutdownOnSwitch() {
  int shutdown_count = 33;
  if (digitalRead(powerSwitchPin) == LOW) {
    while (digitalRead(powerSwitchPin) == LOW) {
      for (int j = 0; j < 4; ++j) {
        strip.setPixelColor(j, shutdown_count, shutdown_count, shutdown_count);
      }
      strip.show();
      delay(50);
      shutdown_count -= 2;
      if (shutdown_count < 0) {
        shutdown_count = 0;
        digitalWrite(trEnablePin, LOW);
        for (int j = 0; j < 4; ++j) {
          digitalWrite(trEnablePin, LOW);
        }
        digitalWrite(trEnablePin, LOW);
        strip.show();
        digitalWrite(trEnablePin, LOW);
        while (1) {
          delay(100);
          digitalWrite(trEnablePin, LOW);
        }
      }
    }
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 8, 8, 8);
    }
    strip.show();
  }
}


void getQuatFromBNO055() {
  static uint8_t system = 0, gyr = 0, acc = 0, mag = 0;
  static int calibration_loop = 0;
  calibration_loop++;
  if (calibration_loop >= 100) {
    calibration_loop = 0;
    bno.getCalibration(&system, &gyr, &acc, &mag);
    calib = system + mag;
  }
  static uint8_t system_okay = 0;
  if (system > 0) system_okay = 1;

  imu::Quaternion quat = bno.getQuat();
  static float qf[4];
  qf[0] = quat.w();
  qf[1] = quat.x();
  qf[2] = quat.y();
  qf[3] = quat.z();
  for (int i = 0; i < 4; ++i) {
    q_int[i] = (int)(qf[i] * 10000);
  }
}
