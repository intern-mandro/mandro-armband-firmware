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

#include "preprocessor.h"
#include "nn.h"

#define WEIGHTS_PATH     "/weights.bin"
#define WEIGHTS_TMP_PATH "/weights.tmp"

#define PAIR_HDR         0xE0
#define PAIR_PACKET_LEN  8       // 헤더 1B + MAC 6B + 체크섬 1B = 8B

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
static const char* CHARACTERISTIC_UUID_WEIGHTS = "abcd1234-5678-1234-5678-abcdef123458"; // 가중치 수신 (신규)
static const char* CHARACTERISTIC_UUID_PAIR = "abcd1234-5678-1234-5678-abcdef123459";   // 페어링용 


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
  uint32_t crcCalc = crc32_calc(payload, totalBytes);
  if (crcCalc != crcRecv) {
    Serial.printf("# 가중치 CRC 불일치 (calc=%08X recv=%08X)\n", crcCalc, crcRecv);
    return WSAVE_ERR_CRC;
  }

  File f = LittleFS.open(WEIGHTS_TMP_PATH, "w");
  if (!f) return WSAVE_ERR_WRITE;
  size_t written = f.write(payload, totalBytes);
  f.close();
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
// BLE
// =========================

BLECharacteristic* pCharacteristic;         // raw samples
BLECharacteristic* pCharacteristicPred;     // predictions
BLECharacteristic* pCharacteristicWeights;  // 가중치 수신 (신규)
BLECharacteristic* pCharacteristicPair;     // 페어링용 MAC 수신 

bool deviceConnected = false;

// ── Pairing: 로봇 의수 MAC 저장 ──────────────────
Preferences pairPrefs;
uint8_t masterMac[6] = {0};
bool hasMasterMac = false;

// ESP32에 저장된 로봇 의수 MAC을 불러오고, 저장 여부를 확인하는 함수 
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
      "# PAIR: 저장된 마스터 %02X:%02X:%02X:%02X:%02X:%02X\n", 
      masterMac[0], 
      masterMac[1], 
      masterMac[2], 
      masterMac[3], 
      masterMac[4], 
      masterMac[5]
    );
  } else {
    Serial.println("# PAIR: 저장된 마스터 없음");
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

// characteristic 3의 NOTIFY로 결과 문자열 전송
void notifyWeightsResult(const char* msg) {
  if (!deviceConnected) return;
  pCharacteristicWeights->setValue((uint8_t*)msg, strlen(msg));
  pCharacteristicWeights->notify();
}

// BLE 가중치 수신 상태 초기화 (연결 끊기거나 오류 시 호출)
void resetBleWeightReceive() {
  bleWeightBufLen = 0;
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
    String chunk = c->getValue();
    const uint8_t* data = (const uint8_t*)chunk.c_str();
    size_t len = chunk.length();

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

    // PAYLOAD + CRC32까지 다 도착 — 저장 시도
    const uint8_t* payload = bleWeightBuf + 8;
    uint32_t crcRecv;
    memcpy(&crcRecv, bleWeightBuf + 8 + declaredLen, 4);

    WeightSaveResult result = saveWeightsIfCrcOk(payload, declaredLen, crcRecv);
    resetBleWeightReceive();

    notifyWeightsResult(weightSaveResultStr(result));
    if (result == WSAVE_OK) {
      // 여기서 바로 재부팅하지 않음 — 이 콜백이 return해야 이번 write의 ATT
      // 확인 응답이 나가고, NOTIFY도 정상적으로 전파됨. 재부팅은 loop()에
      // 맡기고 이 콜백은 즉시 끝냄.
      pendingRestart = true;
      restartAtMs = millis() + 300;  // NOTIFY가 실제로 나갈 시간만 살짝 확보
    }
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

    // 1. 길이 + 헤더 검사
    if (len != PAIR_PACKET_LEN || d[0] != PAIR_HDR) {

      Serial.printf(
        "# PAIR: 형식 불일치 (len=%u, hdr=%02X)\n", 
        (unsigned)len, 
        len ? d[0] : 0
      );

      notifyPairResult("ERR:PAIR_FORMAT");

      return;
    }

    // 2. XOR checksum 검사
    uint8_t chk = 0;

    for (int i = 1; i < PAIR_PACKET_LEN - 1; ++i) {
      chk ^= d[i];
    }

    if (chk != d[PAIR_PACKET_LEN - 1]) {

      Serial.println("# PAIR: 체크섬 불일치");

      notifyPairResult("ERR:PAIR_CRC");
      
      return;
    }

    // 3. MAC 6byte 복사
    memcpy(masterMac, d + 1, 6);

    // 4. NVS에 저장
    pairPrefs.begin("pairing", false);  // pairing 저장공간 열기 

    pairPrefs.putBytes(
      "master_mac", 
      masterMac, 
      6
    );

    pairPrefs.end();

    hasMasterMac = true;   // 현재 유효한 마스터주소 있다고 상태 변경 

    // PC에 성공 응답
    notifyPairResult("OK:PAIR");

    Serial.printf(
      "# PAIR: 저장 완료 %02X:%02X:%02X:%02X:%02X:%02X\n", 
      masterMac[0],
      masterMac[1],
      masterMac[2],
      masterMac[3],
      masterMac[4],
      masterMac[5] 
    );
  }
};

uint8_t txBuffer[PACKET_SIZE];
volatile uint32_t sampleCounter = 0;

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
#define DEBUG_ADC_STATS      0
#define DEBUG_INFER_LATENCY  0


// =========================
// SETUP (IDENTICAL TO RECORDER except final NN init)
// =========================
void setup() {
  Serial.begin(115200);
  pinMode(powerSwitchPin, INPUT_PULLUP);
  pinMode(trEnablePin, OUTPUT);

  strip.begin();
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, i * 2, i * 2, 0);
    }
    strip.show();
    delay(50);
  }

  digitalWrite(trEnablePin, HIGH);

  Wire.begin(SDA_PIN, SCL_PIN);

  while (!bno.begin()) {
    for (int i = 0; i < 15; ++i) {
      for (int j = 0; j < 4; ++j) {
        strip.setPixelColor(j, i * 4, 0, i * 4);
      }
      strip.show();
      delay(10);
    }
  }

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  analogReadResolution(10);

  for (int i = 10; i >= 0; --i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  bno.setExtCrystalUse(true);
  calib = 1;

  BLEDevice::init("ESP32S3_FAST_BLE");
  BLEDevice::setMTU(247);

  BLEServer* pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

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
// vote_buf는 시프트 레지스터다 — runInference()가 값을 왼쪽으로 밀고 최신
// 예측을 항상 vote_buf[N_VOTE-1]에 넣는다. 따라서 아직 다 안 채워진 워밍업
// 구간(vote_count < N_VOTE)에 세야 할 것은 앞쪽이 아니라 **뒤쪽** n_seen개다.
// 앞쪽을 세면 한 번도 쓰이지 않은 0(= class 0 "rest")이 표에 섞여 부팅 직후
// 두 번의 추론이 rest 쪽으로 끌려간다.
int getMostFrequent() {
  int counts[N_CLASSES] = {0};
  int n_seen = (vote_count < N_VOTE) ? vote_count : N_VOTE;
  for (int i = N_VOTE - n_seen; i < N_VOTE; i++) {
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
  // 과반이 없으면 최신 예측으로 폴백. 최신은 항상 마지막 슬롯이다 — 이전엔
  // vote_buf[(vote_count-1) % N_VOTE]였는데 vote_count가 N_VOTE로 포화된
  // 뒤에만 우연히 맞고, 워밍업 2회는 엉뚱한 슬롯을 읽었다.
  if (best_count < VOTE_THRESHOLD * N_VOTE) {
    return vote_buf[N_VOTE - 1];
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
  if (deviceConnected) {
    char pred_msg[80];
    snprintf(pred_msg, sizeof(pred_msg), "%s|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f",
             CLASS_NAMES[final_pred],
             logits[0], logits[1], logits[2], logits[3], logits[4], logits[5]);
    pCharacteristicPred->setValue((uint8_t*)pred_msg, strlen(pred_msg));
    pCharacteristicPred->notify();
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
  // BLE 가중치 수신 콜백(onWrite)에서 예약해둔 재부팅 처리 — 콜백 밖에서
  // 처리해야 그 write의 ATT 확인 응답/NOTIFY가 막히지 않음 (위 설명 참고).
  if (pendingRestart && millis() >= restartAtMs) {
    ESP.restart();
  }

  static uint32_t tick = 0;
  tick = micros();

  checkWeightReceiveUsb();

  shutdownOnSwitch();

  int pos = getEMG();

  // ─── BLE notify every 20 samples (UNCHANGED) ─────────────────────────
  if (pos == 0) {
    if (deviceConnected) {
      pCharacteristic->setValue(txBuffer, PACKET_SIZE);
      pCharacteristic->notify();
    }
  }

  // ─── NEW: inference every INFER_HOP samples, once window is full and
  //          weights have been loaded from LittleFS ─────────────────────
  if (nn.isLoaded() &&
      infer_total_samples >= WINDOW_SIZE &&
      infer_samples_since_last >= (uint32_t)INFER_HOP) {
    infer_samples_since_last = 0;
    runInference();
  }

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
