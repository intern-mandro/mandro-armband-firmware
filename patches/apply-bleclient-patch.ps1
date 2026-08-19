# NimBLE BLEClient.cpp의 BLE_HS_EALREADY 패치를 arduino-cli가 설치한 ESP32 코어에
# 적용한다. 이 패치는 시스템에 전역 설치된 코어 파일을 고치는 것이라 git으로
# 추적되지 않으므로, esp32:esp32 코어를 새로 설치할 때마다(README.md 1단계 이후)
# 다시 실행해야 한다. 이미 적용돼 있으면 아무것도 하지 않고 종료한다.
#
# 사용법 (저장소 어디서든):
#   powershell -File patches\apply-bleclient-patch.ps1

$ErrorActionPreference = "Stop"

$dataDir = "C:\arduino-cli-data"
try {
    $configDump = & arduino-cli config dump 2>$null
    $match = $configDump | Select-String -Pattern '^\s*data:\s*(.+)$'
    if ($match) {
        $dataDir = $match.Matches[0].Groups[1].Value.Trim()
    }
} catch {
    Write-Host "arduino-cli config dump 실패 — 기본 경로($dataDir)로 시도합니다."
}

$coreDir = Join-Path $dataDir "packages\esp32\hardware\esp32\3.3.11\libraries\BLE\src"
$target = Join-Path $coreDir "BLEClient.cpp"

if (-not (Test-Path $target)) {
    Write-Error "BLEClient.cpp를 찾을 수 없습니다: $target`nesp32:esp32 코어가 설치됐는지 확인하세요 (README.md 1단계)."
    exit 1
}

# BLE_HS_EALREADY 자체는 스톡 NimBLE 코드에도 다른 곳(예: 978번째 줄 근처 switch-case)에
# 이미 존재해서 이미 적용 여부 판단 기준으로 못 쓴다 (오탐 발생). 이 패치가 새로 추가한
# 로그 메시지처럼 스톡 코드에 절대 없는 고유 문자열로 판단해야 한다.
if (Select-String -Path $target -Pattern 'MTU already exchanged' -Quiet) {
    Write-Host "이미 패치 적용됨: $target"
    exit 0
}

$backup = "$target.orig-backup"
if (-not (Test-Path $backup)) {
    Copy-Item $target $backup
    Write-Host "원본 백업: $backup"
}

# diff를 git apply로 재구성하지 않고, 이미 패치가 적용된 완성본
# (BLEClient.cpp)을 그대로 덮어쓴다 — 대상 PC의 스톡 파일이 줄바꿈 방식이나
# 코어 마이너 버전 차이로 조금만 달라도 git apply는 컨텍스트 불일치로 실패할
# 수 있지만, 파일 통째로 교체하는 건 그런 문제가 없다.
$patchedFile = Join-Path $PSScriptRoot "BLEClient.cpp"
Copy-Item $patchedFile $target -Force

Write-Host "패치 적용 완료: $target"
