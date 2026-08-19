# patches/

esp32 코어(arduino-cli가 전역 설치하는 `BLEClient.cpp`)에 필요한 로컬 패치와,
만약을 대비한 원본 사본을 담아둔 폴더. 코어 자체는 이 저장소 밖(`<arduino-cli
data dir>/packages/...`)에 설치되고 git으로 추적되지 않으므로, 여기 있는 파일들이
그 변경 내용을 재현/복구할 수 있는 유일한 기록이다.

## 파일 구성

- **`BLEClient.cpp`** — 패치가 이미 적용된 완성본. `apply-bleclient-patch.ps1`이
  이 파일을 그대로 대상 PC에 복사한다 (diff를 `git apply`로 재구성하지 않음 —
  대상 PC의 스톡 파일이 줄바꿈 방식이나 코어 마이너 버전 차이로 조금만 달라도
  `git apply`는 컨텍스트 불일치로 실패할 수 있지만, 파일 통째로 교체하면 그럴
  일이 없다).
- **`BLEClient.cpp.orig`** — esp32 코어 3.3.11의 `BLEClient.cpp` 원본(패치 전) 그대로의
  사본. 참고용/비상용 — 되돌리고 싶을 때, 또는 원본과 정확히 뭐가 달라졌는지
  다시 확인하고 싶을 때 기준점으로 쓴다.
- **`BLEClient.ealready-mtu.patch`** — 원본 → 패치본 변경 내용(unified diff). 실제
  적용에는 안 쓰이고, 코드 리뷰하듯 "정확히 어느 줄이 왜 바뀌었는지"를 짧게
  보고 싶을 때 참고용으로 남겨둔다. 상단 주석에 왜 필요한지도 설명돼 있다.
- **`apply-bleclient-patch.ps1`** — 코어 설치 경로를 자동으로 찾아 `BLEClient.cpp`를
  덮어쓰는 스크립트. README.md 1-1단계에서 사용. 이미 적용돼 있으면 건너뛴다.

## 원본으로 되돌리고 싶을 때

패치 적용 스크립트가 처음 실행될 때 그 PC의 실제 코어 파일 옆에
`BLEClient.cpp.orig-backup`을 자동으로 남겨두므로, 보통은 그걸로 되돌리면 된다:

```powershell
$core = "C:\arduino-cli-data\packages\esp32\hardware\esp32\3.3.11\libraries\BLE\src"
Copy-Item "$core\BLEClient.cpp.orig-backup" "$core\BLEClient.cpp" -Force
```

그 백업마저 없는 상태(다른 PC, 백업 파일 삭제 등)라면 이 폴더의
`BLEClient.cpp.orig`를 대신 쓰면 된다:

```powershell
$core = "C:\arduino-cli-data\packages\esp32\hardware\esp32\3.3.11\libraries\BLE\src"
Copy-Item "patches\BLEClient.cpp.orig" "$core\BLEClient.cpp" -Force
```
