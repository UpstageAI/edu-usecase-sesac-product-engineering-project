# SeSAC Product Engineering Project

이 저장소는 SeSAC에서 진행하는 Product Engineering Proejct의 결과물을 취합하고 공유하는 **레포지토리**입니다.

* 프로젝트 기간: 02. 04 ~ 02. 20

---

## 프로젝트 구조 (Directory Structure)

모든 조의 프로젝트는 `projects/` 디렉토리 하위에서 관리됩니다. 각 조는 자신들의 폴더 안에서만 작업합니다.

```text
.
├── projects/
│   ├── team-1/           # 1조 작업 공간 (uv 기반)
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   └── src/
│   ├── team-2/           # 2조 작업 공간 (uv 기반) 
│   │   └── ...
│   └── (team-n)/         # 각 조별 독립 디렉토리
├── .gitignore          
└── README.md             # 현재 가이드 문서
```

## 레포지토리 활용법 (Step-by-Step)
이 레포지토리를 Fork하고 Pull Request를 올리는 방식을 활용해, 프로젝트를 업로드 합니다!

1. 원본 저장소 Fork
상단의 [Fork] 버튼을 눌러 본인의 GitHub 계정으로 저장소를 복사합니다.

2. 로컬로 Clone & Upstream 설정
```Bash
# 본인 계정의 저장소를 가져옵니다 (your-id 부분을 수정하세요).
git clone https://github.com/{your-id}/edu-usecase-sesac-product-engineering-project.git
cd edu-usecase-sesac-product-engineering-project

# 원본 저장소를 'upstream'이라는 이름으로 연결합니다. (최신 상태 유지를 위해 필수)
git remote add upstream https://github.com/UpstageAI/edu-usecase-sesac-product-engineering-project.git
```

3. 브랜치 생성
```Bash
# 본인 조에 맞는 브랜치를 생성하고 이동합니다. (예: team-1-dev)
git checkout -b team-{n}-dev
```

4. 프로젝트 초기화 (uv init)
```Bash
# 1. 조별 폴더 생성 및 이동
mkdir -p projects/team-{n}
cd projects/team-{n}

# 2. uv 프로젝트 초기화 (pyproject.toml, .python-version 등이 자동 생성됨)
uv init
```
5. 변경사항 제출 (Push & Pull Request)
```
git add .
git commit -m "chore(team-{n}): 초기 환경 설정 및 uv 프로젝트 생성"
git push origin team-{n}-dev
```

## 작업 규칙

* 팀의 폴더에서만 작업해주세요!: 반드시 `projects/team-{n}/` 폴더 내부만 수정합니다.
* 루트 파일 수정 금지: `.gitignore`나 `README.md` 등 루트 폴더의 내용을 수정하지 말아주세요!

## 제출 프로세스 (2-Step)

### Step 1. 프로젝트 등록
최종 제출 이전에 미리 Folder 생성을 위해 해당 작업을 진행합니다. 위 레포지토리 활용법 (Step-by-Step)를 참고합니다.
1. **Fork:** 본 저장소를 포크합니다.
2. **Clone & Setup:** 로컬에 클론 후 `projects/team-{n}` 폴더를 만듭니다.
3. **Init:** 해당 폴더에서 `uv init`을 실행하여 기본 파일을 생성합니다.
4. **PR:** `chore: {n}조 프로젝트 등록`이라는 제목으로 PR을 보냅니다. 
   - *강사님이 확인 후 바로 승인(Merge)할 예정입니다.*

### Step 2. 최종 제출 (프로젝트 종료 시)
1. **Sync:** 작업 전 `git pull upstream main`으로 다른 조의 등록 현황을 반영합니다.
2. **Work:** 본인의 폴더(`projects/team-{n}/`) 내에서 자유롭게 코딩합니다.
3. **Final PR:** 모든 개발이 완료되면 원본 저장소로 최종 PR을 날립니다.

---
