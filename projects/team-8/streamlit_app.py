"""반려동물 보험 추천 시스템 - Streamlit 데모 UI"""

import uuid

import streamlit as st

from app.agents.orchestrator.orchestrator_graph import graph
from app.agents.user_input_template_agent.utils.cli import make_config
from app.agents.user_input_template_agent.state.user_input_template_state import (
    Insurer,
)

# ── 설정 ──────────────────────────────────────────────
INSURER_OPTIONS = [e.value for e in Insurer]

st.set_page_config(page_title="반려동물 보험 추천", layout="wide")
st.title("반려동물 보험 추천 시스템")

# ── 세션 초기화 ───────────────────────────────────────
if "pet_locked" not in st.session_state:
    st.session_state.pet_locked = False
    st.session_state.pet_info = {}
    st.session_state.last_result = None
    st.session_state.thread_id = str(uuid.uuid4())

locked = st.session_state.pet_locked

# ── 사이드바: 입력 폼 ────────────────────────────────
with st.sidebar:
    st.header("반려동물 정보")
    if locked:
        st.caption("반려동물 정보가 고정되었습니다.")

    species = st.selectbox("종", ["강아지", "고양이"], disabled=locked)
    breed = st.text_input("품종", placeholder="예: 말티즈, 페르시안", disabled=locked)
    age = st.number_input("나이 (세)", min_value=0, max_value=30, value=3, disabled=locked)
    gender = st.radio("성별", ["수컷", "암컷"], horizontal=True, disabled=locked)
    is_neutered = st.checkbox("중성화 여부", disabled=locked)
    weight = st.number_input("체중 (kg)", min_value=1, max_value=100, value=5, disabled=locked)

    st.markdown("---")
    st.subheader("건강 상태 (선택)")
    illness_area = st.text_input("자주 아픈 부위", placeholder="예: 피부, 관절", disabled=locked)
    disease_history = st.text_input("질병/수술 이력", placeholder="예: 슬개골 탈구", disabled=locked)

    st.markdown("---")
    st.subheader("보장 선호")
    coverage = st.radio(
        "보장 스타일", ["선택 안함", "최소 보장", "종합 보장"], horizontal=True
    )
    selected_insurers = st.multiselect("선호 보험사", INSURER_OPTIONS)

    st.markdown("---")
    col_run, col_reset = st.columns(2)
    with col_run:
        btn_label = "다시 추천" if locked else "추천 실행"
        run_btn = st.button(btn_label, type="primary", use_container_width=True)
    with col_reset:
        reset_btn = st.button("새 세션", use_container_width=True)

# ── 새 세션 리셋 ──────────────────────────────────────
if reset_btn:
    st.session_state.pet_locked = False
    st.session_state.pet_info = {}
    st.session_state.last_result = None
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# ── state dict 구성 ───────────────────────────────────
gender_map = {"수컷": "male", "암컷": "female"}
coverage_map = {"최소 보장": "minimal", "종합 보장": "comprehensive"}

if locked:
    # 잠긴 경우 저장된 반려동물 정보 사용
    state_dict: dict = dict(st.session_state.pet_info)
else:
    state_dict = {
        "species": species,
        "breed": breed or None,
        "age": age,
        "gender": gender_map[gender],
        "is_neutered": is_neutered,
        "weight": weight,
    }

if coverage in coverage_map:
    state_dict["coverage_style"] = coverage_map[coverage]
if selected_insurers:
    state_dict["preferred_insurers"] = selected_insurers

if locked:
    # 잠긴 경우 저장된 건강 상태 사용
    if "health_condition" in st.session_state.pet_info:
        state_dict["health_condition"] = st.session_state.pet_info["health_condition"]
else:
    health = {}
    if illness_area.strip():
        health["frequent_illness_area"] = illness_area.strip()
    if disease_history.strip():
        health["disease_surgery_history"] = disease_history.strip()
    if health:
        state_dict["health_condition"] = health

# ── 메인: 입력 요약 + 결과 ────────────────────────────
pet = st.session_state.pet_info if locked else state_dict
col_info, col_result = st.columns([1, 2])

with col_info:
    st.subheader("입력 요약")
    gender_kr = {"male": "수컷", "female": "암컷"}
    st.markdown(f"**종**: {pet.get('species', '-')}")
    st.markdown(f"**품종**: {pet.get('breed') or '-'}")
    st.markdown(f"**나이**: {pet.get('age', '-')}세")
    st.markdown(f"**성별**: {gender_kr.get(pet.get('gender', ''), '-')}")
    st.markdown(f"**체중**: {pet.get('weight', '-')}kg")
    st.markdown(f"**중성화**: {'예' if pet.get('is_neutered') else '아니오'}")
    if coverage != "선택 안함":
        st.markdown(f"**보장 스타일**: {coverage}")
    if selected_insurers:
        st.markdown(f"**선호 보험사**: {', '.join(selected_insurers)}")
    health_info = pet.get("health_condition", {}) if locked else {}
    area = health_info.get("frequent_illness_area", "") if locked else illness_area.strip()
    history = health_info.get("disease_surgery_history", "") if locked else disease_history.strip()
    if area:
        st.markdown(f"**자주 아픈 부위**: {area}")
    if history:
        st.markdown(f"**질병/수술 이력**: {history}")

with col_result:
    st.subheader("추천 결과")

    # 추천 실행 → 결과를 세션에 저장 후 rerun
    if run_btn:
        if not (pet.get("breed") if locked else breed):
            st.warning("품종을 입력해주세요.")
            st.stop()

        NODE_LABELS = {
            "user_input_template": "사용자 입력 처리",
            "vet_diagnosis": "수의사 진단 분석",
            "RAG": "보험 약관 검색",
            "save_recommendation": "추천 이력 저장",
            "judge": "보험 상품 검증",
            "composer": "추천 메시지 작성",
        }

        config = make_config(thread_id=st.session_state.thread_id)
        with st.status("파이프라인 실행 중...", expanded=True) as status:
            for chunk in graph.stream(state_dict, config=config, stream_mode="updates"):
                for node_name in chunk:
                    label = NODE_LABELS.get(node_name, node_name)
                    status.update(label=f"{label} 진행 중...")
                    st.write(f"[완료] {label}")
            status.update(label="실행 완료", state="complete")

        result = graph.get_state(config).values

        # 결과 저장
        st.session_state.last_result = result

        # 첫 실행 시 반려동물 정보 + 건강 상태 잠금
        if not locked:
            pet_info = {
                "species": species,
                "breed": breed,
                "age": age,
                "gender": gender_map[gender],
                "is_neutered": is_neutered,
                "weight": weight,
            }
            if illness_area.strip():
                pet_info.setdefault("health_condition", {})["frequent_illness_area"] = illness_area.strip()
            if disease_history.strip():
                pet_info.setdefault("health_condition", {})["disease_surgery_history"] = disease_history.strip()
            st.session_state.pet_locked = True
            st.session_state.pet_info = pet_info

        # 즉시 rerun하여 잠금 상태 반영
        st.rerun()

    # 세션에 저장된 결과 표시
    result = st.session_state.last_result
    if result:
        # 질병 진단
        diseases = result.get("diseases", [])
        if diseases:
            st.markdown("#### 진단 질병")
            for d in diseases:
                st.markdown(
                    f"- **{d.name}** (발병률: {d.incidence_rate}, 시기: {d.onset_period})"
                )

        # 검증 결과
        validation = result.get("validation_result")
        if validation:
            st.markdown("#### 보험 상품 검증")
            policies = validation.get("selected_policies", [])
            for p in policies:
                score = p.get("suitability_score", p.get("score", "-"))
                st.markdown(
                    f"- **{p.get('product_name', '-')}** (적합도: {score}점) — {p.get('reason', '')}"
                )
            summary = validation.get("review_summary")
            if summary:
                st.info(summary)

        # 최종 메시지
        final = result.get("final_message", "")
        if final:
            st.markdown("#### 최종 추천 메시지")
            st.success(final)
    else:
        st.info("왼쪽에서 정보를 입력하고 '추천 실행' 버튼을 눌러주세요.")
