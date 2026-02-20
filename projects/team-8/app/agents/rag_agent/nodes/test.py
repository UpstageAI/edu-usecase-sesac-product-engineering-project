"""
Build bag-of-words vocab files for each insurance policy PDF.

Creates vocab.jsond under data/terms/<pdf_stem>/ using Kiwipiepy tokens
and predefined medical terms. Skips documents that already have a vocab file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from kiwipiepy import Kiwi
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

PREDEFINED_WORDS = [
	"슬관절탈구",
	"고관절탈구",
	"슬관절형성부전",
	"고관절형성부전",
	"대퇴 골두 허혈성 괴사",
	"감염병",
	"전염병",
	"세균감염",
	"바이러스감염",
	"기생충감염",
	"피부질환",
	"피부염",
	"알레르기",
	"알러지반응",
	"위장질환",
	"위염",
	"장염",
	"장질환",
	"간질환",
	"신장질환",
	"방광질환",
	"요로질환",
	"심장질환",
	"심부전",
	"판막질환",
	"호흡기질환",
	"폐질환",
	"기관지염",
	"폐렴",
	"신경계질환",
	"뇌질환",
	"경련",
	"발작",
	"종양",
	"양성종양",
	"악성종양",
	"암",
	"종괴",
	"종창",
	"골절",
	"탈구",
	"염좌",
	"인대손상",
	"근육손상",
	"디스크질환",
	"추간판탈출",
	"관절질환",
	"슬개골탈구",
	"고관절질환",
	"백내장",
	"녹내장",
	"안과질환",
	"치과질환",
	"구강질환",
	"치주질환",
	"치아손상",
	"췌장염",
	"당뇨병",
	"갑상선질환",
	"내분비질환",
	"면역질환",
	"자가면역질환",
	"빈혈",
	"혈액질환",
	"중독",
	"이물섭취",
	"교통사고상해",
	"낙상사고",
	"외상",
	"화상",
	"동상",
	"열사병",
	"탈수",
	"구토",
	"설사",
	"뒷다리 골육종",
	"기타 근골격 계통 양성 신생물",
	"기타 근골격 계통 악성 신생물",
	"기타 근골격 계통 신생물",
	"고관절 이형성",
	"고관절 탈구",
	"무혈성골두괴사",
	"슬개골 탈구",
	"십자 인대 손상 파열",
	"골절",
	"성장판 골절",
	"관절염",
	"퇴행성 관절염",
	"뼈연골",
	"근염",
	"염좌",
	"근골격계 질환",
	"눈 및 부속 기관 양성 신생물",
	"눈 및 부속 기관 악성 신생물",
	"눈 및 부속 기관 신생물",
	"안검 외반",
	"안검 내반",
	"안검염",
	"다래끼",
	"산립종",
	"마이봄선종",
	"체리아이",
	"제3안검 돌출",
	"비루관폐쇄",
	"유루증",
	"첩모난생",
	"첩모중생",
	"이소성첩모",
	"궤양성 각막염",
	"각막궤양",
	"각막 미란",
	"각막 이영양증",
	"각막염",
	"건성 각결막염",
	"결막염",
	"결막 부종",
	"포도막염",
	"홍채염",
	"전안방 출혈",
	"백내장",
	"수정체 탈구",
	"망막 변성",
	"망막 위축",
	"진행성 망막 위축",
	"망막 박리",
	"유리체 변성",
	"녹내장",
	"동양안충증",
	"초자체변성",
	"상공막염",
	"고양이 호산구성 각결막염",
	"눈곱",
	"결막 충혈",
	"눈 가려움",
	"순환기 계통 양성 신생물",
	"순환기 계통 악성 신생물",
	"순환기 계통 신생물",
	"고혈압",
	"저혈압",
	"부정맥",
	"판막 질환",
	"심부전",
	"심비대",
	"확장성",
	"심근병",
	"비대성",
	"제한성",
	"일시적 심근비대증",
	"심근증",
	"대동맥 협착증",
	"폐동맥 협착",
	"선천성 심장 질환",
	"심장사상충",
	"심혈관계 질환",
	"점액성 이첨판막변성",
	"신장 양성 신생물",
	"신장 악성 신생물",
	"신장 신생물",
	"이행상피세포암종",
	"방광 양성 신생물",
	"방광 악성 신생물",
	"방광 신생물",
	"비뇨기계 양성 신생물",
	"비뇨기계 악성 신생물",
	"비뇨기계 신생물",
	"신우 신염",
	"수신증",
	"신장 결석",
	"방광염",
	"방광 결석",
	"요도 폐색",
	"요로 결석증",
	"신경성 배뇨 이상",
	"비뇨기계 질환",
	"혈뇨",
	"요실금",
	"비정상 성분 소변",
	"핍뇨",
	"지방종",
	"조직구종",
	"유두종",
	"피지종",
	"모낭상피종",
	"기저세포종",
	"비만세포종",
	"악성 비만세포종",
	"흑색종",
	"피부 림프종",
	"편평세포암종",
	"항문주위선종",
	"항문주위선암종",
	"피부 신생물",
	"외이도염",
	"외이염",
	"중이염",
	"내이염",
	"농피증",
	"세균성 피부염",
	"말라세지아 피부염",
	"피부 사상균증",
	"곰팡이성 피부염",
	"모낭염",
	"모낭충증",
	"식이 알러지",
	"알러지 피부염",
	"아토피",
	"만성 피부염",
	"지루성 피부염",
	"피하 농양",
	"지방층염",
	"호산구성 육아종",
	"홍반루프스",
	"천포창",
	"지간 피부염",
	"족피부염",
	"꼬리샘 과증식",
	"발톱 주위염",
	"옴진드기",
	"개선충",
	"외부 기생충",
	"피부 질환",
	"귀 가려움",
	"발진",
	"피부염",
	"피부 가려움",
	"탈모",
	"선천적 질병",
	"유전적 질병",
	"파보 바이러스 감염",
	"디스템퍼 바이러스 감염",
	"파라인플루엔자 감염",
	"전염성 간염",
	"아데노 바이러스 2형 감염",
	"광견병",
	"코로나 바이러스 감염",
	"렙토스피라 감염",
	"필라리아 감염",
	"심장사상충 감염",
	"인플루엔자 감염",
	"상해",
	"질병",
	"예방접종",
	"치료",
	"검사",
	"투약",
	"정기검진",
	"예방검사",
	"임신",
	"출산",
	"제왕절개",
	"인공유산",
	"증상 치료",
	"중성화 수술",
	"불임 수술",
	"피임 수술",
	"미용 시술",
	"귀 성형",
	"꼬리 성형",
	"성대 제거",
	"미용성형 수술",
	"손톱 절제",
	"며느리발톱 제거",
	"잔존유치",
	"잠복고환",
	"제대허니아",
	"배꼽부위 탈장",
	"항문낭 제거",
	"외과수술",
	"점안",
	"귀청소",
	"속눈썹 질환",
	"눈물샘 질환",
	"식이요법",
	"의약품 처방",
	"건강보조식품",
	"한방약",
	"한의학 치료",
	"침술",
	"인도의학",
	"허브요법",
	"아로마테라피",
	"대체의료",
	"재활치료",
	"목욕",
	"약욕",
	"처방샴푸",
	"기생충 제거",
	"벼룩 감염",
	"진드기 감염",
	"모낭충 감염",
	"기생충 질환",
	"안락사",
	"해부검사",
]


def get_terms_dir() -> Path:
	try:
		from app.agents.document_parser.constants import TERMS_DIR

		return Path(TERMS_DIR)
	except Exception:
		return Path(__file__).resolve().parents[2] / "document_parser" / "data" / "terms"


try:
	KIWI = Kiwi()
	KIWI_AVAILABLE = True
except ImportError:
	KIWI_AVAILABLE = False
	KIWI = None


def tokenize_korean(text: str) -> List[str]:
	if not KIWI_AVAILABLE or KIWI is None:
		return text.lower().split()

	result = KIWI.tokenize(text)
	tokens: List[str] = []
	for token in result:
		pos = token.tag
		form = token.form.lower()
		if pos in ["NNG", "NNP", "NNB"]:
			tokens.append(form)
		elif pos in ["VV", "VA"]:
			tokens.append(form)

	return tokens


def load_chunks_from_document_parser(source_doc_name: str) -> List[Dict[str, Any]]:
	source_doc_stem = Path(source_doc_name).stem
	terms_dir = get_terms_dir()
	chunks_dir = terms_dir / source_doc_stem / "chunks"
	if not chunks_dir.exists():
		return []

	chunk_files = sorted(chunks_dir.glob(f"{source_doc_stem}_*.py"))
	chunks: List[Dict[str, Any]] = []

	for idx, file_path in enumerate(chunk_files):
		if file_path.name.endswith("_tagging_summary.py"):
			continue

		try:
			module_name = f"dp_chunk_{source_doc_stem}_{idx}"
			spec = importlib.util.spec_from_file_location(module_name, str(file_path))
			if spec is None or spec.loader is None:
				continue

			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
			chunk_obj = getattr(module, "chunk", None)
			if not isinstance(chunk_obj, dict):
				continue

			text = chunk_obj.get("page_content", "")
			chunks.append({"id": file_path.stem, "text": text})
		except Exception:
			continue

	return chunks


def load_chunks_from_terms_pdf(source_doc_name: str) -> List[Dict[str, Any]]:
	terms_dir = get_terms_dir()
	pdf_path = terms_dir / source_doc_name
	if not pdf_path.exists():
		return []

	try:
		reader = PdfReader(str(pdf_path))
	except Exception:
		return []

	chunks: List[Dict[str, Any]] = []
	for idx, page in enumerate(reader.pages, start=1):
		try:
			text = page.extract_text() or ""
		except Exception:
			text = ""
		if not text.strip():
			continue
		chunks.append({"id": f"p{idx}", "text": text})

	return chunks


def build_vocab(chunks: List[Dict[str, Any]], predefined_words: List[str]) -> Dict[str, int]:
	vocab: Dict[str, int] = {}

	for chunk in chunks:
		text = chunk.get("text", "")
		tokens = tokenize_korean(text)
		for token in tokens:
			if token not in vocab:
				vocab[token] = len(vocab)

		text_lower = text.lower()
		for word in predefined_words:
			word_lower = word.lower()
			if word_lower in text_lower and word_lower not in vocab:
				vocab[word_lower] = len(vocab)

	for word in predefined_words:
		word_lower = word.lower()
		if word_lower not in vocab:
			vocab[word_lower] = len(vocab)

	return vocab


def save_vocab_jsond(output_dir: Path, vocab: Dict[str, int], predefined_words: List[str]) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "vocab.jsond"
	payload = {
		"vocab": vocab,
		"predefined_words": predefined_words,
		"total_tokens": len(vocab),
	}
	with output_path.open("w", encoding="utf-8") as fp:
		json.dump(payload, fp, ensure_ascii=True)


def main() -> None:
	terms_dir = get_terms_dir()
	pdf_files = sorted(terms_dir.glob("*.pdf"))
	token_counts: Dict[str, int] = {}

	if not pdf_files:
		print(f"[WARN] No PDF files found in {terms_dir}")
		return

	for pdf_path in pdf_files:
		source_doc_name = pdf_path.name
		source_doc_stem = pdf_path.stem
		output_dir = terms_dir / source_doc_stem
		output_path = output_dir / "vocab.jsond"

		if output_path.exists():
			print(f"[SKIP] vocab.jsond already exists: {output_path}")
			continue

		chunks = load_chunks_from_document_parser(source_doc_name)
		if not chunks:
			chunks = load_chunks_from_terms_pdf(source_doc_name)

		if not chunks:
			print(f"[WARN] No chunks found for {source_doc_name}")
			continue

		vocab = build_vocab(chunks, PREDEFINED_WORDS)
		save_vocab_jsond(output_dir, vocab, PREDEFINED_WORDS)
		token_counts[source_doc_name] = len(vocab)
		print(f"[OK] Saved vocab.jsond for {source_doc_name}")

	if token_counts:
		print("\n[Token counts per document]")
		for doc_name, count in token_counts.items():
			print(f"- {doc_name}: {count}")


if __name__ == "__main__":
	main()
