import { normalizeRiskFactors } from "../lib/normalizers";

const RISK_COLOR = {
  critical: "#d44c2d",
  high: "#e07a20",
  medium: "#af8f1b",
  low: "#2f7b6a",
  passthrough: "#3e5b8d",
};

function toTextList(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function dedupe(items) {
  const seen = new Set();
  return items.filter((item) => {
    if (seen.has(item)) {
      return false;
    }
    seen.add(item);
    return true;
  });
}

function buildSuggestedActions(report, factors) {
  const direct = toTextList(report.suggested_actions);
  if (direct.length > 0) {
    return dedupe(direct);
  }

  const combined = [];
  if (report.prevention_strategy && typeof report.prevention_strategy === "object") {
    combined.push(...toTextList(report.prevention_strategy.short_term));
    combined.push(...toTextList(report.prevention_strategy.long_term));
  }
  if (report.control_gap_analysis && typeof report.control_gap_analysis === "object") {
    combined.push(...toTextList(report.control_gap_analysis.recommendations));
  }
  for (const factor of factors) {
    combined.push(...toTextList(factor.suggestions));
  }

  return dedupe(combined);
}

function formatCase(caseItem) {
  if (!caseItem || typeof caseItem !== "object") {
    return String(caseItem || "");
  }

  const content = String(caseItem.content || "내용 없음");
  const source = String(caseItem.source || "unknown");
  const category = String(caseItem.category || "").trim();
  const distance = Number(caseItem.distance);

  const parts = [content, `출처: ${source}`];
  if (category) {
    parts.push(`분류: ${category}`);
  }
  if (Number.isFinite(distance)) {
    parts.push(`거리: ${distance.toFixed(3)}`);
  }

  if (Array.isArray(caseItem.topic) && caseItem.topic.length > 0) {
    parts.push(`토픽: ${caseItem.topic.map(String).join(", ")}`);
  }

  return parts.join(" | ");
}

export default function ReportCard({ report }) {
  const scoring = report?.risk_scoring && typeof report.risk_scoring === "object" ? report.risk_scoring : {};
  const level = String(scoring.overall_risk_level || "unknown").toLowerCase();
  const factors = normalizeRiskFactors(report || {});

  let score = Number(scoring.overall_risk_score);
  if (!Number.isFinite(score)) {
    if (factors.length > 0) {
      score = factors.reduce((sum, factor) => sum + factor.score, 0) / factors.length;
    } else {
      score = 0;
    }
  }

  const summary =
    String(
      report?.response_summary ||
        report?.input_summary ||
        scoring?.overall_assessment ||
        "요약 정보가 없습니다."
    );

  const actions = buildSuggestedActions(report || {}, factors);
  const similarCases = Array.isArray(report?.similar_cases) ? report.similar_cases : [];
  const evidence = toTextList(report?.evidence_sources);

  return (
    <div className="report-card">
      <div className="report-head">
        <strong>리스크 분석 보고서</strong>
        <span className="report-id">ID: {String(report?.analysis_id || "N/A").slice(0, 12)}</span>
      </div>

      <div className="report-metrics">
        <span className="risk-badge" style={{ backgroundColor: RISK_COLOR[level] || "#455a64" }}>
          종합 리스크: {level}
        </span>
        <span className="score-badge">점수 {score.toFixed(1)}</span>
      </div>

      <section>
        <h4>요약</h4>
        <p>{summary}</p>
      </section>

      <section>
        <h4>주요 리스크 요인</h4>
        {factors.length === 0 ? (
          <p className="muted">리스크 요인 데이터가 없습니다.</p>
        ) : (
          <ul className="report-list">
            {factors.map((factor) => (
              <li key={`${factor.name}-${factor.impact}-${factor.likelihood}`}>
                <strong>{factor.name}</strong>
                <div>
                  영향 {factor.impact}, 가능성 {factor.likelihood}, 점수 {factor.score.toFixed(1)}
                </div>
                {factor.reason ? <p>{factor.reason}</p> : null}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h4>제안 조치</h4>
        {actions.length === 0 ? (
          <p className="muted">제안 조치가 없습니다.</p>
        ) : (
          <ul className="report-list">
            {actions.map((action) => (
              <li key={action}>{action}</li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h4>유사 사례 및 근거</h4>
        {similarCases.length > 0 ? (
          <ul className="report-list">
            {similarCases.map((caseItem, index) => (
              <li key={`${index}-${String(caseItem?.content || "")}`}>{formatCase(caseItem)}</li>
            ))}
          </ul>
        ) : null}
        {evidence.length > 0 ? (
          <ul className="report-list">
            {evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        {similarCases.length === 0 && evidence.length === 0 ? (
          <p className="muted">근거 자료가 없습니다.</p>
        ) : null}
      </section>
    </div>
  );
}
