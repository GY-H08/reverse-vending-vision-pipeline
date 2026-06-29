"""
judge_top_frames.py

상단 카메라에서 수집한 다중 프레임을 가중 투표(weighted voting) 방식으로
판정한다. 단일 프레임의 노이즈에 흔들리지 않도록, 다수의 프레임 중
신뢰도 높은 것들로만 투표하고, 실패 시 단순 다수결로 한 번 더 보완한다.

실제 운영 코드에서 사내 클래스 라벨명, 통신 프로토콜, 설정값을 제거하고
판정 알고리즘 구조만 일반화했다.
"""

from collections import Counter
from typing import Any, Optional


MIN_SCORE = 55          # 이 점수 미만이면 해당 프레임은 투표에서 제외
MIN_SCORE_GAP = 5        # 1위/2위 라벨 점수 차이가 이 값보다 작으면 불확실 처리
MIN_CONFIDENCE = 0.6     # 표(vote) 결과의 최소 신뢰도


def _vote(frames: list[dict[str, Any]], score_key: str, label_key: str,
          min_score_override: Optional[float] = None) -> tuple[Optional[str], Optional[float], float]:
    """
    여러 프레임의 (label, score) 쌍을 모아 가중 투표한다.
    1위 라벨이 충분히 우세하지 않으면 (None, None, 0.0)을 반환한다.
    """
    min_score = min_score_override or MIN_SCORE
    valid = [
        f for f in frames
        if f.get(score_key) is not None and f[score_key] >= min_score
    ]
    if not valid:
        return None, None, 0.0

    weighted = Counter()
    score_sum: dict[str, float] = {}
    for f in valid:
        label = f[label_key]
        weighted[label] += f[score_key]
        score_sum[label] = score_sum.get(label, 0.0) + f[score_key]

    ranked = weighted.most_common()
    if len(ranked) >= 2:
        top_label, top_w = ranked[0]
        second_label, second_w = ranked[1]
        if (top_w - second_w) < MIN_SCORE_GAP:
            return None, None, 0.0
    else:
        top_label, _ = ranked[0]

    count = sum(1 for f in valid if f[label_key] == top_label)
    confidence = count / len(valid)
    if confidence < MIN_CONFIDENCE:
        return None, None, 0.0

    avg_score = score_sum[top_label] / count
    return top_label, avg_score, confidence


def _majority_label(frames: list[dict[str, Any]], label_key: str,
                     score_key: str) -> tuple[Optional[str], Optional[float]]:
    """엄격한 투표가 실패했을 때 사용하는 단순 다수결 fallback."""
    labels = [f[label_key] for f in frames if f.get(label_key)]
    if not labels:
        return None, None
    top_label, _ = Counter(labels).most_common(1)[0]
    scores = [f[score_key] for f in frames if f.get(label_key) == top_label]
    avg = sum(scores) / len(scores) if scores else None
    return top_label, avg


def _is_present_label(label: Optional[str]) -> bool:
    """라벨 문자열에 '있음'이 포함되어 있는지로 존재 여부 판단."""
    return bool(label) and "있음" in label


def judge_top_frames(static_frames: list[dict], ripple_frames: list[dict]) -> dict:
    """
    static_frames(정지 구간): 컵 종류, 이물질, 뚜껑처럼 정적인 항목 판별
    ripple_frames(파동 구간): 휘핑크림, 물처럼 액상류 항목 판별

    컵 종류는 static_frames에서 한 번만 확정하고, 그 결과를
    ripple_frames의 클래스 선택 기준으로도 그대로 사용한다.
    """
    cup_type, cup_score, _ = _vote(static_frames, "score_cup_type", "label_cup_type")
    if cup_type is None:
        cup_type, cup_score = _majority_label(static_frames, "label_cup_type", "score_cup_type")

    if cup_type is None:
        return {"success": False, "reason": "cup_type_vote_failed"}

    is_paper = cup_type == "paper"

    foreign_key = "paper_foreign" if is_paper else "plastic_foreign"
    lid_key = "paper_lid" if is_paper else "plastic_lid"
    whipping_key = "paper_whipping" if is_paper else "plastic_whipping"
    water_key = "paper_water" if is_paper else "plastic_water"

    foreign, foreign_score, _ = _vote(static_frames, f"score_{foreign_key}", f"label_{foreign_key}")
    lid, lid_score, _ = _vote(static_frames, f"score_{lid_key}", f"label_{lid_key}")
    whipping, whipping_score, _ = _vote(ripple_frames, f"score_{whipping_key}", f"label_{whipping_key}")
    water, water_score, _ = _vote(ripple_frames, f"score_{water_key}", f"label_{water_key}")

    labels_to_check = [foreign, whipping, water, lid]
    success = not any(_is_present_label(label) for label in labels_to_check)

    return {
        "success": success,
        "cup_type": cup_type,
        "cup_type_score": cup_score,
        "foreign_material": foreign,
        "foreign_score": foreign_score,
        "whipping_cream": whipping,
        "whipping_cream_score": whipping_score,
        "water": water,
        "water_score": water_score,
        "top_lid": lid,
        "top_lid_score": lid_score,
    }
