"""
judge_side_frames.py

측면 카메라 바코드 판정 로직.

문제 상황:
- 회전판 규격이 커서 컵이 정중앙이 아닌 좌/우에 투입되는 경우가 많았다.
- 단일 중앙 ROI만으로는 좌우로 치우친 바코드를 오인식하거나 놓치는
  케이스가 발생했다.
- 또한 훼손되거나 두 개의 바코드가 겹쳐 붙은 컵이 우연히 한 프레임만
  인식되어 통과되는 부정 사례도 있었다.

해결:
- 바코드 인식 영역을 중앙(주 ROI) + 좌 + 우, 총 3분할로 확장
- 좌/우 ROI에서 단 한 번이라도 바코드가 인식되면 위치 이상으로 즉시 거부
- 다중 프레임 내에서 서로 다른 바코드 값이 2개 이상 검출되면 즉시 거부
  (이중 바코드 / 훼손 컵 판정)
"""

from typing import Any, Optional


MIN_VALID_FRAME_RATIO = 0.5  # 유효 프레임이 전체의 절반 미만이면 판정 실패


def judge_side_frames(frames: list[dict[str, Any]]) -> dict:
    """
    frames: 한 번의 회전 동안 수집된 다중 프레임 리스트.
    각 프레임은 다음 키를 가진다고 가정:
        - barcode_center_state / barcode_center_data
        - barcode_left_state   / barcode_left_data
        - barcode_right_state  / barcode_right_data
        - holder_label / holder_score
    """
    if not frames:
        return {"success": False, "reason": "no_frames_collected"}

    # 1) 좌/우 ROI 안전 게이트: 위치 이상 즉시 거부
    off_center_detected = any(
        f.get("barcode_left_state") == "1" or f.get("barcode_right_state") == "1"
        for f in frames
    )
    if off_center_detected:
        return {
            "success": False,
            "reason": "side_barcode_off_center_detected",
            "barcode_ok": False,
            "barcode_data": None,
        }

    # 2) 중앙 ROI 이중 바코드(훼손 컵) 감지
    detected_values = {
        f["barcode_center_data"]
        for f in frames
        if f.get("barcode_center_state") == "1" and f.get("barcode_center_data")
    }
    if len(detected_values) >= 2:
        return {
            "success": False,
            "reason": "barcode_duplicate_detected",
            "barcode_ok": False,
            "barcode_data": None,
        }

    # 3) 중앙 ROI 바코드 정상 인식 여부 (1개 이상 인식되면 통과 후보)
    barcode_data: Optional[str] = next(iter(detected_values), None)
    barcode_ok = barcode_data is not None

    # 4) 홀더(거치대) 유무 판정 — 있으면 거부
    holder_labels = [f["holder_label"] for f in frames if f.get("holder_label")]
    holder_present = any("있음" in label for label in holder_labels)

    if not barcode_ok:
        return {
            "success": False,
            "reason": "barcode_not_detected",
            "barcode_ok": False,
            "barcode_data": None,
        }

    if holder_present:
        return {
            "success": False,
            "reason": "holder_detected",
            "barcode_ok": True,
            "barcode_data": barcode_data,
        }

    return {
        "success": True,
        "reason": None,
        "barcode_ok": True,
        "barcode_data": barcode_data,
    }
