"""6002번 강남행 직전 정류장의 잔여 좌석으로 목표 정류장 만석을 예측한다."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ROUTE_ID = "233000136"
TARGET_STATION_ID = "233002421"
# 노선 전체 순번 12→13은 회차점(39번) 이전의 강남행 구간이다.
TARGET_SEQUENCE = 13
KOREA_TIME = timezone(timedelta(hours=9))


def valid_seat_count(seats):
    return type(seats) is int and seats >= 0


def predict_full(current_seats):
    """현재 좌석이 0이면 만석, 양수이면 비만석, 알 수 없으면 보류한다."""
    if not valid_seat_count(current_seats):
        return None
    return current_seats == 0


def observation_time(observation):
    value = observation.get("observed_at")
    if not isinstance(value, str):
        return None
    # Python 3.9에서도 UTC의 Z 표기를 읽을 수 있게 한다.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None:
            return None
        return timestamp.astimezone(KOREA_TIME)
    except (ValueError, OverflowError):
        return None


def eligible_before(before):
    """같은 예측 시점을 비교하도록 강남행 12번 정류장의 식별 가능한 관측만 쓴다."""
    for field in ("trip_id", "vehicle_id"):
        value = before.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    sequence = before.get("station_sequence")
    return (before.get("route_id") == ROUTE_ID
            and type(sequence) is int and sequence == TARGET_SEQUENCE - 1
            and observation_time(before) is not None)


def arrival_label(event):
    """A/B 검토와 관측값이 뒷받침하는 도착 정답 또는 제외 사유를 돌려준다."""
    before = event["before"]
    arrival = event.get("arrival")
    if arrival is None:
        return None, "arrival_not_observed"
    if not isinstance(arrival, dict):
        raise ValueError("arrival에는 관측 객체 또는 null이 필요합니다.")
    if event.get("arrival_verified") is not True:
        return None, "arrival_not_verified"
    if any(arrival.get(field) != before[field] for field in ("trip_id", "vehicle_id", "route_id")):
        return None, "different_trip"
    sequence = arrival.get("station_sequence")
    if (arrival.get("station_id") != TARGET_STATION_ID
            or type(sequence) is not int or sequence != TARGET_SEQUENCE):
        return None, "wrong_target"
    before_at = observation_time(before)
    arrived_at = observation_time(arrival)
    if before_at is None or arrived_at is None or arrived_at <= before_at:
        return None, "invalid_arrival_time"
    if arrived_at.date() != before_at.date():
        return None, "different_service_date"
    seats = arrival.get("remaining_seats")
    if not valid_seat_count(seats):
        return None, "unknown_arrival_seats"
    return seats == 0, None


def evaluate(events):
    """한 운행의 첫 12번 관측 하나를 사용한다. 시점이 다른 중복 입력은 거절한다."""
    if not isinstance(events, list):
        raise ValueError("검토 자료는 운행별 행을 담은 JSON 배열이어야 합니다.")
    seen = {}
    reasons = {}
    excluded = 0
    duplicates = 0
    guided = 0
    labels = 0
    scored = 0
    misses = 0
    false_alarms = 0
    for row_number, event in enumerate(events, 1):
        if not isinstance(event, dict) or not isinstance(event.get("before"), dict):
            raise ValueError(f"{row_number}행에 before 관측 객체가 필요합니다.")
        before = event["before"]
        if not eligible_before(before):
            excluded += 1
            continue

        trip_id = before["trip_id"]
        # JSON으로 비교해야 false와 0, true와 1이 같은 자료로 취급되지 않는다.
        signature = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if trip_id in seen:
            if seen[trip_id] != signature:
                raise ValueError(f"{row_number}행의 운행 기록이 중복·불일치합니다. "
                                 "해당 운행의 첫 12번 관측과 도착 근거를 대조하세요.")
            duplicates += 1
            continue
        seen[trip_id] = signature

        prediction = predict_full(before.get("remaining_seats"))
        if prediction is not None:
            guided += 1
        actual, reason = arrival_label(event)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        labels += 1
        if prediction is not None:
            scored += 1
            if actual and not prediction:
                misses += 1
            if prediction and not actual:
                false_alarms += 1

    eligible = len(seen)
    return {
        "input_rows": len(events),
        "excluded_before_rows": excluded,
        "duplicate_rows": duplicates,
        "eligible_runs": eligible,
        "guided_runs": guided,
        "abstained_runs": eligible - guided,
        "guidance_coverage": guided / eligible if eligible else None,
        "abstention_rate": (eligible - guided) / eligible if eligible else None,
        "verified_labels": labels,
        "unlabeled_reasons": reasons,
        "scored_runs": scored,
        "full_misses": misses,
        "false_alarms": false_alarms,
        "accuracy": (scored - misses - false_alarms) / scored if scored else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="B 내부 검토용 JSON 배열")
    args = parser.parse_args()
    try:
        events = json.loads(args.events.read_text(encoding="utf-8"))
        report = evaluate(events)
    except (OSError, ValueError) as error:
        parser.exit(1, f"평가 자료 확인 실패: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
