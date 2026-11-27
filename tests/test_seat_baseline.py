"""가상 운행으로 기준선·정답 분리와 평가 분모를 검사한다."""
from copy import deepcopy
import unittest

from src.seat_baseline import evaluate, predict_full


def example_run(trip_id="day1-out-1", current=2, at_arrival=0):
    before = {
        "trip_id": trip_id,
        "vehicle_id": "test-bus",
        "route_id": "233000136",
        "station_sequence": 12,
        "observed_at": "2026-11-23T08:00:00+09:00",
        "remaining_seats": current,
    }
    arrival = dict(before, station_sequence=13, station_id="233002421",
                   observed_at="2026-11-23T08:03:00+09:00", remaining_seats=at_arrival)
    return {"before": before, "arrival": arrival, "arrival_verified": True}


class BaselineTests(unittest.TestCase):
    def test_zero_positive_and_unknown_inputs(self):
        self.assertIs(predict_full(0), True)
        self.assertIs(predict_full(1), False)
        for invalid in (None, -1, True, False, 0.0, "0", ""):
            with self.subTest(invalid=invalid):
                self.assertIsNone(predict_full(invalid))

    def test_comparable_runs_and_error_counts(self):
        events = [
            example_run("miss", 2, 0),
            example_run("alarm", 0, 1),
            example_run("hit", 0, 0),
            example_run("safe", 1, 5),
            example_run("abstain", None, 0),
            example_run("no-arrival", 3, None),
        ]
        result = evaluate(events)
        self.assertEqual(result["eligible_runs"], 6)
        self.assertEqual(result["verified_labels"], 5)
        self.assertEqual(result["scored_runs"], 4)
        self.assertEqual(result["full_misses"], 1)
        self.assertEqual(result["false_alarms"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["guided_runs"], 5)
        self.assertEqual(result["abstained_runs"], 1)
        self.assertEqual(result["guidance_coverage"], 5 / 6)
        self.assertEqual(result["abstention_rate"], 1 / 6)

    def test_unobserved_and_unverified_arrivals_are_not_ground_truth(self):
        rows = [example_run("no-arrival"), example_run("unverified")]
        rows[0]["arrival"] = None
        rows[1]["arrival_verified"] = 1
        result = evaluate(rows)
        self.assertEqual(result["guided_runs"], 2)
        self.assertEqual(result["unlabeled_reasons"],
                         {"arrival_not_observed": 1, "arrival_not_verified": 1})
        self.assertEqual(result["scored_runs"], 0)
        self.assertIsNone(result["accuracy"])

    def test_arrival_must_be_same_run_at_actual_target_and_later(self):
        for field, value, reason in (
            ("route_id", "other", "different_trip"),
            ("vehicle_id", "other", "different_trip"),
            ("trip_id", "repeat-trip", "different_trip"),
            ("station_id", "other", "wrong_target"),
            ("station_sequence", 14, "wrong_target"),
            ("station_sequence", 13.0, "wrong_target"),
            ("observed_at", "2026-11-22T23:00:00Z", "invalid_arrival_time"),
            ("observed_at", "2026-11-22T22:59:59Z", "invalid_arrival_time"),
            ("observed_at", "2026-11-23T08:01", "invalid_arrival_time"),
            ("remaining_seats", -1, "unknown_arrival_seats"),
            ("remaining_seats", False, "unknown_arrival_seats"),
        ):
            with self.subTest(field=field, value=value):
                row = example_run()
                row["arrival"][field] = value
                result = evaluate([row])
                self.assertEqual(result["verified_labels"], 0)
                self.assertEqual(result["unlabeled_reasons"], {reason: 1})
                self.assertEqual(result["guided_runs"], 1)

    def test_arrival_must_share_korean_morning_service_date(self):
        # 현지 오전 8시의 UTC 표기는 전날 23시다.
        for arrival_time, accepted in (
            ("2026-11-23T08:03:00+09:00", True),
            ("2026-11-22T23:03:00Z", True),
            ("2026-11-23T23:03:00Z", False),
        ):
            with self.subTest(arrival_time=arrival_time):
                row = example_run()
                row["arrival"]["observed_at"] = arrival_time
                result = evaluate([row])
                self.assertEqual(result["verified_labels"], int(accepted))
                self.assertEqual(result["unlabeled_reasons"],
                                 {} if accepted else {"different_service_date": 1})

    def test_scope_and_missing_run_evidence_are_excluded(self):
        for field, value in (
            ("route_id", "other"), ("station_sequence", 11), ("station_sequence", 13),
            ("trip_id", "   "), ("vehicle_id", None), ("observed_at", "08:00"),
        ):
            with self.subTest(field=field, value=value):
                row = example_run()
                row["before"][field] = value
                result = evaluate([row])
                self.assertEqual(result["excluded_before_rows"], 1)
                self.assertEqual(result["eligible_runs"], 0)

    def test_duplicates_do_not_weight_runs_and_conflicts_require_review(self):
        row = example_run(current=0)
        self.assertEqual(evaluate([row, deepcopy(row)])["scored_runs"], 1)
        self.assertEqual(evaluate([row, deepcopy(row)])["duplicate_rows"], 1)
        conflicting = deepcopy(row)
        conflicting["before"]["remaining_seats"] = False
        with self.assertRaisesRegex(ValueError, "중복"):
            evaluate([row, conflicting])

    def test_missing_denominators_and_invalid_format(self):
        result = evaluate([])
        for field in ("accuracy", "guidance_coverage", "abstention_rate"):
            self.assertIsNone(result[field])
        for invalid in ({}, [None], [{"before": None}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate(invalid)


if __name__ == "__main__":
    unittest.main()
