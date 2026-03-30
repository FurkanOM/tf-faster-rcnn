"""Unit tests for mean average precision helpers."""

from __future__ import annotations

import types
import unittest
from typing import Any, Iterable, Iterator, List

from tests.test_support import import_project_module


class _FakeArray:
    def __init__(self, values: Iterable[float]) -> None:
        self.values = list(values)

    def __add__(self, other: Any) -> "_FakeArray":
        other_values = other.values if isinstance(other, _FakeArray) else other
        if isinstance(other_values, list):
            return _FakeArray([left + right for left, right in zip(self.values, other_values)])
        return _FakeArray([value + other_values for value in self.values])

    def __ge__(self, other: float) -> List[bool]:
        return [value >= other for value in self.values]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, _FakeArray):
            item = item.values
        if isinstance(item, list):
            if item and all(isinstance(value, bool) for value in item):
                return _FakeArray(
                    [value for value, is_selected in zip(self.values, item) if is_selected]
                )
            return _FakeArray([self.values[index] for index in item])
        return self.values[item]

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __neg__(self) -> "_FakeArray":
        return _FakeArray([-value for value in self.values])

    def __truediv__(self, other: Any) -> "_FakeArray":
        other_values = other.values if isinstance(other, _FakeArray) else other
        if isinstance(other_values, list):
            return _FakeArray([left / right for left, right in zip(self.values, other_values)])
        return _FakeArray([value / other_values for value in self.values])

    def to_list(self) -> List[float]:
        return list(self.values)


def _make_numpy_stub() -> types.ModuleType:
    numpy_stub = types.ModuleType("numpy")

    def array(values: Iterable[float]) -> _FakeArray:
        return _FakeArray(values)

    def arange(start: float, stop: float, step: float) -> List[float]:
        values = []
        current = start
        while current <= stop + (step / 2):
            values.append(round(current, 10))
            current += step
        return values

    def amax(values: Any) -> float:
        values = values.values if isinstance(values, _FakeArray) else values
        return max(values)

    def argsort(values: Any) -> List[int]:
        values = values.values if isinstance(values, _FakeArray) else values
        return [index for index, _ in sorted(enumerate(values), key=lambda item: item[1])]

    def cumsum(values: Any) -> _FakeArray:
        values = values.values if isinstance(values, _FakeArray) else values
        total = 0
        accumulated = []
        for value in values:
            total += value
            accumulated.append(total)
        return _FakeArray(accumulated)

    def mean(values: Any) -> float:
        values = values.values if isinstance(values, _FakeArray) else values
        return sum(values) / len(values)

    numpy_stub.array = array
    numpy_stub.arange = arange
    numpy_stub.amax = amax
    numpy_stub.argsort = argsort
    numpy_stub.cumsum = cumsum
    numpy_stub.mean = mean
    return numpy_stub


class EvalUtilsTests(unittest.TestCase):
    """Validate metric bookkeeping on deterministic inputs."""

    def setUp(self) -> None:
        self.module, _ = import_project_module(
            "utils.eval_utils",
            extra_modules={"numpy": _make_numpy_stub()},
            clear_modules=("utils.eval_utils", "utils.bbox_utils"),
        )

    def test_init_stats_skips_background_label(self) -> None:
        stats = self.module.init_stats(["bg", "person", "car"])

        self.assertEqual(sorted(stats.keys()), [1, 2])
        self.assertEqual(stats[1]["label"], "person")
        self.assertEqual(stats[2]["total"], 0)

    def test_calculate_ap_uses_11_point_interpolation(self) -> None:
        recall = self.module.np.array([0.25, 0.5, 0.75, 1.0])
        precision = self.module.np.array([1.0, 0.75, 0.5, 0.25])

        ap = self.module.calculate_ap(recall, precision)

        self.assertAlmostEqual(ap, 0.6363636364)

    def test_calculate_map_sorts_predictions_by_score(self) -> None:
        stats = {
            1: {
                "label": "person",
                "total": 1,
                "tp": [1, 0],
                "fp": [0, 1],
                "scores": [0.9, 0.2],
            },
            2: {
                "label": "car",
                "total": 1,
                "tp": [0, 1],
                "fp": [1, 0],
                "scores": [0.1, 0.8],
            },
        }

        calculated_stats, mean_ap = self.module.calculate_mAP(stats)

        self.assertAlmostEqual(calculated_stats[1]["AP"], 1.0)
        self.assertAlmostEqual(calculated_stats[2]["AP"], 1.0)
        self.assertAlmostEqual(mean_ap, 1.0)
        self.assertEqual(calculated_stats[2]["recall"].to_list(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
