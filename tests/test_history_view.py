from __future__ import annotations

import unittest

from ui.history_view import build_history_detail_cells


class HistoryViewFormattingTests(unittest.TestCase):
    def test_detail_cells_use_path_aware_status_and_never_show_nan(self) -> None:
        cells = build_history_detail_cells(
            {
                "ticker": "MRVL",
                "entry_price": 99.05,
                "target_price": 104.00,
                "current_price": 101.25,
                "window_high_price": 104.60,
                "realized_return_pct": -1.2,
                "hit_target": True,
                "sector": "Technology",
            }
        )

        self.assertEqual(cells[0].strip(), "MRVL")
        self.assertEqual(cells[1], "Entry $99.05")
        self.assertEqual(cells[2], "Target $104.00")
        self.assertEqual(cells[3], "High $104.60")
        self.assertEqual(cells[4], "Target hit · -1.2%")
        self.assertEqual(cells[5], "Technology")

    def test_detail_cells_fall_back_to_pending_when_path_data_missing(self) -> None:
        cells = build_history_detail_cells(
            {
                "ticker": "ARM",
                "entry_price": 157.07,
                "target_price": None,
                "current_price": None,
                "window_high_price": None,
                "realized_return_pct": None,
                "hit_target": False,
                "sector": "Technology",
            }
        )

        self.assertEqual(cells[2], "Target pending")
        self.assertEqual(cells[3], "Latest pending")
        self.assertEqual(cells[4], "Pending")
        self.assertNotIn("nan", " ".join(cells).lower())


if __name__ == "__main__":
    unittest.main()
