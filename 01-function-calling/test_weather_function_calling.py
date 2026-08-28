"""Behavior tests for local function-calling weather tools."""

import importlib.util
import json
import os
from pathlib import Path
import unittest


os.environ.setdefault("GEMINI_API_KEY", "test-key")

MODULE_PATH = Path(__file__).with_name("weather_function_calling.py")
SPEC = importlib.util.spec_from_file_location("weather_function_calling", MODULE_PATH)
weather = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(weather)


class GetForecastTests(unittest.TestCase):
    def test_get_forecast_returns_requested_number_of_days(self) -> None:
        forecast = json.loads(weather.get_forecast("Hà Nội", days=2))

        self.assertEqual(forecast["city"], "Hà Nội")
        self.assertEqual(len(forecast["forecast"]), 2)
        self.assertEqual(forecast["forecast"][0]["ngày"], "Hôm nay")


if __name__ == "__main__":
    unittest.main()
