"""Tests for TimingStats — rolling timing statistics tracker."""

import pytest

from segmentation.segmentation import TimingStats


class TestTimingStats:
    """Unit tests for the TimingStats helper class."""

    def test_add_and_get_single_step(self):
        """Adding timings to one step and retrieving the average."""
        ts = TimingStats(max_samples=100)
        ts.add_timing("decode", 10.0)
        ts.add_timing("decode", 20.0)
        ts.add_timing("decode", 30.0)
        avgs = ts.get_average_timings()
        assert avgs["decode"] == pytest.approx(20.0)

    def test_multiple_steps(self):
        """Different step names are tracked independently."""
        ts = TimingStats(max_samples=100)
        ts.add_timing("decode", 5.0)
        ts.add_timing("decode", 15.0)
        ts.add_timing("inference", 100.0)
        ts.add_timing("inference", 200.0)
        avgs = ts.get_average_timings()
        assert avgs["decode"] == pytest.approx(10.0)
        assert avgs["inference"] == pytest.approx(150.0)

    def test_rolling_window_evicts_old_samples(self):
        """When max_samples is exceeded, oldest values are dropped."""
        ts = TimingStats(max_samples=3)
        ts.add_timing("step", 100.0)  # will be evicted
        ts.add_timing("step", 10.0)
        ts.add_timing("step", 20.0)
        ts.add_timing("step", 30.0)
        avgs = ts.get_average_timings()
        # Only the last 3 values (10, 20, 30) should remain
        assert avgs["step"] == pytest.approx(20.0)

    def test_empty_stats_returns_empty_dict(self):
        """No timings added → empty dict."""
        ts = TimingStats()
        avgs = ts.get_average_timings()
        assert avgs == {}

    def test_single_sample_returns_itself(self):
        """A single timing value should be its own average."""
        ts = TimingStats()
        ts.add_timing("x", 42.0)
        assert ts.get_average_timings()["x"] == pytest.approx(42.0)

    def test_max_samples_one(self):
        """With max_samples=1 only the latest value survives."""
        ts = TimingStats(max_samples=1)
        ts.add_timing("s", 1.0)
        ts.add_timing("s", 2.0)
        ts.add_timing("s", 3.0)
        assert ts.get_average_timings()["s"] == pytest.approx(3.0)
