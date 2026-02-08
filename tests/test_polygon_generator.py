"""Tests for PolygonGenerator — synthetic polygon broadcaster used for
frontend testing without a camera."""

import json
import math
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "segmentation"))

from polygon_generator import PolygonGenerator


# ---------------------------------------------------------------------------
# load_polygon_config
# ---------------------------------------------------------------------------

class TestLoadPolygonConfig:
    """Tests for PolygonGenerator.load_polygon_config()."""

    def test_loads_valid_json(self):
        """Should correctly read a well-formed JSON config file."""
        config_path = os.path.join(
            PROJECT_ROOT, "segmentation", "polygon_config", "rectangle.json",
        )
        gen = PolygonGenerator.__new__(PolygonGenerator)
        config = gen.load_polygon_config(config_path)
        assert config["name"] == "rectangle"
        assert "vertices" in config
        assert isinstance(config["vertices"], list)
        assert len(config["vertices"]) >= 3

    def test_missing_file_returns_defaults(self):
        """A missing config file should fall back to the built-in default."""
        gen = PolygonGenerator.__new__(PolygonGenerator)
        config = gen.load_polygon_config("/nonexistent/path.json")
        assert config["name"] == "rectangle"
        assert len(config["vertices"]) == 4

    def test_invalid_json_raises(self):
        """Malformed JSON should raise JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{bad json!!")
            f.flush()
            gen = PolygonGenerator.__new__(PolygonGenerator)
            with pytest.raises(json.JSONDecodeError):
                gen.load_polygon_config(f.name)
        os.unlink(f.name)

    def test_loads_all_shipped_configs(self):
        """All JSON files in polygon_config/ should load without error."""
        config_dir = os.path.join(PROJECT_ROOT, "segmentation", "polygon_config")
        for fname in os.listdir(config_dir):
            if fname.endswith(".json"):
                gen = PolygonGenerator.__new__(PolygonGenerator)
                config = gen.load_polygon_config(os.path.join(config_dir, fname))
                assert "name" in config
                assert "vertices" in config


# ---------------------------------------------------------------------------
# calculate_polygon_data
# ---------------------------------------------------------------------------

class TestCalculatePolygonData:
    """Tests for PolygonGenerator.calculate_polygon_data() — pure math."""

    @pytest.fixture
    def generator(self):
        """A PolygonGenerator with rectangle config (default)."""
        config_path = os.path.join(
            PROJECT_ROOT, "segmentation", "polygon_config", "rectangle.json",
        )
        return PolygonGenerator(config_file=config_path)

    def test_returns_required_keys(self, generator):
        """Output dict must contain position, vertices, rotation."""
        data = generator.calculate_polygon_data(generator.start_time)
        assert "position" in data
        assert "vertices" in data
        assert "rotation" in data
        assert "x" in data["position"]
        assert "y" in data["position"]

    def test_vertex_count_matches_config(self, generator):
        """Number of output vertices must match the config."""
        data = generator.calculate_polygon_data(generator.start_time)
        assert len(data["vertices"]) == len(generator.config["vertices"])

    def test_y_position_oscillates(self, generator):
        """Y position should follow a sine wave — peaks and troughs differ."""
        # At t=0 → sin(0)=0, at quarter-period → sin(π/2)=1
        t0 = generator.start_time
        period = 1.0 / generator.frequency  # 2 seconds
        y_at_0 = generator.calculate_polygon_data(t0)["position"]["y"]
        y_at_quarter = generator.calculate_polygon_data(t0 + period / 4)["position"]["y"]
        y_at_half = generator.calculate_polygon_data(t0 + period / 2)["position"]["y"]
        # At quarter-period the sine is at peak → y should differ from t=0
        assert y_at_quarter != pytest.approx(y_at_0, abs=1.0)
        # At half-period the sine returns to ~0 → y close to t=0
        assert y_at_half == pytest.approx(y_at_0, abs=1.0)

    def test_x_position_is_constant(self, generator):
        """X position should stay at size/2 regardless of time."""
        t0 = generator.start_time
        expected_x = generator.size / 2
        for dt in [0, 0.5, 1.0, 2.0, 5.0]:
            data = generator.calculate_polygon_data(t0 + dt)
            assert data["position"]["x"] == pytest.approx(expected_x)

    def test_rotation_increases_with_time(self, generator):
        """Rotation should increase monotonically over time."""
        t0 = generator.start_time
        r0 = generator.calculate_polygon_data(t0)["rotation"]
        r1 = generator.calculate_polygon_data(t0 + 1.0)["rotation"]
        r2 = generator.calculate_polygon_data(t0 + 2.0)["rotation"]
        assert r1 > r0
        assert r2 > r1
