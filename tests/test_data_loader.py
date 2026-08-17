"""Unit tests for data loader."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import DataLoader, Sample


@pytest.fixture
def sample_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestLoadJSON:
    def test_load_json_list(self, sample_dir):
        data = [
            {"id": "1", "input": "q1", "output": "a1"},
            {"id": "2", "input": "q2", "output": "a2"},
        ]
        path = os.path.join(sample_dir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)

        loader = DataLoader(sample_dir)
        samples = loader.load("test.json")
        assert len(samples) == 2
        assert samples[0].input_text == "q1"

    def test_load_json_wrapped(self, sample_dir):
        data = {"samples": [{"id": "1", "input": "q", "output": "a"}]}
        path = os.path.join(sample_dir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)

        loader = DataLoader(sample_dir)
        samples = loader.load("test.json")
        assert len(samples) == 1


class TestLoadJSONL:
    def test_load_jsonl(self, sample_dir):
        lines = [
            json.dumps({"id": "1", "input": "q1", "output": "a1"}),
            json.dumps({"id": "2", "input": "q2", "output": "a2"}),
        ]
        path = os.path.join(sample_dir, "test.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        loader = DataLoader(sample_dir)
        samples = loader.load("test.jsonl")
        assert len(samples) == 2


class TestLoadCSV:
    def test_load_csv(self, sample_dir):
        path = os.path.join(sample_dir, "test.csv")
        with open(path, "w") as f:
            f.write("id,input,output\n")
            f.write("1,q1,a1\n")
            f.write("2,q2,a2\n")

        loader = DataLoader(sample_dir)
        samples = loader.load("test.csv")
        assert len(samples) == 2
        assert samples[1].expected_output == "a2"


class TestErrors:
    def test_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            DataLoader("/nonexistent/path")

    def test_missing_file(self, sample_dir):
        loader = DataLoader(sample_dir)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.json")

    def test_unsupported_format(self, sample_dir):
        path = os.path.join(sample_dir, "test.xyz")
        with open(path, "w") as f:
            f.write("data")
        loader = DataLoader(sample_dir)
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load("test.xyz")


class TestSaveSamples:
    def test_save_and_load(self, sample_dir):
        samples = [Sample(id="1", input_text="q", expected_output="a")]
        path = os.path.join(sample_dir, "out.json")
        DataLoader.save_samples(samples, path)

        loader = DataLoader(sample_dir)
        loaded = loader.load("out.json")
        assert len(loaded) == 1
        assert loaded[0].input_text == "q"
