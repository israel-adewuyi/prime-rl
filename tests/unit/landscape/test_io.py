import pytest

from prime_rl.landscape.io import append_result


def test_append_result_writes_header_and_row(tmp_path) -> None:
    output_path = tmp_path / "landscape.csv"

    append_result(output_path, {"alpha": 0.0, "beta": 0.0, "loss": 1.23})

    lines = output_path.read_text().strip().splitlines()
    assert lines[0] == "alpha,beta,loss"
    assert lines[1] == "0.0,0.0,1.23"


def test_append_result_rejects_header_mismatch(tmp_path) -> None:
    output_path = tmp_path / "landscape.csv"

    append_result(output_path, {"alpha": 0.0, "beta": 0.0})

    with pytest.raises(ValueError, match="Existing results file has different header"):
        append_result(output_path, {"alpha": 0.0, "loss": 1.0})
