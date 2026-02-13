import sys
from pathlib import Path

from prime_rl.landscape.directions import build_orthogonalized_direction_paths


def test_build_orthogonalized_direction_paths_uses_cli_stem(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["landscape", "@", "configs/landscape/alpha_sort.toml"])

    delta_path, eta_path = build_orthogonalized_direction_paths(
        output_dir=Path("outputs/landscape"),
        orthogonalized_subdir=Path("directions"),
        orthogonalized_suffix="orth",
    )

    assert delta_path == Path("outputs/landscape/directions/alpha_sort_delta_orth.pt")
    assert eta_path == Path("outputs/landscape/directions/alpha_sort_eta_orth.pt")
