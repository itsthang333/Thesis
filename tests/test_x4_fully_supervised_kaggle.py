from pathlib import Path

import pytest

from project.run_x4_fully_supervised_student_kaggle import seed_from_script_name


@pytest.mark.parametrize("seed", (42, 43, 44))
def test_seed_is_bound_by_kernel_filename(seed: int) -> None:
    assert seed_from_script_name(Path(f"btxrd-x4-fully-student-seed{seed}.py")) == seed


@pytest.mark.parametrize(
    "name",
    ("btxrd-x4-fully-student.py", "btxrd-x4-fully-student-seed41.py", "seed442.py"),
)
def test_seed_filename_contract_fails_closed(name: str) -> None:
    with pytest.raises(ValueError):
        seed_from_script_name(Path(name))
