import subprocess
import sys


def test_build_subject_index_cli_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_subject_index.py", "--config", "configs/icdm2026.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Saved subject index to outputs/icdm2026/subject_index.csv" in result.stdout
    assert "Subjects: 248" in result.stdout
    assert "Splits: train=173, val=37, test=38" in result.stdout
    assert "Groups: CN=92, SCD=56, MCI=70, AD=30" in result.stdout

