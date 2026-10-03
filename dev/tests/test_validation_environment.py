"""The preparation step validation runs before its configured commands
(DEVELOPMENT.md, Validation evidence).

A run worktree starts without build/ and without a Hub cache, so the evidence
agent's isolated HF_HOME would make every model load download again. The
target these tests drive links that path to this machine's shared Hugging Face
cache; none of them touches the real one (HF_HOME points at a temporary
directory).
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "dev/Makefile"


def run_target(build: Path, shared: Path | None):
    """`make test-hf-home` with its output under build, and HF_HOME set only
    when shared names one; returns the completed process."""
    environment = os.environ | {"BUILD": str(build)}
    if shared is None:
        environment.pop("HF_HOME", None)
    else:
        environment["HF_HOME"] = str(shared)
    return subprocess.run(
        ["make", "-f", str(MAKEFILE), "test-hf-home"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


class ValidationHFHomeTest(unittest.TestCase):
    def test_links_the_shared_home_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build, shared = root / "build", root / "shared"
            shared.mkdir()
            for _ in range(2):
                result = run_target(build, shared)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                link = build / "test-hf-home"
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), shared.resolve())
                # The link names the cache besides itself, never a copy of it.
                self.assertEqual(
                    sorted(path.name for path in build.iterdir()), ["test-hf-home"]
                )
                self.assertIn("test-hf-home", result.stdout)

    def test_defaults_to_the_home_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home, build = root / "home", root / "build"
            shared = home / ".cache/huggingface"
            shared.mkdir(parents=True)
            environment = os.environ | {"BUILD": str(build), "HOME": str(home)}
            environment.pop("HF_HOME", None)
            result = subprocess.run(
                ["make", "-f", str(MAKEFILE), "test-hf-home"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / "test-hf-home").resolve(), shared.resolve())

    def test_refuses_a_path_that_is_not_a_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, build = root / "shared", root / "build"
            shared.mkdir()
            (build / "test-hf-home").mkdir(parents=True)
            result = run_target(build, shared)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a symbolic link", result.stderr)
            self.assertFalse((build / "test-hf-home").is_symlink())

    def test_creates_a_shared_home_that_does_not_exist_yet(self):
        # A machine that has never downloaded a model still passes its
        # preparation step; the cache is created, not refused.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, build = root / "shared", root / "build"
            result = run_target(build, shared)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(shared.is_dir())
            self.assertEqual((build / "test-hf-home").resolve(), shared.resolve())

    def test_refuses_a_shared_home_that_is_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared"
            shared.write_text("not a directory")
            result = run_target(root / "build", shared)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a directory", result.stderr)
            self.assertFalse((root / "build/test-hf-home").is_symlink())

    def test_refuses_a_relative_HF_HOME(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_target(root / "build", Path("relative/cache"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("absolute path", result.stderr)

    def test_refuses_self_reference_without_creating_a_loop(self):
        # HF_HOME naming the link itself must refuse rather than link a path
        # to itself: the shared cache lives outside the worktree's build/.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            result = run_target(build, build / "test-hf-home")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a symbolic link", result.stderr)
            self.assertFalse((build / "test-hf-home").is_symlink())


if __name__ == "__main__":
    unittest.main()
