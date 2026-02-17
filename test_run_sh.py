import os
import stat
import unittest
import subprocess
import tempfile
import shutil


SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "run.sh")


class TestRunShScript(unittest.TestCase):

    def test_script_exists(self):
        self.assertTrue(os.path.isfile(SCRIPT_PATH))

    def test_script_is_executable(self):
        mode = os.stat(SCRIPT_PATH).st_mode
        self.assertTrue(mode & stat.S_IXUSR)

    def test_script_has_shebang(self):
        with open(SCRIPT_PATH) as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("#!/"))

    def test_script_uses_strict_mode(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("set -euo pipefail", content)

    def test_find_python_returns_interpreter(self):
        result = subprocess.run(
            ["bash", "-c", f"source {SCRIPT_PATH} 2>/dev/null; find_python"],
            capture_output=True, text=True,
            env={**os.environ, "SKIP_MAIN": "1"},
        )
        # find_python should output a python command name
        out = result.stdout.strip()
        self.assertIn("python", out)

    def test_check_python_version_accepts_current(self):
        result = subprocess.run(
            ["bash", "-c",
             f"source {SCRIPT_PATH} 2>/dev/null; "
             f"check_python_version $(find_python)"],
            capture_output=True, text=True,
            env={**os.environ, "SKIP_MAIN": "1"},
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Found Python", result.stderr + result.stdout)

    def test_setup_venv_creates_directory(self):
        tmpdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                ["bash", "-c",
                 f"cd {tmpdir} && VENV_DIR=.venv && "
                 f"source {SCRIPT_PATH} 2>/dev/null; "
                 f"setup_venv $(find_python)"],
                capture_output=True, text=True,
                env={**os.environ, "SKIP_MAIN": "1"},
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, ".venv")))
        finally:
            shutil.rmtree(tmpdir)

    def test_setup_venv_reuses_existing(self):
        tmpdir = tempfile.mkdtemp()
        try:
            venv_path = os.path.join(tmpdir, ".venv")
            os.makedirs(venv_path)
            # Create a minimal bin/activate so the source command doesn't fail
            bin_dir = os.path.join(venv_path, "bin")
            os.makedirs(bin_dir)
            with open(os.path.join(bin_dir, "activate"), "w") as f:
                f.write("# dummy activate\n")
            result = subprocess.run(
                ["bash", "-c",
                 f"cd {tmpdir} && VENV_DIR=.venv && "
                 f"source {SCRIPT_PATH} 2>/dev/null; "
                 f"setup_venv $(find_python)"],
                capture_output=True, text=True,
                env={**os.environ, "SKIP_MAIN": "1"},
            )
            self.assertEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("already exists", combined)
        finally:
            shutil.rmtree(tmpdir)

    def test_install_deps_fails_without_requirements(self):
        tmpdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                ["bash", "-c",
                 f"cd {tmpdir} && REQUIREMENTS=requirements.txt && "
                 f"source {SCRIPT_PATH} 2>/dev/null; "
                 f"install_deps"],
                capture_output=True, text=True,
                env={**os.environ, "SKIP_MAIN": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            combined = result.stdout + result.stderr
            self.assertIn("not found", combined)
        finally:
            shutil.rmtree(tmpdir)

    def test_script_references_main_py(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("main.py", content)

    def test_script_installs_from_pytorch_index(self):
        with open(SCRIPT_PATH) as f:
            content = f.read()
        self.assertIn("download.pytorch.org/whl/cpu", content)


if __name__ == "__main__":
    unittest.main()
