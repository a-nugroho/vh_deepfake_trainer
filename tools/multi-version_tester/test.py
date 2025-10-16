import sys
import os
import importlib
import subprocess

def run_with_package_dir(pkg_name: str, pkg_dir: str, test_script: str):
    """
    Run a test script with a package loaded from a specific directory.

    Args:
        pkg_name (str): The package/module name (e.g. 'numpy')
        pkg_dir (str): Path to the package directory (containing __init__.py)
        test_script (str): Path to the test script to run
    """
    # Ensure absolute paths
    pkg_dir = os.path.abspath(pkg_dir)
    test_script = os.path.abspath(test_script)

    # Run Python in a subprocess with adjusted sys.path
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys, runpy; "
            f"sys.path.insert(0, r'{pkg_dir}'); "
            f"import {pkg_name}; "
            f"print('Using {pkg_name} from:', {pkg_name}.__file__); "
            f"runpy.run_path(r'{test_script}', run_name='__main__')"
        ),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    # Example: testing different local versions
    package_name = "mylib"  # replace with your package name
    test_script = "my_test.py"

    versions = {
        "v1": "/path/to/mylib_v1",
        "v2": "/path/to/mylib_v2",
        "v3": "/path/to/mylib_v3",
    }

    for label, path in versions.items():
        print(f"\n=== Running {package_name} from {label} ===")
        run_with_package_dir(package_name, path, test_script)