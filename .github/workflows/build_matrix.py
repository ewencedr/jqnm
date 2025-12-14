import json
import sys

# list of images/python combinations can be found here:
# https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json

# JAX requires Python 3.9+
python_version = ["3.9", "3.10", "3.11", "3.12"]

# OS versions for each Python version
ubuntu_version = ["ubuntu-22.04", "ubuntu-24.04", "ubuntu-24.04", "ubuntu-24.04"]
macos_version = ["macos-latest", "macos-latest", "macos-latest", "macos-latest"]
windows_version = [None, "windows-latest", "windows-latest", "windows-latest"]

# Package combinations to test (packages, min_python_version)
# JAX has specific numpy/scipy requirements
packages = [
    # Basic JAX installation (uses requirements.txt defaults)
    ("", "3.9"),
    # Test with specific numpy versions
    ("'numpy~=1.26.0'", "3.10"),
    ("'numpy>=2.0'", "3.10"),
]

configurations = []
for idx_pv, pv in enumerate(python_version):
    current_python_version = tuple(int(_) for _ in pv.split("."))

    for os in ubuntu_version, macos_version, windows_version:
        current_os = os[idx_pv]
        if current_os is None:
            continue

        for pk in packages:
            min_python_version = tuple(int(_) for _ in pk[1].split("."))
            if current_python_version < min_python_version:
                continue

            current_configuration = {
                "python-version": pv,
                "os": current_os,
                "packages": pk[0],
            }

            configurations += [current_configuration]

with open(sys.argv[1], "w") as f:
    json.dump(configurations, f)
