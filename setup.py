import os
import re
from setuptools import setup, find_packages

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PACKAGE_DIR, "requirements.txt")) as f:
    install_requires = f.read().splitlines()
    install_requires = [l for l in install_requires if not l.startswith("#")]

tests_requirements_file = os.path.join(PACKAGE_DIR, "tests/requirements.txt")
# in the release tar.gz file, there will be no tests_requirements_file
if os.path.isfile(tests_requirements_file):
    with open(tests_requirements_file) as f:
        tests_require = f.read().splitlines()
        tests_require = [l for l in tests_require if not l.startswith("#")]
else:
    tests_require = []

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

VERSIONFILE = os.path.join(PACKAGE_DIR, "kvdbclient/__version__.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name="kvdbclient",
    description="Client for key value databases",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT License",
    version=version,
    author="Akhilesh Halageri",
    author_email="halageri@princeton.edu",
    packages=find_packages(),
    url="https://github.com/seung-lab/kvdbclient",
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3",
)
