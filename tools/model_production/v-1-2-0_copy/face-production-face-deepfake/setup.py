import os
from distutils.util import convert_path

import setuptools

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + "/requirements.txt"
install_requires = []  # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]

if os.path.isfile(requirement_path):
    with open(requirement_path, "r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()

metadata = {}
ver_path = convert_path("face_deepfake/info.py")
with open(ver_path, "r", encoding="utf-8") as ver_file:
    exec(ver_file.read(), metadata)  # pylint: disable=exec-used

setuptools.setup(
    name="face_deepfake",
    version=metadata["__version__"],
    url=metadata["__homepage__"],
    author=metadata["__author__"],
    author_email=metadata["__author_email__"],
    packages=setuptools.find_packages(),
    keywords=[
        "face",
        "face_deepfake",
    ],
    include_package_data=True,
    install_requires=install_requires,
)
