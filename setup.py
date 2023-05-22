from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["torch>=2.0.1", "torchvision>=0.15.2", "urllib3==1.26.6"]

setup(
    name="medseglib",
    version="1.0.0",
    author="Ilya Zarudin",
    author_email="zarilja@yandex.ru",
    description="A package to segmentation tumor on CT and MRI",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)