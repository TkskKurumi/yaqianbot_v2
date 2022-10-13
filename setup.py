from setuptools import setup, find_packages
pkgs = find_packages()
print(pkgs)
setup(
    name="yaqianbot",
    version="0.0.1",
    description="yaqianbot",
    author="TkskKurumi",
    author_email="zafkielkurumi@gmail.com",
    packages=pkgs
)