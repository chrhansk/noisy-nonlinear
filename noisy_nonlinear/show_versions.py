from importlib.metadata import version
import sys
import platform

if __name__ == "__main__":
    print("Python version: {0}".format(platform.python_version()))
    for package in ['numpy', 'scipy', 'ipopt']:
        print("Version of {0}: {1}".format(package, version(package)))
