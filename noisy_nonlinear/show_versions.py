from importlib.metadata import version
import platform

if __name__ == "__main__":
    print("Python version: {0}".format(platform.python_version()))
    for package in ['numpy', 'scipy', 'cyipopt']:
        print("Version of {0}: {1}".format(package, version(package)))
