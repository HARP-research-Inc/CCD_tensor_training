"""
Setup script for building the C++ TnT POS tagger extension.

Usage:
    python setup.py build_ext --inplace    # Build in-place
    python setup.py install                # Install system-wide
    pip install .                          # Install via pip
"""

import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Try to import setuptools
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

# The main interface is through Pybind11Extension.
# * You can add cxx_std=14/17, and then build_ext can be told to use it
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.

# Note:
#   Sort input source files to ensure bit-for-bit reproducible builds
#   (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "fastpos_cpp",
        [
            "cpp/tnt_tagger.cpp",
            "cpp/bindings.cpp",
        ],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", '"dev"')],
        cxx_std=17,
        # Optimization flags
        extra_compile_args=[
            "-O3" if not sys.platform.startswith("win") else "/O2",
            "-march=native" if not sys.platform.startswith("win") else "",
            "-DNDEBUG",
        ] if "PYBIND11_DEBUG" not in os.environ else [
            "-g", "-O0"
        ],
    ),
]

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import pybind11
        print(f"Found pybind11 version: {pybind11.__version__}")
    except ImportError:
        print("Error: pybind11 is required but not found.")
        print("Install it with: pip install pybind11")
        sys.exit(1)
    
    # Check for C++ compiler
    try:
        if sys.platform.startswith("win"):
            # Windows: check for MSVC
            result = subprocess.run(["cl"], capture_output=True)
            if result.returncode != 0:
                print("Warning: MSVC compiler not found. Install Visual Studio Build Tools.")
        else:
            # Unix: check for g++ or clang++
            for compiler in ["g++", "clang++"]:
                result = subprocess.run([compiler, "--version"], capture_output=True)
                if result.returncode == 0:
                    print(f"Found C++ compiler: {compiler}")
                    break
            else:
                print("Warning: No C++ compiler found. Install g++ or clang++.")
    except Exception as e:
        print(f"Warning: Could not check for C++ compiler: {e}")

class CustomBuildExt(build_ext):
    """Custom build extension class with better error handling."""
    
    def build_extensions(self):
        # Check dependencies before building
        check_dependencies()
        
        # Set compiler flags based on platform
        compiler = self.compiler.compiler_type
        
        if compiler == 'msvc':
            # MSVC-specific flags
            for ext in self.extensions:
                ext.extra_compile_args = ['/O2', '/DNDEBUG']
        else:
            # GCC/Clang flags
            for ext in self.extensions:
                if not hasattr(ext, 'extra_compile_args'):
                    ext.extra_compile_args = []
                ext.extra_compile_args.extend([
                    '-O3', '-march=native', '-DNDEBUG', 
                    '-Wall', '-Wextra'
                ])
        
        # Call parent build method
        super().build_extensions()

def get_long_description():
    """Get long description from README if available."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    else:
        return """
        High-performance C++ implementation of TnT (Trigrams'n'Tags) POS tagger
        with Python bindings. Provides significant speedup over pure Python
        implementations while maintaining the same API.
        """

setup(
    name="fastpos_cpp",
    version="0.1.0",
    author="FastParse Team",
    author_email="",
    url="",
    description="High-performance C++ POS tagger for Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "nltk",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
) 