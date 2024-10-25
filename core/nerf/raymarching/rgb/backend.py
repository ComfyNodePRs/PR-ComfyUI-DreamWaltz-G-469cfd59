import os
import sys
from torch.utils.cpp_extension import load


def setup_windows_build_env():
    """Setup Windows build environment by ensuring required SDK headers and libraries are available"""
    import glob

    # Force use of stdlib setuptools distutils
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    # Find Windows SDK path
    def find_windows_sdk_include():
        sdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include"
        if os.path.exists(sdk_include):
            versions = [
                d
                for d in os.listdir(sdk_include)
                if os.path.isdir(os.path.join(sdk_include, d))
            ]
            if versions:
                latest = sorted(versions)[-1]
                return os.path.join(sdk_include, latest)
        return None

    def find_windows_sdk_lib():
        sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib"
        if os.path.exists(sdk_lib):
            versions = [
                d
                for d in os.listdir(sdk_lib)
                if os.path.isdir(os.path.join(sdk_lib, d))
            ]
            if versions:
                latest = sorted(versions)[-1]
                return os.path.join(sdk_lib, latest)
        return None

    # Find Visual Studio installation
    def find_vs_path():
        for program_files in [r"C:\Program Files (x86)", r"C:\Program Files"]:
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        r"%s\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*"
                        % (program_files, edition)
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]
        return None

    if os.name == "nt":  # Windows only
        vs_path = find_vs_path()
        sdk_include_path = find_windows_sdk_include()
        sdk_lib_path = find_windows_sdk_lib()

        if not vs_path:
            raise RuntimeError("Could not find Visual Studio installation")
        if not sdk_include_path:
            raise RuntimeError("Could not find Windows SDK Include directory")
        if not sdk_lib_path:
            raise RuntimeError("Could not find Windows SDK Lib directory")

        # Add Visual Studio and Windows SDK include paths
        include_paths = [
            os.path.join(vs_path, "include"),
            os.path.join(sdk_include_path, "ucrt"),
            os.path.join(sdk_include_path, "shared"),
            os.path.join(sdk_include_path, "um"),
        ]

        # Add library paths
        lib_paths = [
            os.path.join(vs_path, "lib", "x64"),
            os.path.join(sdk_lib_path, "ucrt", "x64"),
            os.path.join(sdk_lib_path, "um", "x64"),
        ]

        # Set environment variables
        os.environ["INCLUDE"] = ";".join(
            include_paths
            + (
                os.environ.get("INCLUDE", "").split(";")
                if "INCLUDE" in os.environ
                else []
            )
        )
        os.environ["LIB"] = ";".join(
            lib_paths
            + (os.environ.get("LIB", "").split(";") if "LIB" in os.environ else [])
        )


# Set source directory
_src_path = os.path.dirname(os.path.abspath(__file__))

# Common compilation flags
cxx_args = "c++17"

# NVCC flags for CUDA
nvcc_flags = [
    "-O3",
    f"-std={cxx_args}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-allow-unsupported-compiler",
]

# Platform-specific flags
if os.name == "posix":
    c_flags = ["-O3", f"-std={cxx_args}"]
else:  # Windows
    setup_windows_build_env()
    c_flags = ["/O2", f"/std:{cxx_args}"]

# Ensure build directory exists
build_directory = os.path.join(_src_path, "build")
os.makedirs(build_directory, exist_ok=True)

# Clean previous builds
if os.path.exists(os.path.join(build_directory, "_raymarchingrgb.pyd")):
    try:
        os.remove(os.path.join(build_directory, "_raymarchingrgb.pyd"))
    except:
        pass

# Compile and load the extension
try:
    _backend = load(
        name="_raymarchingrgb",
        build_directory=build_directory,
        extra_cflags=c_flags,
        extra_cuda_cflags=nvcc_flags,
        sources=[
            os.path.join(_src_path, "src", f)
            for f in [
                "raymarching.cu",
                "bindings.cpp",
            ]
        ],
        verbose=True,
    )
except Exception as e:
    print(f"Failed to build _raymarchingrgb extension: {str(e)}")
    raise

__all__ = ["_backend"]
