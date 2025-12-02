#!/usr/bin/env python3
"""Test what NVRTC supports in CUDA 13 for builtins and libcudacxx

Uses ctypes to call NVRTC directly - no pip packages needed.
"""

import ctypes
from ctypes import c_char_p, c_size_t, c_void_p, byref, POINTER, c_int

# Load NVRTC library
try:
    nvrtc = ctypes.CDLL("libnvrtc.so")
except OSError:
    print("ERROR: Could not load libnvrtc.so")
    print("Make sure CUDA is installed and in LD_LIBRARY_PATH")
    exit(1)

# Define NVRTC function signatures
nvrtc.nvrtcCreateProgram.argtypes = [POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)]
nvrtc.nvrtcCreateProgram.restype = c_int
nvrtc.nvrtcCompileProgram.argtypes = [c_void_p, c_int, POINTER(c_char_p)]
nvrtc.nvrtcCompileProgram.restype = c_int
nvrtc.nvrtcGetProgramLogSize.argtypes = [c_void_p, POINTER(c_size_t)]
nvrtc.nvrtcGetProgramLogSize.restype = c_int
nvrtc.nvrtcGetProgramLog.argtypes = [c_void_p, c_char_p]
nvrtc.nvrtcGetProgramLog.restype = c_int
nvrtc.nvrtcDestroyProgram.argtypes = [POINTER(c_void_p)]
nvrtc.nvrtcDestroyProgram.restype = c_int

NVRTC_SUCCESS = 0


def get_cuda_include_paths():
    """Find CUDA include paths for NVRTC"""
    import os
    from pathlib import Path

    paths = []
    cuda_root = Path("/usr/local/cuda")

    # Check for aarch64 vs x86_64
    import platform
    arch = platform.machine()
    if arch == "aarch64":
        target = "sbsa-linux"
    else:
        target = "x86_64-linux"

    # CCCL headers (CUDA 13+ location)
    cccl = cuda_root / f"targets/{target}/include/cccl"
    if cccl.exists():
        paths.append(cccl)

    # Target-specific include
    target_inc = cuda_root / f"targets/{target}/include"
    if target_inc.exists():
        paths.append(target_inc)

    # Generic include
    generic_inc = cuda_root / "include"
    if generic_inc.exists():
        paths.append(generic_inc)

    return paths


def test_nvrtc_compile(name: str, code: str, include_cuda_headers: bool = False):
    """Try to compile code with NVRTC and report success/failure"""
    opts = [b"--std=c++17", b"-arch=compute_120"]

    if include_cuda_headers:
        for path in get_cuda_include_paths():
            opts.append(f"-I{path}".encode())

    opts_array = (c_char_p * len(opts))(*opts)

    prog = c_void_p()
    err = nvrtc.nvrtcCreateProgram(byref(prog), code.encode(), b"test.cu", 0, None, None)
    if err != NVRTC_SUCCESS:
        print(f"✗ {name}: FAILED (create error {err})")
        return False

    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts_array)

    # Get log
    log_size = c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    log_str = log.value.decode().strip()

    nvrtc.nvrtcDestroyProgram(byref(prog))

    if err == NVRTC_SUCCESS:
        print(f"✓ {name}: OK")
        return True
    else:
        print(f"✗ {name}: FAILED")
        if log_str:
            for line in log_str.split('\n')[:5]:
                print(f"  {line}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NVRTC Builtin/Feature Tests for CUDA 13 / sm_120")
    print("=" * 60)

    # Show include paths
    paths = get_cuda_include_paths()
    print(f"Include paths found: {len(paths)}")
    for p in paths:
        print(f"  -I{p}")
    print()

    # Test 7: cumm's nvrtc_std.h (uses libcudacxx for CUDA 13+)
    print()
    print("=" * 60)
    print("Testing cumm nvrtc_std.h (CUDA 13+ uses libcudacxx)")
    print("=" * 60)

    # Get cumm include path (assuming we're running from repo root)
    import os
    cumm_include = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include"))
    print(f"cumm include path: {cumm_include}")

    def test_nvrtc_compile_cumm(name: str, code: str):
        """Test with cumm include path"""
        opts = [b"--std=c++17", b"-arch=compute_120"]
        # Add CUDA include paths first
        for path in get_cuda_include_paths():
            opts.append(f"-I{path}".encode())
        # Add cumm include path
        opts.append(f"-I{cumm_include}".encode())
        opts_array = (c_char_p * len(opts))(*opts)

        prog = c_void_p()
        err = nvrtc.nvrtcCreateProgram(byref(prog), code.encode(), b"test.cu", 0, None, None)
        if err != NVRTC_SUCCESS:
            print(f"✗ {name}: FAILED (create error {err})")
            return False

        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts_array)

        # Get log
        log_size = c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        log_str = log.value.decode().strip()

        nvrtc.nvrtcDestroyProgram(byref(prog))

        if err == NVRTC_SUCCESS:
            print(f"✓ {name}: OK")
            return True
        else:
            print(f"✗ {name}: FAILED")
            if log_str:
                for line in log_str.split('\n')[:10]:
                    print(f"  {line}")
            return False

    # Test cumm's nvrtc_std.h - numeric_limits
    test_nvrtc_compile_cumm("nvrtc_std.h - numeric_limits<float>::infinity()", """
#include <tensorview/core/nvrtc_std.h>
__device__ float get_inf() {
    return std::numeric_limits<float>::infinity();
}
""")

    test_nvrtc_compile_cumm("nvrtc_std.h - numeric_limits<float>::quiet_NaN()", """
#include <tensorview/core/nvrtc_std.h>
__device__ float get_nan() {
    return std::numeric_limits<float>::quiet_NaN();
}
""")

    test_nvrtc_compile_cumm("nvrtc_std.h - numeric_limits<int>::max()", """
#include <tensorview/core/nvrtc_std.h>
__device__ int get_max() {
    return std::numeric_limits<int>::max();
}
""")

    # Test cumm's nvrtc_std.h - type_traits
    test_nvrtc_compile_cumm("nvrtc_std.h - is_integral", """
#include <tensorview/core/nvrtc_std.h>
__device__ void test() {
    static_assert(std::is_integral<int>::value, "");
}
""")

    test_nvrtc_compile_cumm("nvrtc_std.h - is_same", """
#include <tensorview/core/nvrtc_std.h>
__device__ void test() {
    static_assert(std::is_same<int, int>::value, "");
}
""")

    # Test cumm's nvrtc_std.h - tuple
    test_nvrtc_compile_cumm("nvrtc_std.h - tuple creation", """
#include <tensorview/core/nvrtc_std.h>
__device__ void test() {
    std::tuple<int, float> t(1, 2.0f);
}
""")

    test_nvrtc_compile_cumm("nvrtc_std.h - std::get", """
#include <tensorview/core/nvrtc_std.h>
__device__ int test() {
    std::tuple<int, float> t(42, 3.14f);
    return std::get<0>(t);
}
""")

    # Test cumm's nvrtc_std.h - array
    test_nvrtc_compile_cumm("nvrtc_std.h - std::array", """
#include <tensorview/core/nvrtc_std.h>
__device__ int test() {
    std::array<int, 3> arr = {1, 2, 3};
    return arr[0];
}
""")

    print()
    print("=" * 60)
    print("Summary:")
    print("  CUDA 13+ uses libcudacxx via nvrtc_std.h")
    print("  All std:: types come from cuda::std::")
    print()
    print("  If tests pass: nvrtc_std.h is working correctly!")
    print("  If tests fail: Check CCCL include paths")
