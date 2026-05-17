from app.core.cuda_engine import CUDAEngine, KernelProfile


def _make_engine() -> CUDAEngine:
    engine = CUDAEngine.__new__(CUDAEngine)
    engine.nvcc_path = "nvcc"
    engine.cuda_std = "c++17"
    engine.arch = "sm_80"
    engine.max_retries = 3
    engine._nvcc_available = False
    from pathlib import Path
    import tempfile
    engine.kernels_dir = Path(tempfile.mkdtemp())
    return engine


def test_classify_matmul():
    engine = _make_engine()
    assert engine._classify_request("write a matrix multiplication kernel") == "matmul"


def test_classify_attention():
    engine = _make_engine()
    assert engine._classify_request("flash attention CUDA kernel") == "attention"


def test_classify_reduction():
    engine = _make_engine()
    assert engine._classify_request("parallel sum reduction") == "reduction"


def test_static_analysis_scores_good_kernel():
    engine = _make_engine()
    good_kernel = """
    __global__ void good(const float* __restrict__ a, float* __restrict__ b) {
        __shared__ float s[32];
        s[threadIdx.x] = a[blockIdx.x * 32 + threadIdx.x];
        __syncthreads();
        b[threadIdx.x] = __shfl_down_sync(0xffffffff, s[threadIdx.x], 1);
        cudaGetLastError();
    }
    """
    profile = engine._static_analysis(good_kernel)
    assert profile.compiled is True
    assert profile.occupancy_pct > 50.0


def test_static_analysis_warns_on_bad_kernel():
    engine = _make_engine()
    bad_kernel = "__global__ void bad(float* a) { a[0] = 1.0f; }"
    profile = engine._static_analysis(bad_kernel)
    assert len(profile.warnings) > 0


def test_optimise_adds_restrict():
    engine = _make_engine()
    kernel = "__global__ void k(const float* a, float* b) { b[0] = a[0]; }"
    profile = KernelProfile(compiled=True, occupancy_pct=80.0)
    _optimised, applied = engine._optimise(kernel, profile, "elementwise")
    assert any("restrict" in opt.lower() for opt in applied)


def test_extract_cuda_code():
    engine = _make_engine()
    text = "Here is the kernel:\n```cuda\n__global__ void k() {}\n```"
    result = engine._extract_cuda_code(text)
    assert result == "__global__ void k() {}"
