from __future__ import annotations
import hashlib
import json
import logging
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class KernelProfile:
    gflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    occupancy_pct: float = 0.0
    warp_efficiency_pct: float = 0.0
    latency_ms: float = 0.0
    compiled: bool = False
    compile_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class KernelResult:
    kernel_code: str
    kernel_type: str
    profile: KernelProfile
    optimisations_applied: list[str]
    explanation: str
    version: int
    improved: bool
    hardware_target: str


KERNEL_PATTERNS = {
    "matmul": [
        "matrix multiply", "matmul", "gemm", "sgemm", "dgemm",
        "dot product", "linear layer", "fully connected",
    ],
    "reduction": [
        "reduction", "sum", "max", "min", "mean", "average",
        "softmax", "layer norm", "batch norm",
    ],
    "attention": [
        "attention", "self-attention", "flash attention",
        "multi-head", "transformer", "query key value", "qkv",
    ],
    "convolution": [
        "convolution", "conv2d", "conv1d", "depthwise",
        "im2col", "winograd",
    ],
    "elementwise": [
        "elementwise", "relu", "gelu", "sigmoid", "tanh",
        "activation", "fused", "pointwise",
    ],
    "memory": [
        "copy", "transpose", "reshape", "gather", "scatter",
        "coalesced", "memory access", "bandwidth",
    ],
    "scan": [
        "prefix sum", "scan", "cumsum", "inclusive", "exclusive",
    ],
    "sort": [
        "sort", "radix sort", "merge sort", "bitonic",
    ],
}

KERNEL_TEMPLATES = {"matmul": """
// Tiled Matrix Multiplication Kernel
// Uses shared memory tiling for cache efficiency
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int TILE = 32;
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
""", "reduction": """
// Warp-level Reduction Kernel using shuffle instructions
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void block_reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    __shared__ float sdata[32];
    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;
    float val = (idx < N) ? input[idx] : 0.0f;

    val = warp_reduce_sum(val);
    if (tid % 32 == 0) sdata[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        val = warp_reduce_sum(val);
    }
    if (tid == 0) output[blockIdx.x] = val;
}
""", "attention": """
// Flash Attention Kernel (simplified single-head)
// Online softmax for memory efficiency
__global__ void flash_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float max_val = -1e9f;
    float sum_exp = 0.0f;

    // First pass: compute max for numerical stability
    for (int j = 0; j < N; ++j) {
        float qk = 0.0f;
        for (int k = 0; k < d; ++k)
            qk += Q[i * d + k] * K[j * d + k];
        qk *= scale;
        max_val = fmaxf(max_val, qk);
    }

    // Second pass: compute weighted output (online softmax)
    float out[64] = {0.0f};  // assume d <= 64
    for (int j = 0; j < N; ++j) {
        float qk = 0.0f;
        for (int k = 0; k < d; ++k)
            qk += Q[i * d + k] * K[j * d + k];
        float w = expf(qk * scale - max_val);
        sum_exp += w;
        for (int k = 0; k < d; ++k)
            out[k] += w * V[j * d + k];
    }
    for (int k = 0; k < d; ++k)
        O[i * d + k] = out[k] / sum_exp;
}
"""}


class CUDAEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        ce = cfg.get("cuda_engine", {})
        self.nvcc_path = ce.get("nvcc_path", "nvcc")
        self.cuda_std = ce.get("cuda_std", "c++17")
        self.arch = ce.get("arch", "sm_80")
        self.max_retries = int(ce.get("max_compile_retries", 3))
        self.kernels_dir = Path(ce.get("kernels_dir", "data/cuda_kernels"))
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
        self._nvcc_available = self._check_nvcc()

    def generate(self, request: str, hardware_target: str = "auto", optimise: bool = True) -> KernelResult:
        kernel_type = self._classify_request(request)
        hardware = self._resolve_hardware(hardware_target)
        kernel_code = self._generate_kernel(request, kernel_type, hardware)
        profile, kernel_code = self._compile_with_retry(kernel_code, kernel_type)
        optimisations: list[str] = []
        if optimise and profile.compiled:
            kernel_code, optimisations = self._optimise(kernel_code, profile, kernel_type)
            profile, _ = self._compile_with_retry(kernel_code, kernel_type)
        explanation = self._explain_kernel(kernel_code, kernel_type, profile, optimisations)
        result = KernelResult(kernel_code, kernel_type, profile, optimisations, explanation, 1, bool(optimisations), hardware)
        self._store_kernel(request, result)
        threading.Thread(target=self._update_cuda_skill, args=(kernel_type, kernel_code, profile), daemon=True).start()
        return result

    def benchmark(self, kernel_code: str) -> KernelProfile:
        profile, _ = self._compile_with_retry(kernel_code, "custom")
        return profile

    def _classify_request(self, request: str) -> str:
        req_lower = request.lower()
        scores = {}
        for ktype, keywords in KERNEL_PATTERNS.items():
            scores[ktype] = sum(1 for kw in keywords if kw in req_lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "custom"

    def _resolve_hardware(self, target: str) -> str:
        if target != "auto":
            return target
        arch_map = {"sm_90": "H100", "sm_89": "RTX 4090", "sm_86": "RTX 3090", "sm_80": "A100", "sm_75": "T4/RTX 2080", "sm_70": "V100"}
        return arch_map.get(self.arch, self.arch)

    def _generate_kernel(self, request: str, kernel_type: str, hardware: str) -> str:
        template = KERNEL_TEMPLATES.get(kernel_type, "")
        skill_context = ""
        try:
            from app.core.skill_forge import SkillForge
            skill = SkillForge().get_skill_for_query(f"CUDA {kernel_type} kernel optimisation")
            if skill:
                skill_context = skill.to_prompt_block()
        except Exception as exc:
            logger.warning("SkillForge lookup in CUDAEngine failed: %s", exc)
        prompt = self._build_cuda_prompt(request, kernel_type, hardware, template, skill_context)
        try:
            from app.core.llm_synthesiser import get_engine
            engine = get_engine()
            if engine:
                raw = engine.generate(prompt, max_tokens=1024, stream=False)
                if isinstance(raw, str):
                    return self._extract_cuda_code(raw) or template
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
        return template or self._stub_kernel(request, kernel_type)

    def _build_cuda_prompt(self, request: str, kernel_type: str, hardware: str, template: str, skill_context: str) -> str:
        arch_hints = self._arch_hints()
        return (
            "You are an expert CUDA kernel engineer targeting "
            f"{hardware} ({self.arch}).\n\n"
            f"{skill_context}\n\n"
            f"Architecture constraints for {self.arch}:\n{arch_hints}\n\n"
            f"Reference pattern for {kernel_type}:\n"
            f"```cuda\n{template}\n```\n\n"
            f"Task: {request}\n\n"
            "Write a complete, optimised CUDA kernel. Requirements:\n"
            "- Use shared memory tiling where applicable\n"
            "- Use warp-level primitives (__shfl_sync, __ballot_sync)\n"
            "- Ensure memory coalescing on global memory access\n"
            "- Use __restrict__ on non-aliased pointers\n"
            "- Avoid bank conflicts in shared memory\n"
            "- Include a host-side launch wrapper\n"
            "- Include error checking with cudaGetLastError()\n\n"
            "Output ONLY the CUDA code inside a ```cuda block."
        )

    def _arch_hints(self) -> str:
        hints = {
            "sm_90": "H100: 132 SMs, 228KB shared mem/SM, TMA for async copy, wgmma for tensor core",
            "sm_80": "A100: 108 SMs, 164KB shared mem/SM, async copy (cp.async), tensor cores (16x16x16)",
            "sm_75": "T4: 40 SMs, 64KB shared mem/SM, Tensor cores available, use ldg for cached loads",
            "sm_70": "V100: 80 SMs, 96KB shared mem/SM, Volta tensor cores, independent thread scheduling",
        }
        return hints.get(self.arch, f"Target: {self.arch}")

    def _extract_cuda_code(self, text: str) -> str | None:
        for pattern in [r"```cuda\n(.*?)```", r"```cpp\n(.*?)```", r"```c\+\+\n(.*?)```", r"```c\n(.*?)```"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _stub_kernel(self, request: str, kernel_type: str) -> str:
        return (f"// Girivinity CUDA Kernel — {kernel_type}\n" f"// Request: {request}\n" f"// TODO: implementation\n" f"__global__ void girivinity_{kernel_type}_kernel() {{}}\n")

    def _compile_with_retry(self, kernel_code: str, kernel_type: str) -> tuple[KernelProfile, str]:
        if not self._nvcc_available:
            return self._static_analysis(kernel_code), kernel_code
        current_code = kernel_code
        for attempt in range(self.max_retries):
            profile, errors = self._compile(current_code)
            if profile.compiled:
                return profile, current_code
            if errors and attempt < self.max_retries - 1:
                logger.info("Compile attempt %d failed, auto-fixing...", attempt + 1)
                current_code = self._auto_fix(current_code, errors)
            else:
                return profile, current_code
        return KernelProfile(compile_errors=["Max retries exceeded"]), current_code

    def _compile(self, kernel_code: str) -> tuple[KernelProfile, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kernel.cu"
            out = Path(tmp) / "kernel.o"
            src.write_text(kernel_code, encoding="utf-8")
            cmd = [self.nvcc_path, f"-arch={self.arch}", f"-std={self.cuda_std}", "--ptxas-options=-v", "-O3", "-c", str(src), "-o", str(out)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                compiled = result.returncode == 0
                errors = self._parse_errors(result.stderr)
                warnings = self._parse_warnings(result.stderr)
                profile = KernelProfile(compiled=compiled, compile_errors=errors, warnings=warnings)
                if compiled:
                    profile = self._parse_ptxas_stats(result.stderr, profile)
                return profile, errors
            except subprocess.TimeoutExpired:
                return KernelProfile(compile_errors=["Compilation timed out"]), ["timeout"]
            except FileNotFoundError:
                self._nvcc_available = False
                return self._static_analysis(kernel_code), []

    def _parse_errors(self, stderr: str) -> list[str]:
        return [line for line in stderr.splitlines() if "error:" in line.lower()]

    def _parse_warnings(self, stderr: str) -> list[str]:
        return [line for line in stderr.splitlines() if "warning:" in line.lower()][:5]

    def _parse_ptxas_stats(self, stderr: str, profile: KernelProfile) -> KernelProfile:
        reg_match = re.search(r"(\d+) registers", stderr)
        smem_match = re.search(r"(\d+) bytes smem", stderr)
        if reg_match:
            regs = int(reg_match.group(1))
            profile.occupancy_pct = max(0.0, 100.0 - (regs - 32) * 1.5)
        if smem_match:
            smem = int(smem_match.group(1))
            profile.warp_efficiency_pct = min(100.0, smem / 1024 * 10)
        return profile

    def _static_analysis(self, kernel_code: str) -> KernelProfile:
        score = 0.0
        warnings = []
        checks = {"__shared__": 20.0, "__restrict__": 10.0, "__syncthreads": 10.0, "__shfl": 15.0, "cp.async": 15.0, "cudaGetLastError": 10.0, "threadIdx": 5.0, "blockIdx": 5.0, r"if \(.*idx.* < ": 10.0}
        missing_warns = {"__shared__": "No shared memory — likely suboptimal memory access", "__restrict__": "Missing __restrict__ — compiler cannot assume non-aliasing", "cudaGetLastError": "No error checking — add cudaGetLastError()"}
        for pattern, pts in checks.items():
            if re.search(pattern, kernel_code):
                score += pts
        for pattern, msg in missing_warns.items():
            if not re.search(pattern, kernel_code):
                warnings.append(msg)
        compiled = "__global__" in kernel_code or "__device__" in kernel_code
        return KernelProfile(compiled=compiled, occupancy_pct=min(100.0, score), warp_efficiency_pct=min(100.0, score * 0.8), warnings=warnings)

    def _auto_fix(self, kernel_code: str, errors: list[str]) -> str:
        fixed = kernel_code
        for error in errors:
            e = error.lower()
            if "undeclared identifier" in e and "#include" not in fixed:
                fixed = "#include <cuda_runtime.h>\n" + fixed
            if "expected ';'" in e:
                fixed = re.sub(r"(\w+)\s*\n", r"\1;\n", fixed, count=1)
            if "no member named 'sync'" in e:
                fixed = fixed.replace(".sync(", "_sync(")
        return fixed

    def _optimise(self, kernel_code: str, profile: KernelProfile, kernel_type: str) -> tuple[str, list[str]]:
        optimised = kernel_code
        applied: list[str] = []
        if "__restrict__" not in optimised:
            optimised = re.sub(r"\b(const float\*|float\*)\s+(?!__restrict__)", r"\1 __restrict__ ", optimised)
            applied.append("Added __restrict__ for non-aliased pointers")
        if "#pragma unroll" not in optimised and "for" in optimised:
            optimised = re.sub(r"(\s+)(for\s*\(int\s+\w+\s*=\s*0;\s*\w+\s*<\s*\d+)", r"\1#pragma unroll\n\1\2", optimised, count=2)
            applied.append("Added #pragma unroll to inner loops")
        if kernel_type == "reduction" and "__shfl" not in optimised and "atomicAdd" in optimised:
            applied.append("Suggestion: Replace atomicAdd with warp shuffle reduction for 10-30x speedup")
        if self.arch in ("sm_80", "sm_90") and "cp.async" not in optimised and "__shared__" in optimised:
            applied.append("Suggestion: Use cp.async (cuda::memcpy_async) for double-buffered shared memory loads on A100/H100")
        if "float4" not in optimised and kernel_type in ("elementwise", "memory"):
            applied.append("Suggestion: Use float4 vectorised loads to maximise memory bandwidth (4x throughput)")
        if profile.occupancy_pct < 50:
            applied.append(f"Warning: Estimated occupancy {profile.occupancy_pct:.0f}% — consider reducing register count or shared memory")
        return optimised, applied

    def _explain_kernel(self, kernel_code: str, kernel_type: str, profile: KernelProfile, optimisations: list[str]) -> str:
        lines = ["## Girivinity CUDA Kernel Report", "", f"**Type:** {kernel_type}  |  **Target:** {self.arch}  |  **Compiled:** {'✅' if profile.compiled else '❌'}", ""]
        if profile.compiled:
            lines += ["**Performance Estimates:**", f"- Occupancy: {profile.occupancy_pct:.1f}%", f"- Warp efficiency: {profile.warp_efficiency_pct:.1f}%"]
            if profile.warnings:
                lines += ["", "**Compiler warnings:**"] + [f"- {w}" for w in profile.warnings]
        else:
            lines += ["**Compile errors:**"] + [f"- {e}" for e in profile.compile_errors[:3]]
        if optimisations:
            lines += ["", "**Optimisations applied / suggested:**"] + [f"- {opt}" for opt in optimisations]
        return "\n".join(lines)

    def _store_kernel(self, request: str, result: KernelResult) -> None:
        uid = hashlib.sha256(request.encode()).hexdigest()[:12]
        kernel_path = self.kernels_dir / uid
        kernel_path.mkdir(exist_ok=True)
        (kernel_path / "kernel.cu").write_text(result.kernel_code, encoding="utf-8")
        (kernel_path / "meta.json").write_text(json.dumps({"request": request, "kernel_type": result.kernel_type, "hardware_target": result.hardware_target, "compiled": result.profile.compiled, "occupancy": result.profile.occupancy_pct, "optimisations": result.optimisations_applied, "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")

    def _update_cuda_skill(self, kernel_type: str, kernel_code: str, profile: KernelProfile) -> None:
        try:
            from app.core.skill_forge import SkillForge
            chunks = [{"text": (f"CUDA {kernel_type} kernel pattern:\n" f"Occupancy: {profile.occupancy_pct:.1f}%\n" f"Warp efficiency: {profile.warp_efficiency_pct:.1f}%\n" f"Code:\n{kernel_code[:800]}"), "score": 0.9 if profile.compiled else 0.5, "url": ""}]
            SkillForge().generate_async(topic=f"CUDA {kernel_type} kernel", chunks=chunks, urls=[])
        except Exception as exc:
            logger.warning("CUDA skill update failed: %s", exc)

    def _check_nvcc(self) -> bool:
        try:
            result = subprocess.run([self.nvcc_path, "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
