#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <assert.h>
#include <cstring>
#include <memory>
#include <limits>

#include "gdr_context.hpp"
#include "tensor_utils.hpp"
#include "cuda_utils.h"

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

#if KERNEL_USE_GDRCOPY == 1
constexpr int MAX_GATHER_TOKENS = 1024 * 16;

// Global GDR contexts for source pointers (shared with permute.cu pattern)
gdr_context_t gather_and_sum_src_ptrs_gdr = nullptr;
gdr_context_t gather_and_sum_src_ptrs_gdr_alt = nullptr;

gdr_context_t get_gather_and_sum_src_ptrs_gdr() {
    static int enter_count = 0;
    if (gather_and_sum_src_ptrs_gdr == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_and_sum_src_ptrs_gdr = std::make_shared<GdrContext>(src_tensor);
    }
    if (gather_and_sum_src_ptrs_gdr_alt == nullptr) {
        auto src_tensor = get_cuda_aligned_tensor(MAX_GATHER_TOKENS, torch::kUInt64);
        gather_and_sum_src_ptrs_gdr_alt = std::make_shared<GdrContext>(src_tensor);
    }
    enter_count = (enter_count + 1) & 1;
    if (enter_count & 1) {
        return gather_and_sum_src_ptrs_gdr;
    } else {
        return gather_and_sum_src_ptrs_gdr_alt;
    }
}
#endif

// Fused gather and sum kernel
// For each output token (sequence), gather topk tokens and sum them
// Each block handles one output token and one chunk of hidden_size
template <class T, int CHUNK_SIZE>
__global__ void gather_and_sum_tokens_kernel(
    T *d_out, 
    uintptr_t *d_in_ptr, 
    const int n,
    const int topk,
    const int hidden_size
) {
    int output_token_id = blockIdx.x;  // Which of the n output tokens
    int chunk_id = blockIdx.y;         // Which chunk of hidden_size
    
    if (output_token_id >= n) return;
    
    bf16 *out = reinterpret_cast<bf16 *>(d_out + int64_t(output_token_id) * hidden_size);
    int chunk_base = chunk_id * CHUNK_SIZE;
    int first_token_idx = output_token_id * topk;
    // Accumulate in FP32 and round once. Repeated BF16 additions lose small
    // expert contributions, particularly with Top-8 routing. Scalar accesses
    // also handle arbitrary row widths and unaligned tensor views safely.
    for (int i = threadIdx.x; i < CHUNK_SIZE && chunk_base + i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            const bf16 *src = reinterpret_cast<const bf16 *>(d_in_ptr[first_token_idx + k]);
            sum += __bfloat162float(src[chunk_base + i]);
        }
        out[chunk_base + i] = __float2bfloat16(sum);
    }
}

#define LAUNCH_GATHER_AND_SUM_KERNEL_(SIZE) \
do { \
    constexpr int chunk_size = (SIZE); \
    dim3 grid(n, (int64_t(hidden_size) + chunk_size - 1) / chunk_size, 1); \
    gather_and_sum_tokens_kernel<T, chunk_size><<<grid, block, 0, stream>>>(dest, src_ptr, n, topk, hidden_size); \
} while(0)

template <class T>
void _gather_and_sum_tokens_cuda(
    T *dest, 
    uintptr_t *src_ptr, 
    int n, 
    int topk, 
    int hidden_size
) {
    static_assert(sizeof(T) == 2);
    assert(hidden_size > 0);
    constexpr int num_threads = 128;
    dim3 block(num_threads, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hidden_size % 2048 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(2048);
    } else if (hidden_size % 960 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(960);
    } else if (hidden_size % 512 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(512);
    } else if (hidden_size % 256 == 0) {
        LAUNCH_GATHER_AND_SUM_KERNEL_(256);
    } else {
        LAUNCH_GATHER_AND_SUM_KERNEL_(128);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gather_and_sum_tokens_cuda_dispatch(
    torch::Tensor dest, 
    int64_t src_ptr, 
    int64_t n, 
    int64_t topk, 
    int64_t hidden_size
) {
    // dest is a cuda ptr with shape [n, hidden_size]
    // src_ptr is a cpu ptr to array of n*topk uintptr_t pointers
    TORCH_CHECK(dest.is_cuda() && dest.scalar_type() == at::kBFloat16, "dest must be a CUDA BFloat16 tensor");
    TORCH_CHECK(dest.dim() == 2 && dest.is_contiguous(), "dest must be a contiguous 2D tensor");
    TORCH_CHECK(n >= 0 && hidden_size > 0 && topk > 0 && dest.size(0) == n && dest.size(1) == hidden_size, "dest shape must match n and hidden_size, and topk must be positive");
    TORCH_CHECK(n <= std::numeric_limits<int>::max() / topk && hidden_size <= std::numeric_limits<int>::max(), "token dimensions exceed kernel limits");
    if (n == 0) return;
    TORCH_CHECK(src_ptr != 0, "src_ptr must point to a host pointer array");
    c10::cuda::CUDAGuard device_guard(dest.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status;
    CUDACHECK(cudaStreamIsCapturing(stream, &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
        "gather_and_sum_tokens uses a temporary host pointer array; call it outside CUDA graph capture");
    uintptr_t* src_ptr_host = reinterpret_cast<uintptr_t*>(src_ptr);
    using scalar_t = c10::BFloat16;
    
#if KERNEL_USE_GDRCOPY == 1
    TORCH_CHECK(n * topk <= MAX_GATHER_TOKENS, "gather pointer array exceeds GDR staging capacity");
    gdr_context_t gather_src_ptrs_gdr = get_gather_and_sum_src_ptrs_gdr();
    gather_src_ptrs_gdr->copy_from_host(src_ptr_host, n * topk * sizeof(uintptr_t));
    auto src_tensor = gather_src_ptrs_gdr->get_tensor();
    src_tensor = src_tensor.narrow(0, 0, n * topk);
    _gather_and_sum_tokens_cuda<scalar_t>(
        dest.data_ptr<scalar_t>(), 
        src_tensor.data_ptr<uintptr_t>(), 
        n, 
        topk, 
        hidden_size
    );
#else
    // Tensor::to retains the pinned allocation until its asynchronous H2D
    // copy completes; the caller's temporary pointer vector is copied now.
    auto host_tensor = torch::empty({n * topk}, torch::TensorOptions()
        .dtype(torch::kUInt64)
        .device(torch::kCPU).pinned_memory(true));
    std::memcpy(host_tensor.data_ptr<uintptr_t>(), src_ptr_host, n * topk * sizeof(uintptr_t));
    auto src_tensor = host_tensor.to(dest.options().dtype(torch::kUInt64), /*non_blocking=*/true);
    _gather_and_sum_tokens_cuda<scalar_t>(
        dest.data_ptr<scalar_t>(), 
        src_tensor.data_ptr<uintptr_t>(), 
        n, 
        topk, 
        hidden_size
    );
#endif
}

TORCH_LIBRARY_FRAGMENT(disag_ops, m) {
    m.def("gather_and_sum_tokens(Tensor(a!) dest, int src_ptr, int n, int topk, int hidden_size) -> ()");
    m.impl("gather_and_sum_tokens", torch::kCUDA, gather_and_sum_tokens_cuda_dispatch);
}
