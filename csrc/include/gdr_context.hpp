#pragma once

#ifndef USE_GDRCOPY
#define USE_GDRCOPY 0
#endif

#if USE_GDRCOPY
#include <gdrapi.h>
#include <immintrin.h>
#endif

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// The legacy name is retained because both Python and C++ callers use this
// object for host metadata staging. The CUDA backend has no GDRCopy dependency.
// All copies complete before returning and are ordered on the tensor device's
// current stream. Callers must establish dependencies on any other streams.
// These host APIs are deliberately not CUDA-graph-capturable: stage metadata
// before replay, and read results after replay on the same stream.
class GdrContext {
public:
    explicit GdrContext(const torch::Tensor& tensor)
        : tensor_(tensor) {
        if (!tensor.is_cuda() || !tensor.is_contiguous()) {
            throw std::runtime_error("GdrContext: tensor must be contiguous and CUDA");
        }
        size_ = tensor.nbytes();
        dev_ptr_ = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        use_gdrcopy_ = gdrcopy_enabled();
        if (size_ == 0) return;
        const c10::cuda::CUDAGuard guard(tensor.device());
        host_stream(); // Reject creation during capture before pinning memory.
#if USE_GDRCOPY
        if (use_gdrcopy_) initialize_gdr();
#endif
    }

    GdrContext() = default;
    ~GdrContext() noexcept { cleanup(); }
    GdrContext(const GdrContext&) = delete;
    GdrContext& operator=(const GdrContext&) = delete;

    static bool gdrcopy_compiled() { return USE_GDRCOPY != 0; }

    // This reports build/runtime selection, not hardware/driver support. A
    // selected GDR backend checks gdr_open/pin/map when a context is constructed.
    static bool gdrcopy_enabled() {
        const char* raw = std::getenv("DMOE_USE_GDRCOPY");
        if (raw == nullptr) return gdrcopy_compiled();
        std::string value(raw);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (value == "0" || value == "false" || value == "off" || value == "no") return false;
        if (value == "1" || value == "true" || value == "on" || value == "yes") return gdrcopy_compiled();
        throw std::runtime_error("DMOE_USE_GDRCOPY must be a boolean (0 or 1)");
    }

    std::string backend() const { return use_gdrcopy_ ? "gdrcopy" : "cuda"; }
    torch::Tensor get_tensor() const { return tensor_; }
    uintptr_t device_ptr() const { return dev_ptr_; }
    void* host_mapped_ptr() const { return cpu_ptr_; } // nullptr for CUDA backend
    size_t size() const { return size_; }

    void copy_from_host(const void* src, size_t nbytes, size_t dst_offset = 0) {
        check_range(nbytes, dst_offset);
        if (nbytes == 0) return;
        if (!src) throw std::runtime_error("GdrContext::copy_from_host: null source");
        const c10::cuda::CUDAGuard guard(tensor_.device());
        auto stream = host_stream();
#if USE_GDRCOPY
        if (use_gdrcopy_) {
            // CPU BAR writes must not race a previous GPU reader/writer.
            check_cuda(cudaStreamSynchronize(stream), "wait before GDR write");
            check_gdr(gdr_copy_to_mapping(mh_, static_cast<char*>(cpu_ptr_) + dst_offset,
                                         src, nbytes), "gdr_copy_to_mapping");
            flush_gdr_writes();
            return;
        }
#endif
        check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(dev_ptr_ + dst_offset),
                                   src, nbytes, cudaMemcpyHostToDevice, stream),
                   "host-to-device copy");
        // src often belongs to a temporary std::vector or Python list. Waiting
        // here makes its lifetime independent of CUDA's pageable-copy behavior.
        check_cuda(cudaStreamSynchronize(stream), "host-to-device completion");
    }

    void copy_to_host(void* dst, size_t nbytes, size_t src_offset = 0) {
        check_range(nbytes, src_offset);
        if (nbytes == 0) return;
        if (!dst) throw std::runtime_error("GdrContext::copy_to_host: null destination");
        const c10::cuda::CUDAGuard guard(tensor_.device());
        auto stream = host_stream();
#if USE_GDRCOPY
        if (use_gdrcopy_) {
            check_cuda(cudaStreamSynchronize(stream), "wait before GDR read");
            check_gdr(gdr_copy_from_mapping(mh_, dst,
                                           static_cast<const char*>(cpu_ptr_) + src_offset,
                                           nbytes), "gdr_copy_from_mapping");
            _mm_lfence();
            return;
        }
#endif
        check_cuda(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(dev_ptr_ + src_offset),
                                   nbytes, cudaMemcpyDeviceToHost, stream),
                   "device-to-host copy");
        check_cuda(cudaStreamSynchronize(stream), "device-to-host completion");
    }

    void copy_from_host_tensor(const torch::Tensor& src, size_t nbytes = 0) {
        check_host_tensor(src);
        if (nbytes == 0) nbytes = src.nbytes();
        if (nbytes > src.nbytes()) throw std::runtime_error("GdrContext: source tensor is too small");
        copy_from_host(src.data_ptr(), nbytes);
    }

    void copy_to_host_tensor(torch::Tensor& dst, size_t nbytes = 0) {
        check_host_tensor(dst);
        if (nbytes == 0) nbytes = dst.nbytes();
        if (nbytes > dst.nbytes()) throw std::runtime_error("GdrContext: destination tensor is too small");
        copy_to_host(dst.data_ptr(), nbytes);
    }

    void copy_from_host_int32(const std::vector<int>& src) {
        copy_from_host(src.data(), src.size() * sizeof(int));
    }
    void copy_from_host_int64(const std::vector<int64_t>& src) {
        copy_from_host(src.data(), src.size() * sizeof(int64_t));
    }
    void copy_from_host_float(const std::vector<float>& src) {
        copy_from_host(src.data(), src.size() * sizeof(float));
    }
    std::vector<int> copy_to_host_int32(int nelems) { return read_vector<int>(nelems); }
    std::vector<int64_t> copy_to_host_int64(int nelems) { return read_vector<int64_t>(nelems); }
    std::vector<float> copy_to_host_float(int nelems) { return read_vector<float>(nelems); }

    void fill(uint8_t value, size_t nbytes, size_t dst_offset = 0) {
        check_range(nbytes, dst_offset);
        if (nbytes == 0) return;
        const c10::cuda::CUDAGuard guard(tensor_.device());
        auto stream = host_stream();
        // CUDA memset works for either backend and obeys GPU stream ordering.
        check_cuda(cudaMemsetAsync(reinterpret_cast<void*>(dev_ptr_ + dst_offset),
                                   value, nbytes, stream), "fill");
        check_cuda(cudaStreamSynchronize(stream), "fill completion");
    }

    static void ensure_gdr_closed() {
#if USE_GDRCOPY
        std::lock_guard<std::mutex> lock(gdr_mutex_);
        if (active_mappings_ != 0) {
            throw std::runtime_error("GdrContext: cannot close GDRCopy with live mappings");
        }
        if (gdr_handle_) {
            check_gdr(gdr_close(gdr_handle_), "gdr_close");
            gdr_handle_ = nullptr;
        }
#endif
    }

private:
    static void check_cuda(cudaError_t rc, const char* operation) {
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::string("GdrContext ") + operation + ": " + cudaGetErrorString(rc));
        }
    }

    void check_range(size_t nbytes, size_t offset) const {
        if (!tensor_.defined()) throw std::runtime_error("GdrContext: uninitialized context");
        if (offset > size_ || nbytes > size_ - offset) {
            throw std::runtime_error("GdrContext: copy exceeds device buffer bounds");
        }
    }

    static void check_host_tensor(const torch::Tensor& tensor) {
        if (!tensor.device().is_cpu() || !tensor.is_contiguous()) {
            throw std::runtime_error("GdrContext: host tensor must be contiguous and on CPU");
        }
    }

    cudaStream_t host_stream() const {
        auto stream = c10::cuda::getCurrentCUDAStream(tensor_.get_device()).stream();
        cudaStreamCaptureStatus status;
        check_cuda(cudaStreamIsCapturing(stream, &status), "query capture status");
        if (status != cudaStreamCaptureStatusNone) {
            throw std::runtime_error("GdrContext: host staging must run outside CUDA graph capture");
        }
        return stream;
    }

    template <typename T>
    std::vector<T> read_vector(int nelems) {
        if (nelems < 0) throw std::runtime_error("GdrContext: element count must be nonnegative");
        const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
        check_range(nbytes, 0); // Check before allocating host memory.
        std::vector<T> result(nelems);
        copy_to_host(result.data(), nbytes);
        return result;
    }

#if USE_GDRCOPY
    static void check_gdr(int rc, const char* operation) {
        if (rc != 0) {
            throw std::runtime_error(std::string("GdrContext ") + operation +
                                     " failed (rc=" + std::to_string(rc) +
                                     "). Set DMOE_USE_GDRCOPY=0 to use CUDA host staging.");
        }
    }

    void initialize_gdr() {
        std::lock_guard<std::mutex> lock(gdr_mutex_);
        if (!gdr_handle_) {
            gdr_handle_ = gdr_open();
            if (!gdr_handle_) {
                throw std::runtime_error("gdr_open failed; check gdrdrv or set DMOE_USE_GDRCOPY=0");
            }
        }
        try {
            check_gdr(gdr_pin_buffer(gdr_handle_, dev_ptr_, size_, 0, 0, &mh_), "gdr_pin_buffer");
            pinned_ = true;
            ++active_mappings_;
            check_gdr(gdr_map(gdr_handle_, mh_, &bar_ptr_, size_), "gdr_map");
            mapped_ = true;
            gdr_info_t info{};
            check_gdr(gdr_get_info(gdr_handle_, mh_, &info), "gdr_get_info");
            if (dev_ptr_ < info.va || dev_ptr_ - info.va > info.mapped_size ||
                size_ > info.mapped_size - (dev_ptr_ - info.va)) {
                throw std::runtime_error("GdrContext: device tensor exceeds BAR mapping");
            }
            cpu_ptr_ = static_cast<char*>(bar_ptr_) + (dev_ptr_ - info.va);
        } catch (...) {
            cleanup_gdr_unlocked();
            throw;
        }
    }

    void flush_gdr_writes() {
        _mm_sfence();
        check_cuda(cudaDeviceFlushGPUDirectRDMAWrites(
                       cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                       cudaFlushGPUDirectRDMAWritesToOwner), "flush GDR writes");
    }

    void cleanup_gdr_unlocked() noexcept {
        if (mapped_) {
            const int rc = gdr_unmap(gdr_handle_, mh_, bar_ptr_, size_);
            if (rc != 0) std::cerr << "GdrContext: gdr_unmap failed (rc=" << rc << ")\n";
            mapped_ = false;
        }
        if (pinned_) {
            const int rc = gdr_unpin_buffer(gdr_handle_, mh_);
            if (rc != 0) std::cerr << "GdrContext: gdr_unpin_buffer failed (rc=" << rc << ")\n";
            pinned_ = false;
            --active_mappings_;
        }
        cpu_ptr_ = nullptr;
        bar_ptr_ = nullptr;
        mh_ = {};
    }

    inline static gdr_t gdr_handle_ = nullptr;
    inline static std::mutex gdr_mutex_;
    inline static size_t active_mappings_ = 0;
    gdr_mh_t mh_{};
    void* bar_ptr_ = nullptr;
    bool pinned_ = false;
    bool mapped_ = false;
#endif

    void cleanup() noexcept {
#if USE_GDRCOPY
        std::lock_guard<std::mutex> lock(gdr_mutex_);
        if (pinned_ || mapped_) cleanup_gdr_unlocked();
#endif
    }

    torch::Tensor tensor_;
    bool use_gdrcopy_ = false;
    void* cpu_ptr_ = nullptr;
    size_t size_ = 0;
    uintptr_t dev_ptr_ = 0;
};

using gdr_context_t = std::shared_ptr<GdrContext>;
