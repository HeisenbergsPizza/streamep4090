"""Host metadata staging with an optional GDRCopy backend.

Build with ENABLE_GDRCOPY=0 to omit gdrapi entirely. Set DMOE_USE_GDRCOPY=0
before starting workers to use CUDA copies with either build. Copies complete
on the tensor device's current stream before returning; they must run outside
CUDA graph capture. Callers must order work on other streams themselves.
"""

from disagmoe_c import (
    GdrContext as GdrContextImpl,
    gdrcopy_compiled,
    gdrcopy_enabled,
)
from disagmoe.utils.tensor_utils import get_cuda_aligned_tensor
import torch

# Runtime configuration is fixed before importing the engine in each worker.
# Contexts remain usable when this is False, including existing C++ callers.
use_gdrcopy_optimization = gdrcopy_enabled()

class GdrContext:
    
    def __init__(self, tensor: torch.Tensor):
        self.gdr_context = GdrContextImpl(tensor)
        self.tensor = tensor

    @property
    def backend(self) -> str:
        return self.gdr_context.backend()
        
    def copy_from_host(self, src: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.copy_from_host(src, nbytes, dst_offset)
        
    def copy_from_host_tensor(self, src: torch.Tensor, nbytes: int = 0) -> None:
        self.gdr_context.copy_from_host_tensor(src, nbytes)
        
    def copy_to_host(self, dest: int, nbytes: int, src_offset: int = 0) -> None:
        self.gdr_context.copy_to_host(dest, nbytes, src_offset)
        
    def copy_to_host_tensor(self, dst: torch.Tensor, nbytes: int = 0) -> None:
        self.gdr_context.copy_to_host_tensor(dst, nbytes)
        
    def fill(self, value: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.fill(value, nbytes, dst_offset)
        
    def copy_from_host_int32(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int32(src)
        
    def copy_from_host_int64(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int64(src)
        
    def copy_from_host_float(self, src: list[float]) -> None:
        self.gdr_context.copy_from_host_float(src)
        
    def copy_to_host_int32(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int32(nelems)
    
    def copy_to_host_float(self, nelems: int) -> list[float]:
        return self.gdr_context.copy_to_host_float(nelems)
        
    def copy_to_host_int64(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int64(nelems)
    
class GdrDoubleBuffer:
    
    def __init__(self, nelems: int, dtype: torch.dtype, device: str = "cuda"):
        self.buffers = [get_cuda_aligned_tensor(nelems, dtype, device=device), get_cuda_aligned_tensor(nelems, dtype, device=device)]
        self.gdr_contexts = [GdrContext(self.buffers[0]), GdrContext(self.buffers[1])]
        self.internal_index = 0
        
    def get_one_handle(self) -> GdrContext:
        self.internal_index = (self.internal_index + 1) & 1
        return self.gdr_contexts[self.internal_index]
