"""Describe a single node as seen by the runtime. topo -m labels are physical links."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from .common import FORMAT, VERSION, file_hash, load_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
QWEN = dict(name="Qwen3-30B-A3B", hidden_size=2048, intermediate_size=768, num_layers=48,
            num_attention_heads=32, num_key_value_heads=4, head_dim=128, num_experts=128, top_k=8)
TRANSPORT_ENV = ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_P2P_LEVEL",
                 "NCCL_CUMEM_HOST_ENABLE", "NCCL_NET", "NCCL_ALGO", "NCCL_PROTO")
OPERATOR_ENV = ("DMOE_USE_GDRCOPY", "KERNEL_USE_GDRCOPY", "ENABLE_GDRCOPY", "ENABLE_NIXL",
                "VLLM_USE_V1", "VLLM_FLASH_ATTN_VERSION", "STREAMEP_SHARED_TOP8",
                "DMOE_WEIGHTED_ROUTER_FILE")


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, cwd=ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def packages():
    versions = {}
    for name in ("torch", "triton", "vllm", "flash-attn", "deep-gemm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def native_extension():
    spec = importlib.util.find_spec("disagmoe_c")
    if not spec or not spec.origin or not Path(spec.origin).is_file():
        return None
    path = Path(spec.origin)
    return {"filename": path.name, "sha256": file_hash(path)}


def get_model(path):
    if path is None:
        return QWEN.copy()
    config = json.loads(Path(path).read_text())
    # HF's intermediate_size often describes a dense MLP, not one routed expert.
    fields = {"hidden_size": "hidden_size", "intermediate_size": "moe_intermediate_size",
              "num_layers": "num_hidden_layers", "num_attention_heads": "num_attention_heads",
              "num_key_value_heads": "num_key_value_heads", "num_experts": "num_experts",
              "top_k": "num_experts_per_tok"}
    for field in fields.values():
        if field not in config:
            raise ValueError(f"Model config requires {field} for this MoE workload")
    model = {key: config[field] for key, field in fields.items()}
    head_dim = config.get("head_dim")
    if head_dim is None:
        if model["num_attention_heads"] <= 0 or model["hidden_size"] % model["num_attention_heads"]:
            raise ValueError("Please specify an integral head_dim in the model config")
        head_dim = model["hidden_size"] // model["num_attention_heads"]
    model.update(name=config.get("_name_or_path") or config.get("model_type") or Path(path).parent.name,
                 head_dim=head_dim)
    return model


def visible_gpus():
    import torch
    devices = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        uuid = getattr(prop, "uuid", None)
        devices.append(dict(logical_index=i, name=prop.name, uuid=str(uuid) if uuid else None,
                            sm=f"{prop.major}.{prop.minor}", memory_bytes=prop.total_memory))
    if len(devices) < 2:
        raise ValueError("At least two visible CUDA GPUs are needed on the same node; check CUDA_VISIBLE_DEVICES")
    return devices


def environment():
    return {key: os.environ.get(key) for key in TRANSPORT_ENV}


def operating_state():
    """Uncontrolled endpoint snapshots, with physical UUIDs and nvidia-smi units."""
    return command(["nvidia-smi", "--query-gpu=uuid,pstate,clocks.current.sm,clocks.current.memory,"
                    "temperature.gpu,power.draw,power.limit", "--format=csv"])


def verify_machine(manifest, stage):
    """Protect against accidentally appending samples from another allocation or environment."""
    import torch
    current = visible_gpus()
    fallback_changed = (not all(gpu["uuid"] for gpu in current)
                        and (platform.node() != manifest["hardware"]["node"]
                             or os.getenv("CUDA_VISIBLE_DEVICES") != manifest["runtime"]["cuda_visible_devices"]))
    if current != manifest["hardware"]["gpus"] or fallback_changed:
        raise ValueError("GPU inventory or host differs from bundle; initialize a new bundle on this allocation")
    if (str(torch.__version__) != manifest["software"]["torch"] or
            torch.version.cuda != manifest["software"]["torch_cuda"]):
        raise ValueError("PyTorch or CUDA build changed since bundle initialization")
    if stage == "network" and environment() != manifest["runtime"]["transport_env"]:
        raise ValueError("NCCL transport settings changed; use a separate bundle for this configuration")
    if stage != "network" and any(os.getenv(key) != value for key, value in manifest["runtime"]["operator_env"].items()):
        raise ValueError("Operator environment changed since initialization; use a separate bundle")
    if stage != "network" and native_extension() != manifest["software"]["native_extension"]:
        raise ValueError("The compiled StreamInfer extension changed since bundle initialization")


def initialize(args):
    import torch
    target = args.bundle.resolve()
    if target.exists():
        raise FileExistsError(f"Choose a new bundle directory: {target}")
    devices = visible_gpus()
    count = len(devices)
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    topo = command(["nvidia-smi", "topo", "-m"])
    physical = command(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                        "--format=csv,noheader"])
    repo = ["git", "-C", str(ROOT)]
    status = command([*repo, "status", "--porcelain"])
    manifest = {
        "format": FORMAT, "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": get_model(args.model_config),
        "parallel": {"dp": count, "ep": count, "tp": 1},
        "hardware": dict(node=platform.node(), nodes=1, cpu=cpu_model(), system=platform.platform(),
                         cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                         interconnect=args.interconnect, interconnect_source="operator supplied physical fabric",
                         gpus=devices, physical_gpu_inventory=physical, nvidia_smi_topo=topo,
                         cuda_peer_access=[[None if i == j else torch.cuda.can_device_access_peer(i, j)
                                            for j in range(count)] for i in range(count)]),
        "runtime": dict(dtype="bfloat16", cuda_graph=args.cuda_graph, cuda_visible_devices=mask,
                        transport_env=environment(), operator_env={key: os.getenv(key) for key in OPERATOR_ENV},
                        weight_source="synthetic; operator shapes from model"),
        "software": dict(torch=str(torch.__version__), torch_cuda=torch.version.cuda, packages=packages(),
                         native_extension=native_extension(),
                         python=platform.python_version(), commit=command([*repo, "rev-parse", "HEAD"]),
                         checkout_dirty=bool(status) if status is not None else None),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".profile-init-", dir=target.parent) as tmp:
        draft = Path(tmp) / "bundle"
        draft.mkdir()
        write_json(draft / "manifest.json", manifest)
        load_manifest(draft)
        draft.rename(target)
    print(target / "manifest.json")


def add_commands(subparsers):
    parser = subparsers.add_parser("init", help="Describe the visible GPUs, physical topology, model and runtime")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--interconnect", choices=("pcie", "nvlink"), required=True,
                        help="Physical single-node fabric; raw per-pair topology is also saved")
    parser.add_argument("--model-config", type=Path, help="HF MoE config.json; default Qwen3-30B-A3B shapes")
    parser.add_argument("--cuda-graph", action="store_true", help="Profile native graph compute; default eager")
    parser.set_defaults(func=initialize)
