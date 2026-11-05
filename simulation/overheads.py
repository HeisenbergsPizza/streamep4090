"""Configurable *assumed exposed* batch costs, not a CPU/GPU pipeline model.

Table 4 reports Schedule separately from Prepare/Launch/Post-process. Only
Schedule is used here. The paper does NOT report a GPU reformation cost; its
default 5% of base compute is an explicit sensitivity-study assumption.
The structural preset instead charges logical rows, buffer fragments, candidate
queues and activation bytes. All of its default coefficients are assumptions.
"""

from dataclasses import dataclass
import math
from numbers import Integral


PAPER_URL = "https://doi.org/10.1145/3830418.3843869"
PAPER_SCHEDULE_US = {"attention": 127.2, "expert": 29.6}
PAPER_SCHEDULE_HIDDEN = {"attention": 0.506, "expert": 0.683}
STRUCTURAL_DEFAULTS = {
    "cpu_fixed_us": 10.0,
    "cpu_per_row_us": 0.02,
    "cpu_per_fragment_us": 0.1,
    "cpu_per_queue_us": 0.05,
    "reformation_fixed_us": 3.0,
    "reformation_bandwidth_gb_s": 500.0,
    "reformation_per_fragment_us": 0.05,
    "reformation_scale": 1.0,
}


def _nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _fraction(value, name):
    value = _nonnegative(value, name)
    if value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class BatchWorkload:
    """Logical inputs selected for one local batch, not physical allocations.

    input_rows counts incoming token/Expert branches, including all inputs to a
    Top-K Attention reduction. fragments counts selected contiguous row spans
    within each logical source buffer, not individual rows or physical allocations;
    candidate_queues counts queues examined by the scheduling decision, including
    empty slots when the scheduler scans a dense queue array.
    contiguous identifies a single contiguous Expert input usable without a copy.
    """

    input_rows: int
    fragments: int
    candidate_queues: int
    hidden_dim: int
    bytes_per_element: int
    contiguous: bool = False

    def __post_init__(self):
        for name in ("input_rows", "fragments", "candidate_queues", "hidden_dim", "bytes_per_element"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"workload {name} must be a nonnegative integer")
        if self.hidden_dim == 0 or self.bytes_per_element == 0:
            raise ValueError("workload hidden_dim and bytes_per_element must be positive")
        if not isinstance(self.contiguous, bool):
            raise ValueError("workload contiguous must be a bool")


@dataclass(frozen=True)
class BatchOverheadEstimate:
    base_compute_ticks: float
    cpu_raw_ticks: float = 0.0
    cpu_exposed_ticks: float = 0.0
    reformation_raw_ticks: float = 0.0
    reformation_exposed_ticks: float = 0.0
    details: dict | None = None

    @property
    def gpu_ticks(self):
        return self.base_compute_ticks + self.reformation_exposed_ticks

    @property
    def total_ticks(self):
        return self.cpu_exposed_ticks + self.gpu_ticks

    def as_dict(self):
        return {
            "base_compute_ticks": self.base_compute_ticks,
            "cpu_raw_ticks": self.cpu_raw_ticks,
            "cpu_exposed_ticks": self.cpu_exposed_ticks,
            "cpu_hidden_ticks": self.cpu_raw_ticks - self.cpu_exposed_ticks,
            "reformation_raw_ticks": self.reformation_raw_ticks,
            "reformation_exposed_ticks": self.reformation_exposed_ticks,
            "reformation_hidden_ticks": self.reformation_raw_ticks - self.reformation_exposed_ticks,
            "gpu_ticks": self.gpu_ticks,
            "total_ticks": self.total_ticks,
        }


class BatchOverheadModel:
    def __init__(self, *, ticks_per_millisecond=10, preset="none",
                 cpu_schedule_scale=1.0, cpu_schedule_overlap=None,
                 reformation_compute_ratio=None, reformation_overlap=0.0,
                 cpu_fixed_us=None, cpu_per_row_us=None,
                 cpu_per_fragment_us=None, cpu_per_queue_us=None,
                 reformation_fixed_us=None, reformation_bandwidth_gb_s=None,
                 reformation_per_fragment_us=None, reformation_scale=None):
        if preset not in ("none", "paper", "structural"):
            raise ValueError("overhead preset must be 'none', 'paper', or 'structural'")
        self.preset = preset
        self.enabled = preset != "none"
        self.ticks_per_ms = _nonnegative(ticks_per_millisecond, "ticks_per_millisecond")
        if self.ticks_per_ms == 0:
            raise ValueError("ticks_per_millisecond must be positive")
        self.cpu_schedule_scale = _nonnegative(cpu_schedule_scale, "cpu_schedule_scale")
        self.cpu_schedule_overlap = (None if cpu_schedule_overlap is None else
                                     _fraction(cpu_schedule_overlap, "cpu_schedule_overlap"))
        if preset == "structural" and self.cpu_schedule_overlap is None:
            self.cpu_schedule_overlap = 0.0
        self.reformation_compute_ratio = _nonnegative(
            (0.05 if preset == "paper" else 0.0) if reformation_compute_ratio is None else reformation_compute_ratio,
            "reformation_compute_ratio")
        self.reformation_overlap = _fraction(reformation_overlap, "reformation_overlap")
        if not self.enabled and (self.cpu_schedule_scale != 1.0 or self.cpu_schedule_overlap is not None
                                 or self.reformation_compute_ratio != 0 or self.reformation_overlap != 0):
            raise ValueError("overhead overrides require --overhead-model paper or structural")
        coefficients = {
            "cpu_fixed_us": cpu_fixed_us,
            "cpu_per_row_us": cpu_per_row_us,
            "cpu_per_fragment_us": cpu_per_fragment_us,
            "cpu_per_queue_us": cpu_per_queue_us,
            "reformation_fixed_us": reformation_fixed_us,
            "reformation_bandwidth_gb_s": reformation_bandwidth_gb_s,
            "reformation_per_fragment_us": reformation_per_fragment_us,
            "reformation_scale": reformation_scale,
        }
        if preset != "structural" and any(value is not None for value in coefficients.values()):
            raise ValueError("structural coefficients require --overhead-model structural")
        if preset == "structural":
            if self.reformation_compute_ratio != 0:
                raise ValueError("structural reformation uses bytes/fragments; reformation_compute_ratio must be zero")
            for name, value in coefficients.items():
                setattr(self, name, _nonnegative(STRUCTURAL_DEFAULTS[name] if value is None else value, name))
            if self.reformation_bandwidth_gb_s == 0:
                raise ValueError("reformation_bandwidth_gb_s must be positive")

    def config(self):
        result = {name: getattr(self, name) for name in (
            "preset", "cpu_schedule_scale", "cpu_schedule_overlap",
            "reformation_compute_ratio", "reformation_overlap")}
        if self.preset == "structural":
            result.update({name: getattr(self, name) for name in STRUCTURAL_DEFAULTS})
        return result

    def evaluate(self, stage, compute_ticks, batch_size=1, workload=None):
        if stage not in ("attention", "expert", "finalize"):
            raise ValueError("overhead stage must be attention, expert, or finalize")
        compute_ticks = _nonnegative(compute_ticks, "compute_ticks")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if not self.enabled or stage == "finalize" or batch_size == 0:
            return BatchOverheadEstimate(compute_ticks)
        if self.preset == "structural":
            return self._evaluate_structural(stage, compute_ticks, batch_size, workload)
        hidden = (PAPER_SCHEDULE_HIDDEN[stage] if self.cpu_schedule_overlap is None
                  else self.cpu_schedule_overlap)
        cpu = PAPER_SCHEDULE_US[stage] * self.cpu_schedule_scale / 1000 * self.ticks_per_ms
        reform = compute_ticks * self.reformation_compute_ratio
        estimate = BatchOverheadEstimate(compute_ticks, cpu, cpu * (1 - hidden),
                                         reform, reform * (1 - self.reformation_overlap))
        if not math.isfinite(estimate.total_ticks):
            raise ValueError("batch overhead estimate must be finite")
        return estimate

    def _evaluate_structural(self, stage, compute_ticks, batch_size, workload):
        if not isinstance(workload, BatchWorkload):
            raise ValueError("structural nonempty attention/expert batches require a BatchWorkload")
        if workload.input_rows < batch_size:
            raise ValueError("workload input_rows must be at least batch_size")
        if not 1 <= workload.fragments <= workload.input_rows:
            raise ValueError("nonempty workload fragments must be between 1 and input_rows")
        if workload.candidate_queues == 0:
            raise ValueError("nonempty workload candidate_queues must be positive")
        if workload.contiguous and workload.fragments != 1:
            raise ValueError("contiguous workload must have exactly one fragment")

        # For Attention, input rows may include K returns reduced to one output
        # row. For Expert, a row is one routed token/Expert branch.
        input_bytes = int(workload.input_rows) * int(workload.hidden_dim) * int(workload.bytes_per_element)
        output_bytes = int(batch_size) * int(workload.hidden_dim) * int(workload.bytes_per_element)
        traffic_bytes = input_bytes + output_bytes
        fastpath = stage == "expert" and workload.contiguous
        try:
            cpu_parts = {
                "fixed": self.cpu_fixed_us,
                "rows": self.cpu_per_row_us * workload.input_rows,
                "fragments": self.cpu_per_fragment_us * workload.fragments,
                "candidate_queues": self.cpu_per_queue_us * workload.candidate_queues,
            }
            reform_parts = {
                "fixed": 0.0 if fastpath else self.reformation_fixed_us,
                "bytes": 0.0 if fastpath else traffic_bytes / self.reformation_bandwidth_gb_s / 1000,
                "fragments": 0.0 if fastpath else self.reformation_per_fragment_us * workload.fragments,
            }
            cpu_us = sum(cpu_parts.values()) * self.cpu_schedule_scale
            reform_unscaled_us = sum(reform_parts.values())
            reform_us = reform_unscaled_us * self.reformation_scale
            cpu = cpu_us / 1000 * self.ticks_per_ms
            reform = reform_us / 1000 * self.ticks_per_ms
        except OverflowError as exc:
            raise ValueError("batch overhead estimate must be finite") from exc
        estimate = BatchOverheadEstimate(
            compute_ticks, cpu, cpu * (1 - self.cpu_schedule_overlap),
            reform, reform * (1 - self.reformation_overlap),
            details={
                "workload": {
                    **{name: int(getattr(workload, name)) for name in (
                        "input_rows", "fragments", "candidate_queues", "hidden_dim", "bytes_per_element")},
                    "contiguous": workload.contiguous,
                    "read_bytes": input_bytes,
                    "write_bytes": output_bytes,
                    "traffic_bytes": traffic_bytes,
                },
                "components": {
                    "cpu_fixed_us": cpu_parts["fixed"],
                    "cpu_rows_us": cpu_parts["rows"],
                    "cpu_fragments_us": cpu_parts["fragments"],
                    "cpu_queues_us": cpu_parts["candidate_queues"],
                    "cpu_unscaled_us": sum(cpu_parts.values()),
                    "cpu_schedule_scale": self.cpu_schedule_scale,
                    "cpu_raw_us": cpu_us,
                    "reformation_fixed_us": reform_parts["fixed"],
                    "reformation_bytes_us": reform_parts["bytes"],
                    "reformation_fragments_us": reform_parts["fragments"],
                    "reformation_unscaled_us": reform_unscaled_us,
                    "reformation_scale": self.reformation_scale,
                    "reformation_raw_us": reform_us,
                    "fast_path": fastpath,
                    "executed_traffic_bytes": 0 if fastpath or self.reformation_scale == 0 else traffic_bytes,
                },
            })
        if not all(math.isfinite(value) for value in estimate.as_dict().values()):
            raise ValueError("batch overhead estimate must be finite")
        return estimate

    def describe(self):
        if self.preset == "structural":
            return {
                **self.config(), "enabled": True, "hardware_calibrated": False,
                "model": "assumed_structural_batch_overhead",
                "formula": "base_compute + cpu_raw*(1-cpu_overlap) + reformation_raw*(1-reformation_overlap)",
                "time_unit": "ticks", "ticks_per_millisecond": self.ticks_per_ms,
                "cpu_formula_us": "cpu_schedule_scale * (cpu_fixed_us + cpu_per_row_us*input_rows + cpu_per_fragment_us*fragments + cpu_per_queue_us*candidate_queues)",
                "reformation_formula_us": "reformation_scale * (reformation_fixed_us + traffic_bytes/(reformation_bandwidth_gb_s*1000) + reformation_per_fragment_us*fragments); zero for contiguous Expert fastpath",
                "traffic_formula_bytes": "(input_rows + batch_size)*hidden_dim*bytes_per_element",
                "cpu_source": "assumed coefficients for scheduling and logical metadata processing; excludes Launch, Post-process, OS scheduling and separate communication threads",
                "cpu_hidden_fraction_applied": {stage: self.cpu_schedule_overlap for stage in PAPER_SCHEDULE_US},
                "reformation_source": "assumed fixed cost, effective activation-copy bandwidth and fragment cost; no hardware calibration",
                "initialization_rationale": "all default coefficients are adjustable sensitivity-study assumptions, not paper measurements or a hardware prediction",
                "coefficient_units": {"cpu_fixed_us": "us/batch", "cpu_per_row_us": "us/input row",
                                      "cpu_per_fragment_us": "us/logical fragment", "cpu_per_queue_us": "us/candidate queue",
                                      "reformation_fixed_us": "us/batch", "reformation_bandwidth_gb_s": "GB/s, decimal bytes (not Gbit/s)",
                                      "reformation_per_fragment_us": "us/logical fragment", "reformation_scale": "dimensionless"},
                "workload_semantics": "input_rows are input activation/Expert-branch rows; fragments count selected contiguous row spans within logical source buffers, not physical allocator fragmentation; candidate_queues are queues examined by the scheduling decision, including empty slots in dense scans; workload read/write/traffic bytes are potential copying volume, components executed_traffic_bytes is zero on the fastpath or when reformation_scale=0 disables modeled work",
                "scale_semantics": "reformation_scale multiplies raw GPU cost before overlap; zero disables modeled GPU reformation work; component cost terms remain unscaled and fast_path describes structural eligibility independently of scale",
                "fastpath_semantics": "one contiguous Expert input bypasses GPU reformation; CPU still charged; Attention always packs/reduces",
                "charge_scope": "once per nonempty local attention/expert batch; no finalize charge",
                "overlap_semantics": "fixed hidden fractions, default zero; no independent CPU resource, thread contention, pipeline, or overlap-capacity simulation",
                "gpu_busy_semantics": "base compute plus exposed GPU reformation; excludes exposed CPU and assumed hidden work",
                "cross_mode_assumption": "same unit costs across modes; each mode must supply its actual logical workload",
                "limitations": ["logical fragments approximate receive-buffer pieces, not allocation layout or CUDA memory transactions",
                                "activation-byte model excludes indices, padding, cache effects and explicit Top-K reduction arithmetic",
                                "CPU coefficients abstract scheduling/metadata work, not CPU instruction execution",
                                "compute profiles must exclude these overheads to avoid double counting"],
            }
        return {
            **self.config(), "enabled": self.enabled, "hardware_calibrated": False,
            "model": "assumed_exposed_batch_overhead",
            "formula": "base_compute + cpu_schedule*(1-cpu_overlap) + base_compute*reformation_ratio*(1-reformation_overlap)",
            "time_unit": "ticks", "ticks_per_millisecond": self.ticks_per_ms,
            "cpu_source": {"url": PAPER_URL, "table": 4, "pdf_page": 13,
                           "scope": "Schedule row only; excludes Prepare, Launch, Post-process",
                           "workload": "GPT-OSS-120B, BF16, Sphere (16 L40S GPUs)",
                           "schedule_us_per_batch": PAPER_SCHEDULE_US.copy(),
                           "reported_hidden_fraction": PAPER_SCHEDULE_HIDDEN.copy()},
            "cpu_hidden_fraction_applied": {
                stage: PAPER_SCHEDULE_HIDDEN[stage] if self.cpu_schedule_overlap is None
                else self.cpu_schedule_overlap for stage in PAPER_SCHEDULE_US},
            "reformation_source": "assumed_compute_ratio; no isolated GPU reformation duration reported in paper",
            "reformation_default_rationale": "5% is a sensitivity-study assumption, NOT the paper's 4.5% combined throughput improvement",
            "charge_scope": "once per nonempty local attention/expert batch; no finalize charge",
            "overlap_semantics": "fixed hidden fractions, including first/isolated batches; no CPU resource, pipeline, or overlap-capacity simulation",
            "gpu_busy_semantics": "base compute plus exposed GPU reformation; excludes exposed CPU and assumed hidden work",
            "cross_mode_assumption": "same per-batch charges for all policies/modes; not measured sync/TBO scheduler overhead",
            "limitations": ["constant CPU cost extrapolated across batch sizes, policies, models and hardware",
                            "no fragment-count/fast-path, Top-k, context or cache dependence in reformation proxy",
                            "compute profiles must exclude these overheads to avoid double counting"],
        }


def add_overhead_arguments(parser):
    group = parser.add_argument_group("assumed CPU scheduling and GPU batch reformation")
    group.add_argument("--overhead-model", choices=("none", "paper", "structural"), default="none",
                       help="paper: Table 4 Schedule + assumed compute ratio; structural: assumed rows/fragments/bytes costs; default off")
    group.add_argument("--cpu-schedule-scale", type=float, default=1.0)
    group.add_argument("--cpu-schedule-overlap", type=float, default=None,
                       help="override hidden fraction [0,1] for both stages; paper defaults A=.506/E=.683, structural 0")
    group.add_argument("--reformation-compute-ratio", type=float, default=None,
                       help="assumed GPU reformation / base compute; paper mode default .05, NOT measured")
    group.add_argument("--reformation-overlap", type=float, default=0.0,
                       help="assumed hidden GPU reformation fraction [0,1]; default 0 (serial)")
    for name, default in STRUCTURAL_DEFAULTS.items():
        unit = ("GB/s (decimal bytes)" if name == "reformation_bandwidth_gb_s" else
                "(dimensionless)" if name == "reformation_scale" else "us")
        group.add_argument("--" + name.replace("_", "-"), type=float, default=None,
                           help=f"structural preset only; assumed default {default:g} {unit}")


def overhead_config_from_args(args):
    return BatchOverheadModel(
        preset=getattr(args, "overhead_model", "none"),
        cpu_schedule_scale=getattr(args, "cpu_schedule_scale", 1.0),
        cpu_schedule_overlap=getattr(args, "cpu_schedule_overlap", None),
        reformation_compute_ratio=getattr(args, "reformation_compute_ratio", None),
        reformation_overlap=getattr(args, "reformation_overlap", 0.0),
        ticks_per_millisecond=getattr(args, "ticks_per_ms", 10),
        **{name: getattr(args, name, None) for name in STRUCTURAL_DEFAULTS},
    ).config()
