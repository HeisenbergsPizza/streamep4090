# Simulation network contract

The simulators model transfer completion events and queueing, not NCCL execution.
The CPU runner (`python -m simulation.run`) defaults to one logical host. It
accepts an assumed per-receiver bandwidth in **Gb/s** and startup latency in us:

```
bytes = expert_token_copies * hidden_dim * bytes_per_element
seconds = latency_us / 1e6 + bytes / (bandwidth_gbps * 1e9 / 8)
```

These parameters are assumptions until calibrated on the target machine. A100
or H100 names in a compute-profile filename do not calibrate the network.

## Local transfers and receive resources

Same-GPU transfers cost zero and do not acquire a network resource. For remote
transfers, `gpu // n_gpu_per_host` identifies the host. Each receiving GPU has
one intra-host FIFO and one inter-host FIFO. Transfers on one FIFO serialize;
different destinations or different channel types may run concurrently. This
is an explicit resource model, not a claim about all physical NVLink/NVSwitch
and RDMA topologies. Sender ports, switch contention, control-plane overhead,
and compute/communication interference are not modeled.

The transfer unit counts routed **expert copies**, including multiple experts
on the same destination. Dispatch groups one attention batch by destination;
return groups one expert batch by the originating attention GPU. No assumed
cross-expert payload deduplication is applied.

## Execution models

- Async: dispatch and return share each destination's two receive FIFOs. Each
  destination releases its own tokens immediately on completion. Expert compute
  releases the compute resource before launching return traffic; the next
  compute may overlap with that return. The next attention stage requires all
  of a token's Top-k results. A local transfer still follows event ordering.
- Sync: all transfers in each dispatch/return phase are aggregated by receiver
  and channel, and the phase waits for the slowest receiver. There is a global
  stage boundary.
- TBO: both microbatches and both communication directions share receiver/channel
  availability throughout the iteration. Compute waits for that microbatch's
  preceding transfer. Final token completion includes the final return; the next
  iteration begins after both microbatches finish. Compute remains the existing
  single centralized resource abstraction.

With no `net_delay_fn`, the Python APIs use fixed remote dispatch/return delays
without receive contention. Local delay remains zero in all three models.

## Legacy experiment driver

`experiment.py` retains a beta-only bytes/bandwidth network model when bandwidth
mode is enabled: the two fixed-delay configuration columns are then unused.
When bandwidth mode is disabled, both same-host and cross-host remote traffic
use its fixed `net_delay_inter_host`; `net_delay_intra_host` is not used by that
legacy path. The new runner exposes the actual single-host bandwidth and startup
latency directly. See [README.md](README.md) for commands and metric definitions.
