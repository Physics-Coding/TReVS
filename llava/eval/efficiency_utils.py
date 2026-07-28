import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers.generation.streamers import BaseStreamer


@dataclass
class SampleEfficiencyRecord:
    question_id: str
    category: str
    total_time_ms: float
    ttft_ms: float
    prefill_time_ms: float
    prefill_vit_ms: float
    prefill_projector_ms: float
    prefill_route_ms: float
    prefill_route_K1_ms: float
    prefill_route_K2_ms: float
    prefill_llm_ms: float
    decode_llm_ms: float
    peak_gpu_memory_allocated_bytes: int
    peak_gpu_memory_allocated_gib: float
    kv_cache_bytes: float
    kv_cache_mib: float
    core_flops: float
    core_tflops: float
    routing_overhead_flops: float
    output_tokens: int
    routing_text_len: int
    prefill_seq_len_routed: int
    prefill_seq_len_phase: int
    n_vis_input: int
    n_vis_routed: int
    n_vis_phase: int
    vit_flops: float
    projector_flops: float
    llm_prefill_flops: float
    llm_decode_flops: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tflops_llm(prefill)"] = float(self.llm_prefill_flops / 1e12)
        return payload


class EfficiencyRuntimeContext:
    def __init__(self):
        self.prefill_start_event: Optional[torch.cuda.Event] = None
        self.prefill_end_event: Optional[torch.cuda.Event] = None
        self.vit_start_event: Optional[torch.cuda.Event] = None
        self.vit_end_event: Optional[torch.cuda.Event] = None
        self.projector_start_event: Optional[torch.cuda.Event] = None
        self.projector_end_event: Optional[torch.cuda.Event] = None
        self.route_start_event: Optional[torch.cuda.Event] = None
        self.route_end_event: Optional[torch.cuda.Event] = None
        self.route_k1_start_event: Optional[torch.cuda.Event] = None
        self.route_k1_end_event: Optional[torch.cuda.Event] = None
        self.route_k2_start_event: Optional[torch.cuda.Event] = None
        self.route_k2_end_event: Optional[torch.cuda.Event] = None
        self.llm_prefill_start_event: Optional[torch.cuda.Event] = None
        self.llm_prefill_end_event: Optional[torch.cuda.Event] = None
        self.llm_decode_start_event: Optional[torch.cuda.Event] = None
        self.llm_decode_end_event: Optional[torch.cuda.Event] = None
        self.metrics: Dict[str, Any] = {}


class FirstGeneratedTokenStreamer(BaseStreamer):
    def __init__(self):
        self._saw_prompt = False
        self.first_token_time: Optional[float] = None
        self.first_token_ids: Optional[List[int]] = None
        self.generated_token_count: int = 0

    def put(self, value):
        if value is None:
            return
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        token_count = int(value.numel()) if torch.is_tensor(value) else len(value)
        self.generated_token_count += token_count
        if self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()
            if torch.is_tensor(value):
                self.first_token_ids = value.detach().cpu().view(-1).tolist()
            else:
                self.first_token_ids = list(value)

    def end(self):
        return


class PopeEfficiencyAggregator:
    def __init__(self):
        self.records: List[SampleEfficiencyRecord] = []

    def add(self, record: SampleEfficiencyRecord):
        self.records.append(record)

    def summary(self) -> Dict[str, Any]:
        num_samples = len(self.records)
        if num_samples == 0:
            return {
                "num_samples": 0,
                "total_time_ms_sum": 0.0,
                "total_time_ms_avg": 0.0,
                "ttft_ms_sum": 0.0,
                "ttft_ms_avg": 0.0,
                "prefill_time_ms_sum": 0.0,
                "prefill_time_ms_avg": 0.0,
                "prefill_vit_ms_avg": 0.0,
                "prefill_projector_ms_avg": 0.0,
                "prefill_route_ms_avg": 0.0,
                "prefill_route_K1_ms_avg": 0.0,
                "prefill_route_K2_ms_avg": 0.0,
                "prefill_llm_ms_avg": 0.0,
                "decode_llm_ms_avg": 0.0,
                "peak_gpu_memory_allocated_gib_max": 0.0,
                "kv_cache_mib_avg": 0.0,
                "core_tflops_avg": 0.0,
                "tflops_llm(prefill)": 0.0,
            }
        return {
            "num_samples": num_samples,
            "total_time_ms_sum": sum(r.total_time_ms for r in self.records),
            "total_time_ms_avg": sum(r.total_time_ms for r in self.records) / num_samples,
            "ttft_ms_sum": sum(r.ttft_ms for r in self.records),
            "ttft_ms_avg": sum(r.ttft_ms for r in self.records) / num_samples,
            "prefill_time_ms_sum": sum(r.prefill_time_ms for r in self.records),
            "prefill_time_ms_avg": sum(r.prefill_time_ms for r in self.records) / num_samples,
            "prefill_vit_ms_avg": sum(r.prefill_vit_ms for r in self.records) / num_samples,
            "prefill_projector_ms_avg": sum(r.prefill_projector_ms for r in self.records) / num_samples,
            "prefill_route_ms_avg": sum(r.prefill_route_ms for r in self.records) / num_samples,
            "prefill_route_K1_ms_avg": sum(r.prefill_route_K1_ms for r in self.records) / num_samples,
            "prefill_route_K2_ms_avg": sum(r.prefill_route_K2_ms for r in self.records) / num_samples,
            "prefill_llm_ms_avg": sum(r.prefill_llm_ms for r in self.records) / num_samples,
            "decode_llm_ms_avg": sum(r.decode_llm_ms for r in self.records) / num_samples,
            "peak_gpu_memory_allocated_gib_max": max(r.peak_gpu_memory_allocated_gib for r in self.records),
            "kv_cache_mib_avg": sum(r.kv_cache_mib for r in self.records) / num_samples,
            "core_tflops_avg": sum(r.core_tflops for r in self.records) / num_samples,
            "tflops_llm(prefill)": sum(r.llm_prefill_flops for r in self.records) / num_samples / 1e12,
        }

    def write_samples_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def write_summary_json(self, path: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = self.summary()
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def write_report_markdown(self, path: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        summary = self.summary()
        if extra:
            summary = {**summary, **extra}
        lines = [
            "# POPE Efficiency Report",
            "",
            f"- num_samples: {summary.get('num_samples', 0)}",
            f"- total_time_ms_sum: {summary.get('total_time_ms_sum', 0.0):.6f}",
            f"- total_time_ms_avg: {summary.get('total_time_ms_avg', 0.0):.6f}",
            f"- ttft_ms_sum: {summary.get('ttft_ms_sum', 0.0):.6f}",
            f"- ttft_ms_avg: {summary.get('ttft_ms_avg', 0.0):.6f}",
            f"- prefill_time_ms_sum: {summary.get('prefill_time_ms_sum', 0.0):.6f}",
            f"- prefill_time_ms_avg: {summary.get('prefill_time_ms_avg', 0.0):.6f}",
            f"- prefill_vit_ms_avg: {summary.get('prefill_vit_ms_avg', 0.0):.6f}",
            f"- prefill_projector_ms_avg: {summary.get('prefill_projector_ms_avg', 0.0):.6f}",
            f"- prefill_route_ms_avg: {summary.get('prefill_route_ms_avg', 0.0):.6f}",
            f"- prefill_route_K1_ms_avg: {summary.get('prefill_route_K1_ms_avg', 0.0):.6f}",
            f"- prefill_route_K2_ms_avg: {summary.get('prefill_route_K2_ms_avg', 0.0):.6f}",
            f"- prefill_llm_ms_avg: {summary.get('prefill_llm_ms_avg', 0.0):.6f}",
            f"- decode_llm_ms_avg: {summary.get('decode_llm_ms_avg', 0.0):.6f}",
            f"- tflops_llm(prefill): {summary.get('tflops_llm(prefill)', 0.0):.6f}",
            f"- peak_gpu_memory_allocated_gib_max: {summary.get('peak_gpu_memory_allocated_gib_max', 0.0):.6f}",
            f"- kv_cache_mib_avg: {summary.get('kv_cache_mib_avg', 0.0):.6f}",
            f"- core_tflops_avg: {summary.get('core_tflops_avg', 0.0):.6f}",
        ]
        for key, value in summary.items():
            if key in {
                "num_samples",
                "total_time_ms_sum",
                "total_time_ms_avg",
                "ttft_ms_sum",
                "ttft_ms_avg",
                "prefill_time_ms_sum",
                "prefill_time_ms_avg",
                "prefill_vit_ms_avg",
                "prefill_projector_ms_avg",
                "prefill_route_ms_avg",
                "prefill_route_K1_ms_avg",
                "prefill_route_K2_ms_avg",
                "prefill_llm_ms_avg",
                "decode_llm_ms_avg",
                "tflops_llm(prefill)",
                "peak_gpu_memory_allocated_gib_max",
                "kv_cache_mib_avg",
                "core_tflops_avg",
            }:
                continue
            lines.append(f"- {key}: {value}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def _get_config_value(config, primary: str, fallback: Optional[str] = None) -> int:
    if hasattr(config, primary):
        return int(getattr(config, primary))
    if fallback and hasattr(config, fallback):
        return int(getattr(config, fallback))
    raise AttributeError(f"Config missing both {primary!r} and {fallback!r}.")


def estimate_kv_cache_bytes(
    batch_size: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    bytes_per_elem: int,
    prefill_seq_len_routed: int,
    prefill_seq_len_phase: int,
    phase_transition_layer: int,
) -> float:
    if not (1 <= phase_transition_layer < num_hidden_layers):
        raise ValueError(
            "phase_transition_layer is the number of full-length decoder blocks and must satisfy "
            f"1 <= phase_transition_layer < {num_hidden_layers}, got {phase_transition_layer}."
        )
    full_length_layers = phase_transition_layer
    pruned_length_layers = num_hidden_layers - full_length_layers
    token_sum = full_length_layers * prefill_seq_len_routed + pruned_length_layers * prefill_seq_len_phase
    return 2.0 * batch_size * num_key_value_heads * head_dim * bytes_per_elem * token_sum


def estimate_vit_flops(vision_cfg, include_minor_ops: bool = True) -> float:
    d_v = int(vision_cfg.hidden_size)
    L_v = int(vision_cfg.num_hidden_layers)
    H_v = int(vision_cfg.num_attention_heads)
    d_ff_v = int(vision_cfg.intermediate_size)
    image_size = int(vision_cfg.image_size)
    patch_size = int(vision_cfg.patch_size)
    n_patch = (image_size // patch_size) ** 2
    n_vit = n_patch + 1
    f_qkvo = 8.0 * n_vit * (d_v ** 2)
    f_attn = 4.0 * (n_vit ** 2) * d_v
    f_mlp = 4.0 * n_vit * d_v * d_ff_v
    total = L_v * (f_qkvo + f_attn + f_mlp)
    if include_minor_ops:
        total += L_v * (10.0 * n_vit * d_v + 8.0 * n_vit * d_ff_v + 5.0 * H_v * (n_vit ** 2))
    return total


def estimate_projector_flops(model, include_minor_ops: bool = True) -> float:
    projector = model.get_model().mm_projector
    n_proj = int(model.get_vision_tower().num_patches)
    if isinstance(projector, torch.nn.Linear):
        return 2.0 * n_proj * projector.in_features * projector.out_features
    if not isinstance(projector, torch.nn.Sequential):
        raise NotImplementedError(f"Unsupported projector type: {type(projector)}")
    total = 0.0
    current_dim = int(model.config.mm_hidden_size)
    for layer in projector:
        if isinstance(layer, torch.nn.Linear):
            out_dim = int(layer.out_features)
            total += 2.0 * n_proj * current_dim * out_dim
            current_dim = out_dim
        elif isinstance(layer, torch.nn.GELU):
            if include_minor_ops:
                total += 8.0 * n_proj * current_dim
        elif isinstance(layer, torch.nn.LayerNorm):
            if include_minor_ops:
                total += 5.0 * n_proj * current_dim
        else:
            raise NotImplementedError(f"Unsupported projector layer for FLOPs: {type(layer)}")
    return total


def estimate_llm_prefill_flops(config, seq_len: int, include_minor_ops: bool = True) -> float:
    L = int(config.num_hidden_layers)
    d = int(config.hidden_size)
    d_ff = int(config.intermediate_size)
    H = int(config.num_attention_heads)
    dominant = 8.0 * seq_len * (d ** 2) + 4.0 * (seq_len ** 2) * d + 4.0 * seq_len * d * d_ff
    total = L * dominant
    if include_minor_ops:
        total += L * (10.0 * seq_len * d + 8.0 * seq_len * d_ff + 5.0 * H * (seq_len ** 2))
    return total


def estimate_llm_decode_flops(config, kv_start_len: int, output_tokens: int, include_minor_ops: bool = True) -> float:
    L = int(config.num_hidden_layers)
    d = int(config.hidden_size)
    d_ff = int(config.intermediate_size)
    H = int(config.num_attention_heads)
    total = 0.0
    for t in range(1, int(output_tokens) + 1):
        s_t = kv_start_len + t - 1
        step = 8.0 * (d ** 2) + 4.0 * s_t * d + 4.0 * d * d_ff
        if include_minor_ops:
            step += 10.0 * d + 8.0 * d_ff + 5.0 * H * s_t
        total += L * step
    return total


def estimate_route_overhead_flops(hidden_size: int, routing_text_len: int, n_vis_input: int, k1: int, k2: int) -> float:
    total_budget = int(k1) + int(k2)
    return 2.0 * routing_text_len * n_vis_input * hidden_size + float(total_budget * n_vis_input)


def build_sample_efficiency_record(
    question_id: str,
    category: str,
    total_time_ms: float,
    ttft_ms: float,
    peak_gpu_memory_allocated_bytes: int,
    output_tokens: int,
    runtime_metrics: Dict[str, Any],
    model,
    phase_transition_layer: int,
) -> SampleEfficiencyRecord:
    prefill_seq_len_routed = int(runtime_metrics.get("prefill_seq_len_routed", 0))
    prefill_seq_len_phase = int(runtime_metrics.get("prefill_seq_len_phase", prefill_seq_len_routed))
    routing_text_len = int(runtime_metrics.get("routing_text_len", 0))
    n_vis_input = int(runtime_metrics.get("n_vis_input", 0))
    n_vis_routed = int(runtime_metrics.get("n_vis_routed", 0))
    n_vis_phase = int(runtime_metrics.get("n_vis_phase", 0))
    k1 = int(runtime_metrics.get("double_track_k1", 0))
    k2 = int(runtime_metrics.get("double_track_k2", 0))

    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    num_kv_heads = _get_config_value(model.config, "num_key_value_heads", "num_attention_heads")
    kv_bytes = estimate_kv_cache_bytes(
        batch_size=1,
        num_hidden_layers=int(model.config.num_hidden_layers),
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_elem=2,
        prefill_seq_len_routed=prefill_seq_len_routed,
        prefill_seq_len_phase=prefill_seq_len_phase,
        phase_transition_layer=int(phase_transition_layer),
    )
    kv_mib = kv_bytes / (2 ** 20)

    vit_flops = estimate_vit_flops(model.get_vision_tower().config)
    projector_flops = estimate_projector_flops(model)
    llm_prefill_flops = estimate_llm_prefill_flops(model.config, prefill_seq_len_routed)
    llm_decode_flops = estimate_llm_decode_flops(model.config, prefill_seq_len_phase, output_tokens)
    routing_overhead_flops = estimate_route_overhead_flops(
        hidden_size=int(model.config.hidden_size),
        routing_text_len=routing_text_len,
        n_vis_input=n_vis_input,
        k1=k1,
        k2=k2,
    )
    core_flops = vit_flops + projector_flops + llm_prefill_flops + llm_decode_flops
    core_tflops = core_flops / 1e12

    return SampleEfficiencyRecord(
        question_id=str(question_id),
        category=str(category),
        total_time_ms=float(total_time_ms),
        ttft_ms=float(ttft_ms),
        prefill_time_ms=float(runtime_metrics.get("prefill_time_ms", 0.0)),
        prefill_vit_ms=float(runtime_metrics.get("prefill_vit_ms", 0.0)),
        prefill_projector_ms=float(runtime_metrics.get("prefill_projector_ms", 0.0)),
        prefill_route_ms=float(runtime_metrics.get("prefill_route_ms", 0.0)),
        prefill_route_K1_ms=float(runtime_metrics.get("prefill_route_K1_ms", 0.0)),
        prefill_route_K2_ms=float(runtime_metrics.get("prefill_route_K2_ms", 0.0)),
        prefill_llm_ms=float(runtime_metrics.get("prefill_llm_ms", 0.0)),
        decode_llm_ms=float(runtime_metrics.get("decode_llm_ms", 0.0)),
        peak_gpu_memory_allocated_bytes=int(peak_gpu_memory_allocated_bytes),
        peak_gpu_memory_allocated_gib=float(peak_gpu_memory_allocated_bytes / (2 ** 30)),
        kv_cache_bytes=float(kv_bytes),
        kv_cache_mib=float(kv_mib),
        core_flops=float(core_flops),
        core_tflops=float(core_tflops),
        routing_overhead_flops=float(routing_overhead_flops),
        output_tokens=int(output_tokens),
        routing_text_len=routing_text_len,
        prefill_seq_len_routed=prefill_seq_len_routed,
        prefill_seq_len_phase=prefill_seq_len_phase,
        n_vis_input=n_vis_input,
        n_vis_routed=n_vis_routed,
        n_vis_phase=n_vis_phase,
        vit_flops=float(vit_flops),
        projector_flops=float(projector_flops),
        llm_prefill_flops=float(llm_prefill_flops),
        llm_decode_flops=float(llm_decode_flops),
    )


def collect_runtime_metrics(ctx: Optional[EfficiencyRuntimeContext]) -> Dict[str, Any]:
    if ctx is None:
        return {}
    torch.cuda.synchronize()
    metrics = dict(ctx.metrics)

    def elapsed(start_name: str, end_name: str) -> float:
        start_event = getattr(ctx, start_name, None)
        end_event = getattr(ctx, end_name, None)
        if start_event is None or end_event is None:
            return 0.0
        return float(start_event.elapsed_time(end_event))

    metrics["prefill_time_ms"] = elapsed("prefill_start_event", "prefill_end_event")
    metrics["prefill_vit_ms"] = elapsed("vit_start_event", "vit_end_event")
    metrics["prefill_projector_ms"] = elapsed("projector_start_event", "projector_end_event")
    metrics["prefill_route_ms"] = elapsed("route_start_event", "route_end_event")
    metrics["prefill_route_K1_ms"] = elapsed("route_k1_start_event", "route_k1_end_event")
    metrics["prefill_route_K2_ms"] = elapsed("route_k2_start_event", "route_k2_end_event")
    metrics["prefill_llm_ms"] = elapsed("llm_prefill_start_event", "llm_prefill_end_event")
    metrics["decode_llm_ms"] = elapsed("llm_decode_start_event", "llm_decode_end_event")
    return metrics
