from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch

from kv_cache_engine.compression import CompressionAgent
from kv_cache_engine.config import KVCacheXConfig
from kv_cache_engine.eviction import EvictionAgent
from kv_cache_engine.eviction.importance_model import attention_labels, build_feature_matrix
from kv_cache_engine.monitor import MonitorAgent
from kv_cache_engine.scheduler import SchedulerAgent
from kv_cache_engine.types import KVCacheState, LayerKVCache, RunSummary
from kv_cache_engine.utils import monotonic_ms, resolve_device, resolve_dtype, seed_everything


def _maybe_wrap_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache

        if isinstance(past_key_values, DynamicCache):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)
    except Exception:
        return past_key_values


def _aggregate_attention_scores(
    attentions: tuple[torch.Tensor, ...] | None,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if not attentions:
        return torch.linspace(0.2, 1.0, steps=seq_len, device=device)
    per_layer = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        layer_scores = layer_attention.mean(dim=(0, 1, 2))
        per_layer.append(layer_scores.squeeze(0))
    if not per_layer:
        return torch.linspace(0.2, 1.0, steps=seq_len, device=device)
    return torch.stack(per_layer, dim=0).mean(dim=0)


def _embedding_norms(model: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    embeddings = model.get_input_embeddings()(token_ids)
    return embeddings.norm(dim=-1).squeeze(0).detach()


def _build_prefill_state(
    model: torch.nn.Module,
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    input_ids: torch.Tensor,
    attentions: tuple[torch.Tensor, ...] | None,
) -> KVCacheState:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
    attention_scores = _aggregate_attention_scores(attentions, positions.numel(), input_ids.device)
    layers = [
        LayerKVCache(key=key.contiguous(), value=value.contiguous(), compression_method="raw")
        for key, value in past_key_values
    ]
    return KVCacheState(
        layers=layers,
        positions=positions,
        token_ids=input_ids.squeeze(0).detach().to(torch.long),
        cumulative_attention=attention_scores.detach(),
        recent_attention=attention_scores.detach(),
        embedding_norms=_embedding_norms(model, input_ids),
        step_index=0,
        metadata={"mode": "prefill"},
    )


def _update_state_from_step(
    model: torch.nn.Module,
    previous_state: KVCacheState,
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    input_token_id: torch.Tensor,
    input_position: int,
    attentions: tuple[torch.Tensor, ...] | None,
    attention_decay: float,
) -> KVCacheState:
    device = previous_state.positions.device
    layers = [
        LayerKVCache(key=key.contiguous(), value=value.contiguous(), compression_method="raw")
        for key, value in past_key_values
    ]
    positions = torch.cat(
        [
            previous_state.positions,
            torch.tensor([input_position], device=device, dtype=torch.long),
        ]
    )
    token_ids = torch.cat([previous_state.token_ids, input_token_id.reshape(-1).to(torch.long)])
    embedding_norms = torch.cat(
        [
            previous_state.embedding_norms,
            _embedding_norms(model, input_token_id.view(1, 1).to(device)).reshape(-1),
        ]
    )
    attention_scores = _aggregate_attention_scores(attentions, positions.numel(), device)
    cumulative_attention = torch.cat(
        [
            previous_state.cumulative_attention * attention_decay,
            torch.zeros(1, device=device),
        ]
    )
    cumulative_attention = cumulative_attention + attention_scores
    return KVCacheState(
        layers=layers,
        positions=positions,
        token_ids=token_ids,
        cumulative_attention=cumulative_attention.detach(),
        recent_attention=attention_scores.detach(),
        embedding_norms=embedding_norms.detach(),
        step_index=previous_state.step_index + 1,
        metadata={"mode": "decode"},
    )


def _token_nll(logits: torch.Tensor, token_id: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=token_id.view(1, 1))
    return float(-gathered.item())


def _sample_importance_view(state: KVCacheState, max_points: int = 256) -> dict:
    if state.seq_len <= max_points:
        return {
            "positions": state.positions.detach().cpu().tolist(),
            "importance_scores": state.recent_attention.detach().cpu().tolist(),
        }
    sample_indices = torch.linspace(
        0, state.seq_len - 1, steps=max_points, device=state.positions.device
    ).round().to(torch.long)
    return {
        "positions": state.positions.index_select(0, sample_indices).detach().cpu().tolist(),
        "importance_scores": state.recent_attention.index_select(0, sample_indices)
        .detach()
        .cpu()
        .tolist(),
    }


@dataclass
class InferenceArtifacts:
    summary: RunSummary
    generated_token_ids: list[int]
    prompt_token_count: int

    def to_log_dict(self) -> dict:
        payload = asdict(self.summary)
        payload["generated_token_ids"] = self.generated_token_ids
        payload["prompt_token_count"] = self.prompt_token_count
        return payload


class ModelManager:
    def __init__(self, config: KVCacheXConfig):
        self.config = config
        self.device = resolve_device(config.runtime.device)
        self.dtype = resolve_dtype(self.device, config.runtime.dtype)
        self.model = None
        self.tokenizer = None

    def load(self, model_name: str | None = None) -> tuple[torch.nn.Module, object]:
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed_everything(self.config.runtime.seed)
        target_model_name = model_name or self.config.model.name
        tokenizer = AutoTokenizer.from_pretrained(target_model_name, revision=self.config.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "revision": self.config.model.revision,
            "trust_remote_code": self.config.model.trust_remote_code,
        }
        if self.device == "cuda":
            load_kwargs["torch_dtype"] = self.dtype
        if self.config.model.attention_implementation:
            load_kwargs["attn_implementation"] = self.config.model.attention_implementation

        try:
            model = AutoModelForCausalLM.from_pretrained(target_model_name, **load_kwargs)
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(target_model_name, **load_kwargs)

        model = model.to(self.device)
        model.eval()
        if self.config.runtime.torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer


class InferenceRunner:
    def __init__(self, config: KVCacheXConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.monitor = MonitorAgent(config.monitor, config.edge)
        self.compression_agent = CompressionAgent(config.compression)
        self.eviction_agent = EvictionAgent(config.eviction)
        self.scheduler = SchedulerAgent(config.scheduler, config.edge)

    @property
    def device(self) -> str:
        return self.model_manager.device

    def run(
        self,
        prompt: str,
        workload_name: str,
        mode: str,
        max_new_tokens: int | None = None,
        forced_tokens: Iterable[int] | None = None,
        model_name: str | None = None,
    ) -> InferenceArtifacts:
        model, tokenizer = self.model_manager.load(model_name=model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        max_steps = max_new_tokens or self.config.model.max_new_tokens
        forced_list = list(forced_tokens) if forced_tokens is not None else None
        self.compression_agent.reset_runtime_cache()
        self.monitor.reset()

        if mode == "no_cache":
            return self._run_no_cache(model, tokenizer, input_ids, workload_name, max_steps, forced_list)
        if mode == "standard_cache":
            return self._run_standard_cache(
                model, tokenizer, input_ids, workload_name, max_steps, forced_list
            )
        if mode == "kvcachex":
            return self._run_kvcachex(model, tokenizer, input_ids, workload_name, max_steps, forced_list)
        raise ValueError(f"Unsupported inference mode: {mode}")

    def collect_importance_training_data(
        self,
        prompt: str,
        max_steps: int,
        model_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, tokenizer = self.model_manager.load(model_name=model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        current_input = input_ids
        past_key_values = None
        state = None
        feature_batches = []
        label_batches = []

        with torch.no_grad():
            for step in range(max_steps):
                model_kwargs = {
                    "input_ids": current_input,
                    "past_key_values": _maybe_wrap_past_key_values(past_key_values),
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": True,
                }
                if past_key_values is not None:
                    absolute_position = input_ids.shape[1] + step - 1
                    model_kwargs["position_ids"] = torch.tensor(
                        [[absolute_position]], device=current_input.device, dtype=torch.long
                    )
                outputs = model(**model_kwargs)
                if state is None:
                    state = _build_prefill_state(model, outputs.past_key_values, input_ids, outputs.attentions)
                else:
                    absolute_position = input_ids.shape[1] + step - 1
                    state = _update_state_from_step(
                        model,
                        previous_state=state,
                        past_key_values=outputs.past_key_values,
                        input_token_id=current_input.squeeze(0),
                        input_position=absolute_position,
                        attentions=outputs.attentions,
                        attention_decay=self.config.monitor.attention_decay,
                    )
                attention_scores = (
                    _aggregate_attention_scores(outputs.attentions, state.seq_len, current_input.device)
                    .detach()
                    .cpu()
                    .numpy()
                )
                feature_batches.append(build_feature_matrix(state))
                label_batches.append(
                    attention_labels(attention_scores, keep_ratio=self.config.eviction.target_keep_ratio)
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                current_input = next_token.view(1, 1)
                past_key_values = outputs.past_key_values

        if not feature_batches:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.concatenate(feature_batches, axis=0), np.concatenate(label_batches, axis=0)

    def _run_no_cache(
        self,
        model: torch.nn.Module,
        tokenizer,
        input_ids: torch.Tensor,
        workload_name: str,
        max_steps: int,
        forced_tokens: list[int] | None,
    ) -> InferenceArtifacts:
        sequence = input_ids.clone()
        generated: list[int] = []
        with torch.no_grad():
            for step in range(max_steps):
                start_ms = monotonic_ms()
                outputs = model(input_ids=sequence, use_cache=False, return_dict=True)
                latency_ms = monotonic_ms() - start_ms
                logits = outputs.logits[:, -1, :]
                predicted = torch.argmax(logits, dim=-1)
                next_token = (
                    torch.tensor([forced_tokens[step]], device=sequence.device, dtype=torch.long)
                    if forced_tokens is not None
                    else predicted
                )
                nll = _token_nll(logits, next_token)
                agreement = float((predicted == next_token).float().item())
                generated.append(int(next_token.item()))
                sequence = torch.cat([sequence, next_token.view(1, 1)], dim=1)
                self.monitor.record_step(
                    step=step,
                    latency_ms=latency_ms,
                    cache_state=None,
                    nll=nll,
                    token_agreement=agreement,
                    metadata={"sequence_length": int(sequence.shape[1])},
                )
        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        summary = self.monitor.finalize(
            workload_name=workload_name,
            mode="no_cache",
            run_kind="teacher_forced" if forced_tokens is not None else "greedy",
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=len(generated),
            output_text=output_text,
        )
        return InferenceArtifacts(summary=summary, generated_token_ids=generated, prompt_token_count=int(input_ids.shape[1]))

    def _run_standard_cache(
        self,
        model: torch.nn.Module,
        tokenizer,
        input_ids: torch.Tensor,
        workload_name: str,
        max_steps: int,
        forced_tokens: list[int] | None,
    ) -> InferenceArtifacts:
        current_input = input_ids
        past_key_values = None
        state = None
        generated: list[int] = []

        with torch.no_grad():
            for step in range(max_steps):
                start_ms = monotonic_ms()
                model_kwargs = {
                    "input_ids": current_input,
                    "past_key_values": _maybe_wrap_past_key_values(past_key_values),
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": self.config.model.use_attention_outputs and step == 0,
                }
                if past_key_values is not None:
                    absolute_position = input_ids.shape[1] + step - 1
                    model_kwargs["position_ids"] = torch.tensor(
                        [[absolute_position]], device=current_input.device, dtype=torch.long
                    )
                outputs = model(**model_kwargs)
                latency_ms = monotonic_ms() - start_ms
                logits = outputs.logits[:, -1, :]
                predicted = torch.argmax(logits, dim=-1)
                next_token = (
                    torch.tensor([forced_tokens[step]], device=current_input.device, dtype=torch.long)
                    if forced_tokens is not None
                    else predicted
                )
                nll = _token_nll(logits, next_token)
                agreement = float((predicted == next_token).float().item())

                if state is None:
                    state = _build_prefill_state(model, outputs.past_key_values, input_ids, outputs.attentions)
                else:
                    absolute_position = input_ids.shape[1] + step - 1
                    state = _update_state_from_step(
                        model,
                        previous_state=state,
                        past_key_values=outputs.past_key_values,
                        input_token_id=current_input.squeeze(0),
                        input_position=absolute_position,
                        attentions=None,
                        attention_decay=self.config.monitor.attention_decay,
                    )
                self.monitor.record_step(
                    step=step,
                    latency_ms=latency_ms,
                    cache_state=state,
                    nll=nll,
                    token_agreement=agreement,
                    metadata={
                        "cache_tokens": state.seq_len,
                        **_sample_importance_view(state),
                    },
                )
                generated.append(int(next_token.item()))
                current_input = next_token.view(1, 1)
                past_key_values = outputs.past_key_values

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        summary = self.monitor.finalize(
            workload_name=workload_name,
            mode="standard_cache",
            run_kind="teacher_forced" if forced_tokens is not None else "greedy",
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=len(generated),
            output_text=output_text,
        )
        return InferenceArtifacts(summary=summary, generated_token_ids=generated, prompt_token_count=int(input_ids.shape[1]))

    def _run_kvcachex(
        self,
        model: torch.nn.Module,
        tokenizer,
        input_ids: torch.Tensor,
        workload_name: str,
        max_steps: int,
        forced_tokens: list[int] | None,
    ) -> InferenceArtifacts:
        current_input = input_ids
        stored_state = None
        generated: list[int] = []

        with torch.no_grad():
            for step in range(max_steps):
                last_snapshot = self.monitor.snapshot()
                decode_pruned = False
                if stored_state is None:
                    plan = self.scheduler.plan(
                        state=KVCacheState(
                            layers=[],
                            positions=torch.zeros(0, device=input_ids.device, dtype=torch.long),
                            token_ids=torch.zeros(0, device=input_ids.device, dtype=torch.long),
                            cumulative_attention=torch.zeros(0, device=input_ids.device),
                            recent_attention=torch.zeros(0, device=input_ids.device),
                            embedding_norms=torch.zeros(0, device=input_ids.device),
                        ),
                        current_memory_bytes=int(last_snapshot["current_memory_bytes"]),
                        last_latency_ms=float(last_snapshot["last_latency_ms"]),
                        step_index=step,
                    )
                    model_kwargs = {
                        "input_ids": current_input,
                        "use_cache": True,
                        "return_dict": True,
                        "output_attentions": True,
                    }
                    working_state = None
                else:
                    working_state = self.compression_agent.decompress_state(stored_state)
                    plan = self.scheduler.plan(
                        state=working_state,
                        current_memory_bytes=int(last_snapshot["current_memory_bytes"]),
                        last_latency_ms=float(last_snapshot["last_latency_ms"]),
                        step_index=step,
                    )
                    if self.eviction_agent.should_prune_decode(
                        working_state, target_cache_tokens=plan.target_cache_tokens
                    ):
                        working_state = self.eviction_agent.prune_state(
                            working_state, target_cache_tokens=plan.target_cache_tokens
                        )
                        decode_pruned = True
                    past_key_values = tuple((layer.key, layer.value) for layer in working_state.layers)
                    absolute_position = input_ids.shape[1] + step - 1
                    model_kwargs = {
                        "input_ids": current_input,
                        "past_key_values": _maybe_wrap_past_key_values(past_key_values),
                        "use_cache": True,
                        "return_dict": True,
                        "output_attentions": plan.collect_attentions,
                        "position_ids": torch.tensor(
                            [[absolute_position]], device=current_input.device, dtype=torch.long
                        ),
                    }

                start_ms = monotonic_ms()
                outputs = model(**model_kwargs)
                latency_ms = monotonic_ms() - start_ms
                logits = outputs.logits[:, -1, :]
                predicted = torch.argmax(logits, dim=-1)
                next_token = (
                    torch.tensor([forced_tokens[step]], device=current_input.device, dtype=torch.long)
                    if forced_tokens is not None
                    else predicted
                )
                nll = _token_nll(logits, next_token)
                agreement = float((predicted == next_token).float().item())

                if stored_state is None:
                    raw_state = _build_prefill_state(
                        model,
                        outputs.past_key_values,
                        input_ids,
                        outputs.attentions,
                    )
                    raw_state = self.eviction_agent.prune_prefill_state(
                        raw_state,
                        attentions=outputs.attentions,
                        target_cache_tokens=plan.target_cache_tokens,
                    )
                else:
                    absolute_position = input_ids.shape[1] + step - 1
                    raw_state = _update_state_from_step(
                        model,
                        previous_state=working_state,
                        past_key_values=outputs.past_key_values,
                        input_token_id=current_input.squeeze(0),
                        input_position=absolute_position,
                        attentions=outputs.attentions if plan.collect_attentions else None,
                        attention_decay=self.config.monitor.attention_decay,
                    )
                stored_state = (
                    self.compression_agent.compress_state(
                        raw_state,
                        method_override=plan.compression_method,
                        previous_state=stored_state if stored_state is not None and not decode_pruned else None,
                    )
                    if plan.apply_compression
                    else raw_state
                )
                bandwidth_bytes = raw_state.estimated_bytes() + stored_state.estimated_bytes()
                self.monitor.record_step(
                    step=step,
                    latency_ms=latency_ms,
                    cache_state=stored_state,
                    nll=nll,
                    token_agreement=agreement,
                    bandwidth_bytes=bandwidth_bytes,
                    metadata={
                        "cache_tokens": stored_state.seq_len,
                        "target_cache_tokens": plan.target_cache_tokens,
                        "memory_pressure": plan.memory_pressure,
                        "latency_pressure": plan.latency_pressure,
                        **_sample_importance_view(stored_state),
                    },
                )
                generated.append(int(next_token.item()))
                current_input = next_token.view(1, 1)

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        summary = self.monitor.finalize(
            workload_name=workload_name,
            mode="kvcachex",
            run_kind="teacher_forced" if forced_tokens is not None else "greedy",
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=len(generated),
            output_text=output_text,
        )
        summary.extra_metrics["materialization_stats"] = self.compression_agent.materialization_stats()
        self.compression_agent.reset_runtime_cache()
        return InferenceArtifacts(summary=summary, generated_token_ids=generated, prompt_token_count=int(input_ids.shape[1]))
