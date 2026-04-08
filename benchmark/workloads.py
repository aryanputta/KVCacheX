from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kv_cache_engine.config import WorkloadConfig


@dataclass
class WorkloadSample:
    name: str
    category: str
    prompt: str
    target_tokens: int
    metadata: dict = field(default_factory=dict)


def load_prompt_seeds(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clip_prompt(tokenizer, prompt: str, max_tokens: int) -> str:
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0][:max_tokens]
    clipped = tokenizer.decode(token_ids, skip_special_tokens=True)
    while tokenizer(clipped, return_tensors="pt").input_ids.shape[1] > max_tokens:
        token_ids = tokenizer(clipped, return_tensors="pt").input_ids[0][: max_tokens - 8]
        clipped = tokenizer.decode(token_ids, skip_special_tokens=True)
    return clipped


def _expand_prompt(tokenizer, seed_text: str, target_tokens: int, header: str) -> str:
    prompt = header.strip() + "\n\n" + seed_text.strip()
    while tokenizer(prompt, return_tensors="pt").input_ids.shape[1] < target_tokens:
        prompt = prompt + "\n\n" + seed_text.strip()
    return _clip_prompt(tokenizer, prompt, target_tokens)


def build_workloads(
    tokenizer,
    config: WorkloadConfig,
    max_supported_tokens: int,
) -> list[WorkloadSample]:
    seeds = load_prompt_seeds(config.benchmark_prompts_path)
    seed_lookup = {entry["category"]: entry["text"] for entry in seeds}
    target_cap = max(256, max_supported_tokens)
    workloads: list[WorkloadSample] = []

    for target in config.long_context_targets:
        target_tokens = min(target, target_cap)
        prompt = _expand_prompt(
            tokenizer,
            seed_lookup["long_context"],
            target_tokens=target_tokens,
            header=(
                "Summarize the most persistent themes in the following reference material, "
                "then answer a factual question about a detail near the beginning and one near the end."
            ),
        )
        workloads.append(
            WorkloadSample(
                name=f"long_context_{target_tokens}",
                category="long_context",
                prompt=prompt,
                target_tokens=target_tokens,
            )
        )

    streaming_target = min(config.synthetic_prompt_tokens, target_cap)
    streaming_chunks = []
    chunk_index = 1
    while True:
        chunk = f"[chunk {chunk_index}] {seed_lookup['streaming']}"
        streaming_chunks.append(chunk)
        prompt = (
            "You are receiving an ongoing event stream. Track unresolved incidents, new escalations, "
            "and resource constraints as the updates arrive.\n\n"
            + "\n".join(streaming_chunks)
        )
        if tokenizer(prompt, return_tensors="pt").input_ids.shape[1] >= streaming_target:
            break
        chunk_index += 1
    workloads.append(
        WorkloadSample(
            name=f"streaming_{streaming_target}",
            category="streaming",
            prompt=_clip_prompt(tokenizer, prompt, streaming_target),
            target_tokens=streaming_target,
            metadata={"chunks": chunk_index},
        )
    )

    multi_turn_target = min(config.synthetic_prompt_tokens, target_cap)
    turns = []
    speaker = "User"
    turn_index = 1
    while True:
        turns.append(f"{speaker} turn {turn_index}: {seed_lookup['conversation']}")
        speaker = "Assistant" if speaker == "User" else "User"
        prompt = (
            "Continue this multi-turn technical support conversation while preserving earlier constraints, "
            "user preferences, and troubleshooting steps.\n\n"
            + "\n".join(turns)
        )
        if tokenizer(prompt, return_tensors="pt").input_ids.shape[1] >= multi_turn_target:
            break
        turn_index += 1
    workloads.append(
        WorkloadSample(
            name=f"multi_turn_{multi_turn_target}",
            category="multi_turn",
            prompt=_clip_prompt(tokenizer, prompt, multi_turn_target),
            target_tokens=multi_turn_target,
            metadata={"turns": turn_index},
        )
    )

    return workloads
