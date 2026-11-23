"""Verify the three Exp8 checkpoints on CPU."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_toml(name):
    with (HERE / name).open("rb") as f:
        return tomllib.load(f)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_checkpoint(ref, offline, export_root=None):
    from huggingface_hub import snapshot_download

    if export_root is not None and "export" in ref:
        path = (export_root / ref["export"]).resolve(strict=True)
        with (path / "model.safetensors").open("rb") as weights:
            if hashlib.file_digest(weights, "sha256").hexdigest() != ref["weight_sha256"]:
                raise ValueError(f"{path}: weights differ from the published Exp5 export")
    else:
        path = Path(
            snapshot_download(
                ref["model"],
                revision=ref["revision"],
                local_files_only=offline,
                allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "model*.safetensors"],
            )
        )
        if path.name != ref["revision"]:
            raise ValueError(f"Unexpected snapshot revision {path.name} for {ref['model']}")
    return dict(repo=ref["model"], revision=ref["revision"], path=str(path))


def resolve_models(spec, offline, export_root=None):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    models = {}
    reference = None
    for label, ref in spec["checkpoints"].items():
        models[label] = resolve_checkpoint(ref, offline, export_root)
        path = Path(models[label]["path"])
        cfg = AutoConfig.from_pretrained(path, local_files_only=True).to_dict()
        if cfg["model_type"] != "qwen2" or (cfg["hidden_size"], cfg["num_hidden_layers"]) != (896, 24):
            raise ValueError(f"{label}: unexpected Qwen2.5-0.5B architecture")
        shapes = {}
        locations = {}
        index = path / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        for name in shards:
            with safe_open(str(path / name), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key in shapes or (weight_map is not None and weight_map.get(key) != name):
                        raise ValueError(f"{label}: unexpected tensor {key} in {name}")
                    tensor = shard.get_slice(key)
                    shapes[key] = (tensor.get_shape(), tensor.get_dtype())
                    locations[key] = path / name
        if weight_map is not None and shapes.keys() != weight_map.keys():
            raise ValueError(f"{label}: checkpoint index and shards disagree")
        if cfg["tie_word_embeddings"] and "lm_head.weight" in shapes:
            # HF can omit the tied output head; the Exp5 exporter stores both copies.
            with (
                safe_open(str(locations["lm_head.weight"]), framework="pt") as head,
                safe_open(str(locations["model.embed_tokens.weight"]), framework="pt") as embed,
            ):
                if not torch.equal(head.get_tensor("lm_head.weight"), embed.get_tensor("model.embed_tokens.weight")):
                    raise ValueError(f"{label}: tied embedding and output weights differ")
            del shapes["lm_head.weight"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        signature = (
            {
                k: cfg.get(k)
                for k in (
                    "vocab_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                    "rope_parameters",
                    "layer_types",
                    "max_position_embeddings",
                    "rms_norm_eps",
                    "tie_word_embeddings",
                )
            },
            shapes,
            tokenizer.get_vocab(),
            tokenizer.chat_template,
            tokenizer.eos_token_id,
        )
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError(f"{label}: checkpoint geometry or tokenizer differs from the base")
    if len({(m["repo"], m["revision"]) for m in models.values()}) != len(models):
        raise ValueError("Two checkpoints resolved to the same snapshot")
    return models


def check_cpu_forwards(models):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tokenizer = AutoTokenizer.from_pretrained(models["base"]["path"], local_files_only=True)
    message = {"role": "user", "content": "Sort these letters alphabetically: c a b"}
    text = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    for label, snapshot in models.items():
        model, info = AutoModelForCausalLM.from_pretrained(
            snapshot["path"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        issues = {key: value for key, value in info.items() if value}
        if issues:
            raise ValueError(f"{label}: checkpoint load issues: {issues}")
        with torch.inference_mode():
            logits = model.eval()(**inputs, use_cache=False, logits_to_keep=1).logits
        if logits.device.type != "cpu" or tuple(logits.shape) != (1, 1, model.config.vocab_size):
            raise ValueError(f"{label}: unexpected logits on {logits.device} with shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"{label}: nonfinite CPU logits")
        print(f"{label} {snapshot['revision']}: loaded on CPU, {logits.shape[-1]} finite logits", flush=True)
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, help="Use the two checksum-verified Exp5 model exports here")
    parser.add_argument("--offline", action="store_true", help="Resolve checkpoints from existing caches")
    args = parser.parse_args()
    models = resolve_models(load_toml("sweep.toml"), args.offline, args.export_root)
    check_cpu_forwards(models)


if __name__ == "__main__":
    main()
