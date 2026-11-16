# Colocated rollout for `run_qwen3_5_torchtitan.sh` on one TPU v7x host

**Status: needs open pull requests; tested in one passing 10-update run on TPU v7x.** This page describes how to run
[`run_qwen3_5_torchtitan.sh`](https://github.com/verl-project/verl/blob/1ec15fed25f3fbdb79b15b8e4213292d9038e2d3/examples/tpu/grpo/run_qwen3_5_torchtitan.sh)
of verl main with the vLLM rollout inside the trainer processes, on one TPU v7x host (4 chips) instead of a trainer slice
plus a rollout slice. It needs open pull requests of verl-hardware-plugin:
[#143](https://github.com/verl-project/verl-hardware-plugin/pull/143) to
[#146](https://github.com/verl-project/verl-hardware-plugin/pull/146) (design and evidence in
[#142](https://github.com/verl-project/verl-hardware-plugin/issues/142)) and the fix in
[#157](https://github.com/verl-project/verl-hardware-plugin/pull/157). The [command](#command) below is the one we ran
on 2026-11-15 with Qwen3.5-0.8B for 10 updates: the job completed, and the run passed every check of its pre-registered
read-out (see [What the run showed](#what-the-run-showed); two earlier runs are under [Earlier runs](#earlier-runs)).
Longer runs, other models and more than one host have not been tested with this command.

This page is for verl main's script, not for the tagged stack in [`qwen3_5_v7x_stack.md`](qwen3_5_v7x_stack.md): the
verl-hardware-plugin of that stack has no in-process rollout. The script, `Dockerfile` and `README.md` of verl main
differ from the files of the same names next to this page, so the links below to those three files point at verl main
`1ec15fed`, and the commands run verl main's copies.

## How it works

On TPU, a TensorCore can be opened by one process only, and vllm-torchtpu has no sleep mode, so verl's usual
colocation, which keeps the rollout in separate processes, cannot share chips with the trainer. With
`VERL_TPU_ROLLOUT_INPROC=1`, each trainer rank (one process per TensorCore, 8 per v7x host) also builds one vLLM
engine with tensor parallelism 1 in its own process, on the trainer's TPU runtime. verl's synchronous V1 trainer then
alternates generation and training on the same chips; the weight update is an in-place copy inside each process. The
trainer and the engine both stay resident, so the engine's KV cache gets an explicit byte budget. Every weight update
is checked while the run goes on (see [What stops a run](#what-stops-a-run)). Below, update s is the trainer's s-th
optimizer step, and sync s is the weight update to the engines that follows it; sync 0 installs the trainer's initial
weights, before update 1.

## Software

| Component | Version | Notes |
|---|---|---|
| verl | main [`1ec15fed`](https://github.com/verl-project/verl/commit/1ec15fed25f3fbdb79b15b8e4213292d9038e2d3) | This script, unchanged |
| verl-hardware-plugin | `ac817788d74391c9a47fb527328bf37e72307004` (tree `e43f8533`): the head of #146, `61b1c384`, with the two commits of #157, `dc83cb90` and `623d8e36`, cherry-picked on top; plugin main under them is [`16be0a49`](https://github.com/verl-project/verl-hardware-plugin/commit/16be0a491de807bbd9767f7894aa21feb41e1b52) | #143 is the feature; [#144](https://github.com/verl-project/verl-hardware-plugin/pull/144) stops the run when an engine function recompiles after warm-up; [#145](https://github.com/verl-project/verl-hardware-plugin/pull/145) is a faster output path (it turns itself off on any vLLM but 0.32.0); #146 adds first-token times to the timeline; #157 accepts a weight sync after a zero-gradient update as a no-op instead of failing the staleness check. [#151](https://github.com/verl-project/verl-hardware-plugin/pull/151) to [#156](https://github.com/verl-project/verl-hardware-plugin/pull/156) are #143's code split into six parts for review (#156's tree is #143's head); #157 is stacked on #156 |
| Image | [`Dockerfile`](https://github.com/verl-project/verl/blob/1ec15fed25f3fbdb79b15b8e4213292d9038e2d3/examples/tpu/grpo/Dockerfile) of verl main, unchanged | Base `rayproject/ray:2.56.0-py312`; vLLM 0.32.0, vllm-torchtpu `732c4346`, TorchTitan `109bd618`, torch 2.13.0, torch-tpu 0.2.0, tpu-sync-torch 0.0.1.dev20261012084241, jax/jaxlib 0.12.0.1, libtpu 0.0.50, tokamax 0.0.17, TransferQueue 0.1.8. At build, the in-process rollout checks the vLLM and vllm-torchtpu code it hooks into and refuses to build if a hook that correctness depends on has changed (see [What the build refuses](#what-the-build-refuses)), so keep these pins |

Get verl main and build the plugin tree of `ac817788` from the heads of the two pull requests (`--no-commit` needs no
git identity, and `git write-tree` checks the result):

```bash
git clone https://github.com/verl-project/verl.git
git -C verl checkout 1ec15fed25f3fbdb79b15b8e4213292d9038e2d3
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
git fetch origin pull/146/head pull/157/head
git checkout --detach 61b1c384887ca7f9adc73da983d33d6a99c03f3d
git cherry-pick --no-commit dc83cb90e0e5e7934569fefb465f60621af8edfe 623d8e3649fc5d37a244d20cf426e8cc40e474e3
git write-tree   # prints e43f8533a68a65a9b55eddb04ffeadf8e8409b47, the tree of ac817788
cd ../verl
```

Run the commands below from the root of `verl`; each job ships `../verl-hardware-plugin` with it. The commands above use
the heads of #146 and #157 as of 2026-11-15. If `git checkout` fails or `git write-tree` prints another tree, one of
them has moved since: check out `ac817788` instead, from branch `feat/qwen35/inproc-p5-stale-noop` of the
verl-hardware-plugin fork that these pull requests come from.

Build the image as in step 2 of verl main's
[README.md](https://github.com/verl-project/verl/blob/1ec15fed25f3fbdb79b15b8e4213292d9038e2d3/examples/tpu/grpo/README.md).
Our run did not use a prebuilt image: it ran the Dockerfile's steps, in order, at container start on the same base image
(index digest `sha256:2951c07de396a8b746f9c678b52c6e2282e614e00f80e6846a9ccd12945ae6b0`), then checked the installed
versions against the pins above. To build on the same base, check that `rayproject/ray:2.56.0-py312` still resolves to
that digest (`docker buildx imagetools inspect rayproject/ray:2.56.0-py312`), or pin the `FROM` line to
`rayproject/ray:2.56.0-py312@sha256:2951c07de396a8b746f9c678b52c6e2282e614e00f80e6846a9ccd12945ae6b0`. The Dockerfile
does not pin every package, so a later build can resolve newer releases; ours resolved on 2026-11-15, with transformers
5.19.0 (the Dockerfile allows `>=5.18.0,<5.20.0`).

Prepare GSM8K as in step 3 of the README. We used `openai/gsm8k` at revision `740312add88f781978c0658806c59bc2815b9866`,
preprocessed with `datasets` 5.1.0 and `pyarrow` 26.0.1; [GSM8K data](qwen3_5_v7x_stack.md#gsm8k-data) shows how to pin
the revision. The checkpoint is below; [Host and container](#host-and-container) says where both go.

**Precision patches.** verl-hardware-plugin main applies numerical patches to the server rollout (FP32 rotary
embedding and SiLU, and the opt-in switches `VERL_TPU_VLLM_RMSNORM_CUDA_ORDER` and `VERL_TPU_VLLM_ATTN_FP32_STATS`,
from [verl-project/verl-hardware-plugin#107](https://github.com/verl-project/verl-hardware-plugin/pull/107)). The
in-process engine does not apply them (see "Not in this PR" in
[#154](https://github.com/verl-project/verl-hardware-plugin/pull/154)'s description), so with
`VERL_TPU_ROLLOUT_INPROC=1` the rollout runs without them, and those two switches have no effect on it. (Without
`VERL_TPU_ROLLOUT_INPROC=1`, the server rollout still applies them.) Applying them in process changes the engine's TPU
numerics and needs its own TPU run; that is a follow-up.

## Checkpoint and LM head

`MODEL_PATH` must be a text-only Qwen3.5 checkpoint (`Qwen3_5ForCausalLM`, `model_type` `qwen3_5_text`), as for the
disaggregated script. The in-process build accepts only the model layouts its tests cover; for Qwen3.5 that is the
dense text-only model. A multimodal checkpoint class and the Qwen3.5 MoE models are refused.

TorchTitan's Qwen3.5-0.8B trains a separate LM head, while the Hugging Face checkpoint ties the LM head to the input
embedding. The trainer and the in-process engine must agree on that.

**Untied checkpoint (what we ran).** We ran the text-only checkpoint of
[`qwen3_5_v7x_stack.md`](qwen3_5_v7x_stack.md#text-only-checkpoint), made from
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision
`2fc06364715b967f1860aea9cf38778875588b17`: `tie_word_embeddings=false`, and an explicit `lm_head.weight`, a copy of
the embedding, beside the 320 language-model tensors. Every tensor keeps the source's dtype: 285 tensors in bfloat16,
`lm_head.weight` among them, and 36 in float32. The trainer trains its own LM head and the engine holds a separate one,
which every weight update overwrites. With Python, `torch` and `safetensors` (a CPU is enough), save this as
`make_text_untied.py`:

```python
# make_text_untied.py SRC DST: the text-only checkpoint with an untied LM head
import json, shutil, sys
from collections import Counter
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
prefix, weights = "model.language_model.", {}
index = json.loads((src / "model.safetensors.index.json").read_text())
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(str(src / shard), framework="pt") as f:
        for name in f.keys():
            if name.startswith(prefix):
                weights["model." + name[len(prefix):]] = f.get_tensor(name)  # no dtype cast
            elif not name.startswith(("model.visual.", "mtp.")):
                raise SystemExit(f"unexpected tensor {name}")
assert len(weights) == 320, len(weights)
weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
dst.mkdir(parents=True)
shard = "model-00001-of-00001.safetensors"
save_file(weights, str(dst / shard), metadata={"format": "pt"})
total = sum(t.numel() * t.element_size() for t in weights.values())
(dst / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"total_size": total}, "weight_map": {k: shard for k in sorted(weights)}}, indent=2) + "\n")
config = json.loads((src / "config.json").read_text())["text_config"]
config.update(architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", tie_word_embeddings=False)
(dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "LICENSE"):
    shutil.copyfile(src / name, dst / name)
print(len(weights), "tensors", dict(Counter(str(t.dtype) for t in weights.values())))
```

Then download the pinned revision and convert it, outside the `verl` checkout:

```bash
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir /tmp/Qwen3.5-0.8B
python3 make_text_untied.py /tmp/Qwen3.5-0.8B /tmp/Qwen3.5-0.8B-text-untied
```

It prints `321 tensors {'torch.bfloat16': 285, 'torch.float32': 36}`. We checked on CPU that it writes the checkpoint of
our run byte for byte (all 9 files). The shard's file name contains a number because TorchTitan's state-dict adapter
reads the shard number from the file names in the index. This page calls the result `Qwen3.5-0.8B-text-untied`, and the
command below reads it as `MODEL_PATH=/data/assets/hf/Qwen3.5-0.8B-text-untied`: in the README's layout, copy it to
`assets/hf/` in the bucket, as step 3 of the README does for the other models. Without `MODEL_PATH`, the script reads
`/data/assets/hf/Qwen3.5-0.8B`; do not put the Hugging Face checkpoint there, because its multimodal, tied layout is
refused.

**Tied checkpoint** (`tie_word_embeddings=true`, no `lm_head.weight`): needs
[#131](https://github.com/verl-project/verl-hardware-plugin/pull/131) and
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tie_embeddings=True`, so that the trainer trains one tensor for
both. Without it, the trainer's LM head moves away from the embedding after the first update, and the in-process weight
check should stop the run at the next verified update, because the tied engine holds one tensor for both (we expect
this from the code and a CPU test of a mismatched LM head; we have not run this case on TPU). #131 is stacked on
[#130](https://github.com/verl-project/verl-hardware-plugin/pull/130), and as of 2026-11-15 GitHub reports both as
conflicting with plugin main, so use the untied checkpoint.

## Host and container

You need a GKE cluster with a TPU v7x node pool that can schedule one `2x2x1` slice (1 host, 4 chips), a registry for
the image, and the tools listed at the top of verl main's README.md. Of the README's steps, this page uses step 2 (the
image), step 3 (GSM8K, and the model with the changes above) and the `RUNTIME_ENV` of step 5; it replaces step 1 (the
code) and step 4 (the Ray cluster).

Our run used one GKE pod on the TPU host, with:

- `google.com/tpu: "4"` as request and limit (the pod takes the 4 chips; the Ray node below offers their 8 TensorCores
  as `TPU: 8`);
- the node selectors `cloud.google.com/gke-tpu-accelerator: tpu7x` and `cloud.google.com/gke-tpu-topology: 2x2x1`, and a
  toleration for the `google.com/tpu` taint;
- the host's network;
- a memory-backed volume of 64 GiB at `/dev/shm`, so that Ray's object store (32 GB below) fits in it;
- no memory request or limit;
- `/data` on the pod's local disk: before starting Ray, we wrote the converted checkpoint and the GSM8K files there, at
  the paths the script reads.

GKE set the pod's `TPU_*` variables (for example `TPU_ACCELERATOR_TYPE=tpu7x-8`). Mounting the bucket at `/data`
through Cloud Storage FUSE, as the README's `RayCluster` does, should work the same way; we have not run it.

## Ray cluster

One TPU v7x `2x2x1` slice: 1 host, 4 chips, 8 TensorCores. The host joins Ray with the resources
`{"TPU": 8, "tpu-group-0": 1}` (see the [Ray cluster layout](qwen3_5_v7x_stack.md#ray-cluster-layout) of the stack
page; here there is no rollout slice). The synchronous trainer uses only the trainer's resource pool (`trainer.nnodes` x
`trainer.n_gpus_per_node`, the script's `NNODES_TRAINER` and `N_CHIPS_TRAINER`, default 1 x 8); no rollout pool is
created, so `NNODES_ROLLOUT` and `N_CHIPS_ROLLOUT` are not used.

We ran Ray's head on the TPU host itself, as a single Ray node, in the container described above:

```bash
RAY_memory_monitor_refresh_ms=0 RAY_memory_usage_threshold=0.99 TPU_NAME=tpu-group-0 \
  ray start --head --resources='{"TPU": 8, "tpu-group-0": 1}' --object-store-memory=32000000000
ray status   # Resources: 0.0/8.0 TPU and 0.0/1.0 tpu-group-0
export RAY_ADDRESS=http://127.0.0.1:8265
```

The two `RAY_memory_*` variables turn off Ray's memory monitor on this node, as verl main's `ray-tpu-v6e-2slice.yaml`
does for its TPU workers. The script exports them too, but inside the job, after the node has started, so they do not
reach the node. `TPU_NAME=tpu-group-0` is kept as in our run; the `tpu-group-0` resource itself comes from
`--resources`. Run `ray status` before you set `RAY_ADDRESS`: `ray status` takes the cluster's address, not the
dashboard URL. `RAY_ADDRESS` is the node's dashboard at Ray's default port; we submitted the job from inside the same
container, from the root of the `verl` checkout there. Our own `ray start` line also fixed Ray's ports and the node IP
address (the pod used the host's network) and passed `--disable-usage-stats`; none of this changes the job. Submitting
from another machine through `kubectl port-forward` to port 8265 of the pod, as steps 4 and 5 of the README do for its
head service, should also work; we have not run it.

The README's KubeRay layout, with a separate CPU head pod, should also work if its TPU worker group has one replica with
the v7x values of section 8 of the README (`numOfHosts: 1`, `google.com/tpu: "4"`, `gke-tpu-accelerator: tpu7x`,
`gke-tpu-topology: 2x2x1`), joins Ray with the resources above (`ray status` shows them) and has a 64 GiB `/dev/shm`. We
have not run it.

## Settings: what changes against the disaggregated script

Environment of the script (set with `env` in the command):

| Variable | Disaggregated (script default) | Colocated | Why |
|---|---|---|---|
| `VERL_TPU_ROLLOUT_INPROC` | unset | `1` | Turns the in-process rollout on; only the exact value `1` does. verl-hardware-plugin forwards `VERL_TPU_*` variables to every worker |
| `MODEL_NAME` | `Qwen3.5-35B-A3B` | `Qwen3.5-0.8B` | Dense text-only model (see above) |
| `MODEL_PATH` | `/data/assets/hf/${MODEL_NAME}` | `/data/assets/hf/Qwen3.5-0.8B-text-untied` | The untied checkpoint (see above) |
| `ROLLOUT_TENSOR_PARALLEL_SIZE` | all rollout chips (8) | `1` | One engine per trainer rank; any other tensor, data or pipeline parallel size is refused |
| `MAX_NUM_SEQS` | 256 | `128` | The KV budget below must hold `max_num_seqs` requests of `max_model_len` tokens, or the build refuses |
| `TRAINER_EXTRA_LIBTPU_INIT_ARGS` | unset | `--xla_tpu_use_dynamic_smem_negotiation=true` | vllm-torchtpu would add this flag when it is imported, but in process the trainer starts the TPU runtime first, so the trainer's flags must already carry it; the build refuses when they do not. The script appends it to `VERL_TPU_TRAINER_LIBTPU_INIT_ARGS`, which the plugin joins to `LIBTPU_INIT_ARGS` with a space. Write any further flags space-separated, as the script does |
| `USE_KL_LOSS` | `True` | `False` | We have not run this recipe with a KL loss. A KL loss is not refused, but the reference model then also lives in each trainer process |
| `NUM_WARMUP_BATCHES` | 2 | not used | Only the asynchronous trainer reads it |

Arguments appended to the script (it passes them on to `verl.trainer.main_ppo`; a later value of a key replaces the
script's):

| Argument | Script default | Why |
|---|---|---|
| `trainer.v1.trainer_mode=sync` | `separate_async` | The in-process rollout serves verl's synchronous trainer only; its sampler raises in any other mode |
| `actor_rollout_ref.hybrid_engine=True` | `False` | The trainer workers host the rollout |
| `actor_rollout_ref.rollout.free_cache_engine=False` | `True` (verl's default) | There is no sleep mode on TPU; with `True`, verl adds a barrier after the weight update, and the per-sync staleness probe then only logs its verdict instead of stopping the run |
| `actor_rollout_ref.rollout.checkpoint_engine.backend=naive` | `tpu-sync` | Weight updates go through verl's in-process `naive` path |
| `++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=10737418240` | unset | The engine's KV budget (10 GiB per TensorCore); required, because the engine shares device memory with the trainer |
| `++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=null` | `${SEED}` | The script passes `SEED` as an explicit vLLM seed, which replaces verl's per-replica seed (`SEED + replica rank`). The in-process build checks that each engine has the per-replica seed, so on every rank but rank 0 it would fail. `null` removes the explicit seed; in our runs each engine then had its per-replica seed, and the check passed on all 8 ranks |
| `trainer.v1.sampler.custom_sampler.path=pkg://verl_hardware_plugin.accelerators.tpu.rollout.inproc.sampler` and `trainer.v1.sampler.custom_sampler.name=FastPollReplayBufferSync` | unset | The sampler that stops a step whose rollout lost or aborted requests before the update |
| `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.model_kernels=none` and `+actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.model_kernels=none` | unset (`auto`) | `auto` turns on `moe` and `lce` (the fused linear cross-entropy kernel) because TorchTitan `109bd618` has `_use_tpu_moe_kernels`, and `kda` if the installed Tokamax also provides its KDA op; it does not check whether the model is a MoE. On the tagged stack's TorchTitan, `auto` fails at initialization for a dense model (see [`qwen3_5_v7x_stack.md`](qwen3_5_v7x_stack.md#dense-qwen35-models)); we have not tried `auto` with `109bd618` |
| `++data.apply_chat_template_kwargs.enable_thinking=False` | `True` (the script adds the key with `+`) | Thinking off, so that the responses of Qwen3.5-0.8B finish within the script's 2,048-token budget. In our first run, with thinking on, every response stopped at that limit and validation at update 0 scored 0 of 128 (see [Earlier runs](#earlier-runs)). The key needs `++`: a second `+` on a key the script already added fails Hydra's composition |

Length and logging of our run, on top of the colocated settings:

| Setting | Script default | Our run | Why |
|---|---|---|---|
| `TOTAL_TRAINING_STEPS` | 100 | `10` | Run length |
| `TEST_FREQ` | 20 | `10` | Validation at updates 0 and 10, on the script's default 128 test problems (`VAL_MAX_SAMPLES`) |
| `VERL_TPU_INPROC_LAST_STEP` | unset | `10` | The run's last update; its weight sync is verified too. Set it to `TOTAL_TRAINING_STEPS` |
| `"VERL_LOGGING_LEVEL": "INFO"` in `RUNTIME_ENV` | unset | `INFO` | Shows the in-process rollout's per-rank build and weight-update lines; refusals and failed checks are logged as warnings either way |

The script already sets `actor_rollout_ref.rollout.enable_prefix_caching=False`, which the in-process rollout requires,
and `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`; keep both.

The build checks the KV budget against `MAX_NUM_SEQS` requests of the script's `max_model_len` (by default 512 prompt
plus 2,048 response tokens) on every rank, and logs the budget's capacity per rank (`inproc kv capacity rank <r>`);
raise the budget or lower `MAX_NUM_SEQS` if it refuses. With the settings above, the check passed on all 8 ranks in our
run. The trainer's memory comes on top of the engine's resident KV cache and weights.

## Command

`RUNTIME_ENV` is the one of step 5 of verl main's
[README.md](https://github.com/verl-project/verl/blob/1ec15fed25f3fbdb79b15b8e4213292d9038e2d3/examples/tpu/grpo/README.md),
with `"VERL_LOGGING_LEVEL": "INFO"` added to its `env_vars`. This is the command of our run:

```bash
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1", "VERL_LOGGING_LEVEL": "INFO"}}'

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-0.8B MODEL_PATH=/data/assets/hf/Qwen3.5-0.8B-text-untied \
  VERL_TPU_ROLLOUT_INPROC=1 ROLLOUT_TENSOR_PARALLEL_SIZE=1 MAX_NUM_SEQS=128 \
  USE_KL_LOSS=False TRAINER_EXTRA_LIBTPU_INIT_ARGS="--xla_tpu_use_dynamic_smem_negotiation=true" \
  TOTAL_TRAINING_STEPS=10 TEST_FREQ=10 VERL_TPU_INPROC_LAST_STEP=10 \
  bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh \
  trainer.v1.trainer_mode=sync \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=10737418240 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=null \
  trainer.v1.sampler.custom_sampler.path=pkg://verl_hardware_plugin.accelerators.tpu.rollout.inproc.sampler \
  trainer.v1.sampler.custom_sampler.name=FastPollReplayBufferSync \
  +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.model_kernels=none \
  +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.model_kernels=none \
  ++data.apply_chat_template_kwargs.enable_thinking=False
```

For a longer run, raise `TOTAL_TRAINING_STEPS` and set `VERL_TPU_INPROC_LAST_STEP` to the same value; `TEST_FREQ` sets
how often validation runs. We have run this command only as written, for 10 updates. With a tied checkpoint and #131,
also pass `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tie_embeddings=True` (not run).

Before the run, we checked on CPU that verl main's Hydra configuration composes with these arguments after the script's
own, and that the result has the values above (sync mode, the hybrid engine, rollout tensor parallel size 1, prefix
caching off, `free_cache_engine=False`, the KV budget, no explicit engine seed, the sampler, `max_num_seqs` 128,
thinking off). We also checked on CPU that the trainer flags the script builds from these settings contain the flag
that the build checks for.

## Checking the run

Follow the job with `ray job logs -f <JOB_ID>`. With INFO logging, our run showed:

- at build, on each rank: `inproc kv capacity rank <r>: {...}` (a warning instead when the check fails);
- at the verified syncs, 0, 1, 5 and 10 with the command above, on each rank:
  `inproc parity step <s> rank <r>: PARITY EXACT`;
- from sync 1 on: `inproc stale probe step <s> rank <r>: ... changed <k>/<n> vs step <s-1>`;
- one `step:<n> - ...` line of verl's metrics per update, and `step:0` with the first validation. The validation score
  is `val-core/openai/gsm8k/acc/mean@1`, and the training HBM peak is the largest `actor/perf/max_memory_allocated_gb`;
- at the end, `Job '<JOB_ID>' succeeded` from `ray job submit`; `ray job status <JOB_ID>` prints the same line.

Ray folds lines of the same pattern (ignoring words with numbers in them) from different processes into one line
marked `[repeated <n>x across cluster]`, so the job log does not show every rank's line: for each verified sync, ours
showed one rank's parity line, then the same line marked `[repeated 7x across cluster]`. Every rank's lines are in the
`worker-*.err` files under `/tmp/ray/session_latest/logs` on the node. Some early updates, update 1 above all, took
several times the median step time; a slow early update is not by itself a failure.

```bash
ray job logs <JOB_ID> | grep -E 'PARITY|stale probe|kv capacity|val-core'
```

## What the run showed

On 2026-11-15 we ran the command above on one TPU v7x `2x2x1` host, with the code at `ac817788`, the image's packages
(see [Software](#software)) and the untied checkpoint. The job completed all 10 updates, and the run passed every check
of its pre-registered read-out, a list of pass and fail checks on the run's logs that we fixed before the run. Among
them:

- at the checked syncs 0, 1, 5 and 10, the installed weights matched the trainer's export exactly on all 8 trainer
  ranks;
- at every sync from 1 to 10, on all 8 ranks, the staleness check's sample of the engine's weights changed after an
  update with a nonzero gradient norm;
- no engine function recompiled after warm-up (#144's check).

Validation on the script's 128 GSM8K test problems: 63 correct at update 0 and 75 at update 10
(`val-core/openai/gsm8k/acc/mean@1` 0.4921875 and 0.5859375). Training HBM peak (the highest per-step
`actor/perf/max_memory_allocated_gb` of the run): 24.7 GiB per TensorCore.

Not part of the read-out's checks: the median step time was 15.5 s and the median weight update 0.21 s (verl's
`timing_s/step` and `timing_s/update_weights`, steps 4-8), at the script's 64 responses per step (16 prompts, 4
responses each). These are the numbers of our
[comment on #142](https://github.com/verl-project/verl-hardware-plugin/issues/142#issuecomment-6909605382), which also
has runs of the same in-process code (at #146's head) in our own training setup rather than this script.

## Earlier runs

The command above is the third of our runs on the example script, each on one 4-chip TPU v7x host for 10 planned
updates:

1. **2026-11-13**: the code at #146's head (`61b1c384`) and the command above without the thinking-off argument. The
   run stopped at update 1: the staleness check found its sample of the engine weights unchanged after the first
   optimizer step, while parity was exact at syncs 0 and 1. Every response had stopped at the 2,048-token limit, and
   validation at update 0 scored 0 of 128. We infer, but did not observe (update 1's rewards and gradient norm were not
   logged before the stop), that every reward at update 1 was 0 and the gradient therefore exactly 0. With every GRPO
   advantage at 0, AdamW's first step leaves the weights bit-identical: its update is 0 while both moment estimates are
   still 0, and its weight decay multiplies the weights by `1 - lr * weight_decay`, which is exactly 1 in fp32 at a
   small learning rate such as the start of a warm-up. So nothing new reached the engine. #157 accepts such a sync as a
   no-op; turning thinking off lets the model answer within the budget.
2. **2026-11-14**: the code at `ac817788` and the command above. It trained all 10 updates, but our harness's final
   job-status check marked the run as failed.
3. **2026-11-15**: the same code and command, with only that harness check fixed. It passed (above).

#157's no-op path did not come up in the second and third runs: every sync from 1 to 10 followed an update with a
nonzero gradient norm. The CPU tests in #157's description cover it.

## What the build refuses

The in-process rollout checks these before or while building the engine on every rank, and stops the run with a
message that names the problem:

| Setting | Result |
|---|---|
| Rollout tensor, data or pipeline parallel size other than 1 (verl's default rollout tensor parallel size is 2; the script's default is all rollout chips) | Refused |
| A model layout its tests do not cover (for example a Qwen3.5 MoE model or a multimodal checkpoint class), or quantized weights | Refused |
| `actor_rollout_ref.rollout.enable_prefix_caching=True` | Refused: the cache would serve KV computed under the previous weights after an update |
| No `engine_kwargs.vllm.kv_cache_memory_bytes`, or a budget that cannot hold `max_num_seqs` requests of `max_model_len` tokens | Refused (the capacity check only logs with `VERL_TPU_INPROC_INTEGRITY=report`) |
| `VERL_FULL_DETERMINISM` set to anything other than empty or `0`, or `actor_rollout_ref.rollout.full_determinism=True` | Refused: it would reseed the trainer's random state in the same process |
| The trainer's libtpu flags without the `--xla_tpu_use_dynamic_smem_negotiation` flag | Refused |
| An explicit `engine_kwargs.vllm.seed` (the script's default) | The engine check fails on every rank but rank 0, and the build stops on all ranks |
| `trainer.v1.trainer_mode` other than `sync` | The sampler raises |
| vLLM, vllm-torchtpu or tpu-sync code that changed at a point the in-process rollout depends on | Refused, naming the point |
| A malformed `VERL_TPU_INPROC_*` value | Refused; an unknown `VERL_TPU_INPROC_*` name is logged and not read |

Not refused, but not run with this recipe: more than one host, a KL loss (reference model in the trainer processes),
and `free_cache_engine=True` (see above).

## What stops a run

After the build, these checks stop the run with a message that names the problem:

- **Weight parity.** At the verified syncs (updates 0 and 1, every `VERL_TPU_INPROC_PARITY_EVERY` updates, the update
  in `VERL_TPU_INPROC_LAST_STEP`, the first sync of a process, and every sync until one has passed), every installed
  weight is compared bit for bit with what the trainer sent, and every engine weight must have been written. Every
  later sync must send the same tensor names, shapes and dtypes as the first verified sync that passed.
- **Staleness.** After every sync, a sample of the engine's weights must have changed if the trainer took an optimizer
  step since the previous check (`INTEGRITY FAILED ... unchanged since sync <n>`, or `STALE WEIGHTS`). Unchanged weights
  pass when the optimizer skipped every step in between (a non-finite gradient norm), and, with #157, as a no-op sync
  when the trainer's export of the same tensors is also unchanged and every optimizer step since the previous check had
  a gradient norm of exactly 0.
- **Recompilation** (#144). After warm-up, an engine function that recompiles more often than #144 allows (for new input
  shapes up to its precompiled count, up to 8 times for any other reason), a fresh compile of engine code, or a
  recompile at dynamo's limit stops the run.
- **Lost requests.** A step whose rollout lost or aborted requests, or whose batch is missing responses, fails on every
  rank before `update_actor`.

`VERL_TPU_INPROC_INTEGRITY=report` turns the staleness and KV-capacity verdicts into log lines, and
`VERL_TPU_INPROC_RECOMPILE_GUARD=report` does the same for recompilation; a parity, coverage or export failure always
stops the run.

Tuning switches, all optional: `VERL_TPU_INPROC_PARITY_EVERY` (default 5; `0` keeps only the fixed verified syncs),
`VERL_TPU_INPROC_LAST_STEP` (the run's last update, also verified; set it to `TOTAL_TRAINING_STEPS`),
`VERL_TPU_INPROC_INTEGRITY` (`fatal` or `report`), `VERL_TPU_INPROC_GC_RSS_GIB` (default 8; `0` turns it off),
`VERL_TPU_INPROC_RECOMPILE_GUARD` (#144: `fatal` or `report`) and `VERL_TPU_INPROC_LAZY_OUTPUT` (#145: `0` turns the
faster output path off).

## Not tested

We have not run this recipe with:

- more than 10 updates, more than one host, or another model;
- a KL loss, `free_cache_engine=True`, or thinking on (the script's default) with #157's fix in place;
- #157's no-op path on TPU: it did not come up in our runs;
- the tied checkpoint with #131 (see above: #131 conflicts with plugin main);
- the precision follow-up described above;
- vllm-torchtpu main with vLLM 0.33.0, or TorchTitan main;
- a Ray head on a separate CPU pod, or an image built from the Dockerfile: our run started Ray on the TPU host and ran
  the Dockerfile's steps at container start;
- the inputs read through a Cloud Storage FUSE mount at `/data`: our run had the checkpoint and the GSM8K files on
  the pod's local disk;
- a TPU VM without GKE: GKE set our pod's `TPU_*` variables.

AI assistance: developed with an AI coding assistant; the author has reviewed every changed line, and the tests above
were run as described (on TPU v7x where stated).
