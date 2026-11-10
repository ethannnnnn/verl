# Software stack for `run_qwen3_5_torchtitan.sh` on TPU v7x

[`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) is tested on TPU v7x with the software stack below. The forks of
the five repositories tag it `stack/qwen35/20261104`. They belong to the GitHub account that hosts this verl fork,
written `<OWNER>` below. Check out each fork at its tagged commit, and build the image from [`Dockerfile`](Dockerfile)
with the [Dockerfile delta](#dockerfile-delta) applied.

[Open pull requests](#open-pull-requests) lists the changes that make this recipe faster and are under review upstream:
what each one changes, how to turn it on, and which of them apply to this stack. [Measured results](#measured-results)
gives our step times and how we measured them. They come from our own experiment harness, not from this script.

## Repositories

| Repository | Tagged commit | Upstream base | Notes |
|---|---|---|---|
| `<OWNER>/verl` | `918af6c2172153a1c6bfddf8ebba326c4ac77ecf` | [verl-project/verl](https://github.com/verl-project/verl) `31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe` | This recipe |
| `<OWNER>/verl-hardware-plugin` | `1b9056f5a8d0d10fd85538823ca03fac520acc89` | [verl-project/verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) `56d2cd370a6c957aa78e7da507e86c8b354b32e9` | Shipped with the job (`py_modules`), see [README.md](README.md) |
| `<OWNER>/torchtitan` | `aaf742e4a3c990b7b72796142411e51ec44f8e90` | [google-pytorch/torchtitan](https://github.com/google-pytorch/torchtitan) `8ad0bfcdebd927c2fedef8c1e2155bd859c280f5` | `/torchtitan` in the image (`TORCHTITAN_REPO`, `TORCHTITAN_COMMIT`), see the [Dockerfile delta](#dockerfile-delta) |
| `<OWNER>/vllm-torchtpu` | `732c43464b5d4601d2529f2ee42af20c3be11c99` | [vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu), the same commit | `/vllm-torchtpu` in the image (`VLLM_TORCHTPU_COMMIT`) |
| `<OWNER>/torch_tpu` | `005903d0ef0dde919b832530451e568912c6f464` | [google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu) tag `v0.2.0`, the same commit | Installed as the `torch-tpu==0.2.0` wheel |

## Base image

`rayproject/ray:2.56.0-py312` (Python 3.12, Ray 2.56.0), image index digest
`sha256:2951c07de396a8b746f9c678b52c6e2282e614e00f80e6846a9ccd12945ae6b0`.

## Dense Qwen3.5 models

On this stack, run a dense Qwen3.5 model (e.g. `Qwen3.5-0.8B`) with `model_kernels=none` on every TorchTitan engine:
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.model_kernels=none`,
`+actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.model_kernels=none` (the reference model used by the KL loss reads its
own `engine_kwargs`, which do not inherit the actor's) and `+critic.torchtitan.engine_kwargs.tpu.model_kernels=none`. The
script passes its arguments on to `verl.trainer.main_ppo`; see [README.md](README.md) for an example.

verl-hardware-plugin `1b9056f5`, like its base `56d2cd37`, defaults to `model_kernels=auto`. That turns on the Qwen3.5
kernels `moe` and `lce` whenever `torchtitan/experiments/tpu/qwen3_5/config_registry.py` has `_use_tpu_moe_kernels`,
which torchtitan `aaf742e4` and its base `8ad0bfcd` do. The plugin does not check whether the model is a MoE. A dense
model has no `TpuGroupedExperts` and no TPU token dispatcher, so the TorchTitan engine fails at initialization with:

```
ValueError: The TPU MoE kernels need TpuGroupedExperts and a TPU token dispatcher (_TPU_MOE_CONVERTERS)
```

`model_kernels=none` turns the optional kernels off. Qwen3.5-0.8B loses nothing by it. `lce` only sets tuned fused
cross-entropy tiles (1024, 2048, 640), and those apply only when the hidden size is a multiple of 2048 and the
vocabulary a multiple of 640. Qwen3.5-0.8B has a hidden size of 1024, so it keeps the kernel's default tiles either way.

## Ray cluster layout

Training and rollout run on separate TPU slices. A TPU v7x host has 4 chips and 8 TensorCores, so each TPU host joins
the Ray cluster with `--resources='{"TPU": 8, "tpu-group-<n>": 1}'`, and `N_CHIPS_TRAINER` and `N_CHIPS_ROLLOUT` keep
their default of 8. verl-hardware-plugin gives the trainer the lowest-numbered `tpu-group-<n>` and each rollout replica
the next unclaimed one, or the last one once all are claimed. The stack is tested with these layouts:

| Chips (trainer + rollout) | Trainer | Rollout | Script settings |
|---|---|---|---|
| 8 (4 + 4) | One `2x2x1` slice: 1 host, 8 ranks, `tpu-group-0` | One `2x2x1` slice: 1 host, one TP8 engine, `tpu-group-1` | The defaults |
| 32 (16 + 16) | One `2x2x4` slice: 4 hosts, 32 ranks, `tpu-group-0` on all 4 hosts | Four `2x2x1` slices: one TP8 engine on each host, `tpu-group-1` on all 4 hosts | `NNODES_TRAINER=4 NNODES_ROLLOUT=4 ROLLOUT_TENSOR_PARALLEL_SIZE=8` |

The four rollout slices are independent, so the 16 + 16 layout needs `ROLLOUT_TENSOR_PARALLEL_SIZE=8`: by default the
script spans one engine over all 32 rollout TensorCores, which needs a single 4-host rollout slice. If the platform sets
multi-slice variables (`MEGASCALE_*`) on the TPU hosts, unset them before `ray start`, so that the trainer and rollout
slices keep separate TPU runtimes.

## Text-only checkpoint

The stack is tested with [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision
`2fc06364715b967f1860aea9cf38778875588b17`, converted to a text-only checkpoint:

- `config.json`: the source's `text_config`, with `architectures=["Qwen3_5ForCausalLM"]`, `model_type="qwen3_5_text"` and
  `tie_word_embeddings=false`.
- Weights: the `model.language_model.*` tensors, renamed to `model.*`, plus `lm_head.weight`, a copy of
  `model.embed_tokens.weight`. The vision encoder (`model.visual.*`) and the multi-token prediction weights (`mtp.*`)
  are dropped. TorchTitan's Qwen3.5 models train a separate `lm_head` rather than tying it to the embedding, so the
  checkpoint carries it explicitly. With the plugin option `tie_embeddings=True` (see
  [Open pull requests](#open-pull-requests)), the trainer instead trains one tensor for both, loaded from the two equal
  copies, and logs a warning that the checkpoint config has `tie_word_embeddings=false`.
- Files: the weights as safetensors whose file name contains a number, e.g. `model-00001-of-00001.safetensors`, and a
  matching `model.safetensors.index.json` (TorchTitan's state-dict adapter reads the shard number from the file names in
  the index), plus the source's `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `chat_template.jinja` and `LICENSE`.

## GSM8K data

[`gsm8k.py`](../../data_preprocess/gsm8k.py) loads `openai/gsm8k` (config `main`) from the Hugging Face Hub at its latest
revision. The stack is tested with revision `740312add88f781978c0658806c59bc2815b9866` (7,473 train and 1,319 test
rows). To pin it, download that revision and pass it to the script:

```bash
hf download openai/gsm8k --repo-type dataset --revision 740312add88f781978c0658806c59bc2815b9866 --local-dir /tmp/gsm8k-raw
PYTHONPATH=. python examples/data_preprocess/gsm8k.py --local_dataset_path /tmp/gsm8k-raw --local_save_dir /tmp/gsm8k
```

## Python packages

| Package | Version | Source |
|---|---|---|
| `jax`, `jaxlib` | 0.12.0.1 | PyPI, hash-pinned overlay (see below) |
| `libtpu` | 0.0.50 | [libtpu releases](https://storage.googleapis.com/jax-releases/libtpu_releases.html) |
| `tokamax` | 0.0.17 | PyPI, hash-pinned overlay (see below) |
| `torchao` | 0.17.0 | PyPI |
| `torch` | 2.13.0+cpu | [PyTorch CPU index](https://download.pytorch.org/whl/cpu) |
| `torchvision` | 0.28.0+cpu | [PyTorch CPU index](https://download.pytorch.org/whl/cpu) |
| `torch-tpu` | 0.2.0 | PyPI |
| `tpu-sync-torch` | 0.0.1.dev20261012084241 | PyPI |
| `transformers` | 5.19.0 | PyPI |
| `tensordict` | 0.15.1 | PyPI |
| `TransferQueue` | 0.1.11 | PyPI, in the [`Dockerfile`](Dockerfile) at the tag and in the hash-pinned overlay (see below) |
| `ray` | 2.56.0 | Base image |
| `vllm` | 0.32.0 | Source, tag `v0.32.0` (`5df63a7203cdb9328d824a27b682e77c7b9928be`), editable install |

The overlay, the last step of the [Dockerfile delta](#dockerfile-delta), installs these wheels (Python 3.12,
`linux/amd64`) with `pip install --no-deps --require-hashes --only-binary=:all:` after the Dockerfile's other installs:

| Wheel | sha256 |
|---|---|
| `jax-0.12.0.1-py3-none-any.whl` | `e9dfd256ba090259b10f97f0755d4fc92535b8cb38c0a40df6488a8c8a7dd8ae` |
| `jaxlib-0.12.0.1-cp312-cp312-manylinux_2_27_x86_64.whl` | `78318e9ac79c92972f81c77090f557bdb4a5a49821751c6ee3f6a9c66a948842` |
| `tokamax-0.0.17-py3-none-any.whl` | `232a1be65c52b3c436f3836f96889a521c77199df91be8d40a1af63016f0c2d6` |
| `transferqueue-0.1.11-py3-none-any.whl` | `dfc5adb71e55732bb239f964d8e7fecea6eeaa66d9bdbdcc0b12ffe9a3b0c120` |

## Dockerfile delta

Relative to [`Dockerfile`](Dockerfile) at the tagged verl commit `918af6c2`, which fetches torchtitan `89ef29b5` from
google-pytorch/torchtitan and pins jax/jaxlib 0.11.2 and tokamax 0.0.14.

- torchtitan: the stack uses `aaf742e4` from `<OWNER>/torchtitan`, which is google-pytorch/torchtitan `8ad0bfcd` plus
  one commit, `[TPU][Qwen3.5] chunked_gated_delta: take each chunk from one unbind`. google-pytorch/torchtitan does not
  have that commit, so the image fetches the fork. Replace `<OWNER>` in `TORCHTITAN_REPO`.
- Overlay: torchtitan `8ad0bfcd` pins jax/jaxlib 0.12.0.1 and tokamax 0.0.17 in
  `torchtitan/experiments/tpu/requirements.txt`, and verl's `pyproject.toml` requires `TransferQueue>=0.1.11` (V1 async
  checkpoints use its checkpoint API). The Dockerfile at the tag already installs TransferQueue 0.1.11, so the overlay
  changes only jax, jaxlib and tokamax. vllm-torchtpu `732c4346` still declares `jax==0.11.2` and `jaxlib==0.11.2`, so
  `pip check` reports those two requirements.

```diff
diff --git a/examples/tpu/grpo/Dockerfile b/examples/tpu/grpo/Dockerfile
index f8de8fb..8b6cddf 100644
--- a/examples/tpu/grpo/Dockerfile
+++ b/examples/tpu/grpo/Dockerfile
@@ -5,8 +5,10 @@ FROM rayproject/ray:2.56.0-py312

 ARG TORCH_TPU_VERSION=0.2.0
 ARG TPU_SYNC_VERSION=0.0.1.dev20261012084241
-# 10/25/2026 HEAD of torchtitan
-ARG TORCHTITAN_COMMIT=89ef29b5a0d03a6883079ab37ff1bdc498fe3485
+# Tag stack/qwen35/20261104 of the torchtitan fork: 11/03/2026 HEAD of torchtitan (8ad0bfcd) plus one commit.
+# Replace <OWNER> with the GitHub account of the forks.
+ARG TORCHTITAN_REPO=https://github.com/<OWNER>/torchtitan.git
+ARG TORCHTITAN_COMMIT=aaf742e4a3c990b7b72796142411e51ec44f8e90
 # Optimized Qwen3.5 (google-pytorch/torchtitan#80, last of the TPU Qwen3.5 perf series #73-#80):
 # ARG TORCHTITAN_COMMIT=3ebfb20abc89613b29b38634b464fbca0a24c96b

@@ -18,7 +20,7 @@ RUN apt-get update && apt-get install -y gcc-12 g++-12 build-essential cmake nin
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
     && rm -rf /var/lib/apt/lists/*

-ADD https://github.com/google-pytorch/torchtitan.git#${TORCHTITAN_COMMIT} /torchtitan
+ADD ${TORCHTITAN_REPO}#${TORCHTITAN_COMMIT} /torchtitan
 ADD https://github.com/vllm-project/vllm-torchtpu.git#${VLLM_TORCHTPU_COMMIT} /vllm-torchtpu
 ADD https://github.com/vllm-project/vllm.git#v0.32.0 /vllm
 # vllm-torchtpu is the TPU backend, so vLLM must not install tpu-inference.
@@ -66,6 +68,17 @@ pip install --no-deps "torch==${torch}+cpu" "torchvision==${torchvision}+cpu" --
 pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}"
 pip install --no-deps "jax==${jax}" "jaxlib==${jax}" "libtpu==${libtpu}" numba==0.65.0 "${transformers}" "fastapi<0.137" \
     -f "${libtpu_links}"
+
+# Overlay: jax/jaxlib and tokamax as pinned by torchtitan/experiments/tpu/requirements.txt, and the TransferQueue
+# that verl's V1 async checkpoints need. Hash-pinned PyPI wheels, installed without dependency changes.
+cat > /tmp/overlay.txt <<'REQ'
+jax==0.12.0.1 --hash=sha256:e9dfd256ba090259b10f97f0755d4fc92535b8cb38c0a40df6488a8c8a7dd8ae
+jaxlib==0.12.0.1 --hash=sha256:78318e9ac79c92972f81c77090f557bdb4a5a49821751c6ee3f6a9c66a948842
+tokamax==0.0.17 --hash=sha256:232a1be65c52b3c436f3836f96889a521c77199df91be8d40a1af63016f0c2d6
+TransferQueue==0.1.11 --hash=sha256:dfc5adb71e55732bb239f964d8e7fecea6eeaa66d9bdbdcc0b12ffe9a3b0c120
+REQ
+pip install --no-deps --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r /tmp/overlay.txt
+rm /tmp/overlay.txt
 EOF

 ENV PYTHONPATH=/torchtitan
```

## Open pull requests

These pull requests make Qwen3.5 GRPO on TPU v7x faster, or fix what the faster configuration needs. As of 2026-11-10
all of them are open and none is merged; #3070 is a draft. Each one targets the main branch of its repository, which is
newer than the tested stack above; [On the tested stack](#on-the-tested-stack) says which of them apply to the stack as
they are. The tracking issues give their status and the changes that will follow them:
[vllm-project/vllm-torchtpu#2957](https://github.com/vllm-project/vllm-torchtpu/issues/2957),
[google-pytorch/torchtitan#167](https://github.com/google-pytorch/torchtitan/issues/167),
[verl-project/verl-hardware-plugin#129](https://github.com/verl-project/verl-hardware-plugin/issues/129) and
[verl-project/verl#8595](https://github.com/verl-project/verl/issues/8595).

| Pull request | Change | Default | To turn it on | Head |
|---|---|---|---|---|
| [vllm-project/vllm-torchtpu#2951](https://github.com/vllm-project/vllm-torchtpu/pull/2951) | The sampler skips top-k / top-p when no request sets them, also without speculative decoding | Always on | - | `257d258c` |
| [vllm-project/vllm-torchtpu#2952](https://github.com/vllm-project/vllm-torchtpu/pull/2952) | Sampled-token log-probs come with the asynchronous output copy instead of a blocking read per step; stacked on #2951 | Always on | - | `c265cd8e` |
| [vllm-project/vllm-torchtpu#3070](https://github.com/vllm-project/vllm-torchtpu/pull/3070) (draft) | The decode loop reuses the device copies of unchanged sampling parameters and block tables | Always on | - | `cc908901` |
| [google-pytorch/torchtitan#168](https://github.com/google-pytorch/torchtitan/pull/168) | `TPUSelectiveAC` flags `recompute_bmm` and `save_pallas_attention` | Off | Fields of `TPUSelectiveAC.Config`, see [Notes](#notes) | `fac85b8b` |
| [google-pytorch/torchtitan#169](https://github.com/google-pytorch/torchtitan/pull/169) | The Qwen3.5 model computes its document ids once per forward | Always on | - | `e30b6ae7` |
| [google-pytorch/torchtitan#170](https://github.com/google-pytorch/torchtitan/pull/170) | `TPUSelectiveAC` flag `save_kda_residuals`, for Tokamax's KDA kernels | Off | A field of `TPUSelectiveAC.Config`, see [Notes](#notes) | `8be36ba3` |
| [verl-project/verl-hardware-plugin#130](https://github.com/verl-project/verl-hardware-plugin/pull/130) | Fixed-shape micro-batches for the TorchTitan engine | Off | `FIXED_SHAPE_ROWS` | `729a3d3f` |
| [verl-project/verl-hardware-plugin#131](https://github.com/verl-project/verl-hardware-plugin/pull/131) | One weight shared by the input embedding and the LM head; stacked on #130 | Off | `tie_embeddings` | `e845331d` |
| [verl-project/verl-hardware-plugin#132](https://github.com/verl-project/verl-hardware-plugin/pull/132) | Trainer-only libtpu flags: comma lists, per-flag override, rollout workers keep the job's flags; stacked on #130 | Off | `VERL_TPU_TRAINER_LIBTPU_INIT_ARGS` | `bd0d8fcc` |
| [verl-project/verl-hardware-plugin#135](https://github.com/verl-project/verl-hardware-plugin/pull/135) | Checks the rollout's LM head after every tpu-sync install and fails closed when it should be tied and is not | The check always runs; it raises only with `tie_word_embeddings` | tpu-sync's `tie_word_embeddings` | `bd3f6279` |
| [verl-project/verl#8596](https://github.com/verl-project/verl/pull/8596) | Configurable replay-buffer poll interval for the V1 trainer | 2.0 s, as before | `trainer.v1.sampler.poll_interval` | `36a2d85c` |
| [verl-project/verl#8627](https://github.com/verl-project/verl/pull/8627) | TransferQueue 0.1.11 in the TPU GRPO image | - | The image | `1432991a` |
| [verl-project/verl#8628](https://github.com/verl-project/verl/pull/8628) | The IS and RS sequence metrics skip rows without valid tokens | Always on (metrics only) | - | `22d5a862` |

[Turning them on](#turning-them-on) gives the exact settings for `run_qwen3_5_torchtitan.sh`.

**Accuracy note on the rollout changes.** In our 30-step runs, the configuration with all changes of the series then
available (with #2951 and #2952, before #3070) and with rollout-side settings and changes that are not proposed here
produced more repetitive, length-capped validation responses at step 30 than one without the rollout changes: 126 vs 23
of 1,319, one seed each. #2951 and #2952 are exact on CPU, and we have not attributed the difference to any one change.
#3070 reuses only byte-identical uploads; its check on TPU v7x, on a stack that also had a stage-timing change that is
not proposed yet, found identical token ids, log-prob bits and ranks over 14,336 requests.

### On the tested stack

Three of the pull requests apply to the tested stack as they are (#2951, #2952 and #8596), and the stack already has
earlier versions of four others (#130, #131, #8627 and #8628):

| Repository | Tested commit | With the open pull requests |
|---|---|---|
| verl | `918af6c2` (this branch adds documentation only) | Merge #8596, which branches from `31cfccb4`, this recipe's upstream base (commands below). #8627 and #8628 are here already, as the earlier commits `d2edc97e` and `2d081ef3`. |
| verl-hardware-plugin | `1b9056f5` | Has earlier versions of #130 and #131 (the `fixed_shape_*` and `tie_embeddings` options), and the `embed_head_accum` option, which is item 4 of the plugin's tracking issue and not a pull request yet. #132 and #135 build on plugin main and do not apply. The fork branch `feat/qwen35/trainer-libtpu-flags` (`0f781bde`, which is `1b9056f5` plus one commit) has the version of #132's change that our runs used. |
| TorchTitan | `aaf742e4` | None needed: the tested plugin does not set the flags of #168, #169 brought no measurable speed change, and #170 needs the Tokamax GatedDeltaNet kernels of TorchTitan main. All three heads contain TorchTitan main, which moved `activation_checkpoint.py` to `components/`. Their change commits for #168 (`d47bb488`) and #169 (`3f26cfd0`, `7944a7d3`) do apply to `aaf742e4`; the image then needs a TorchTitan commit in a repository you control. |
| vllm-torchtpu | `732c4346` | #2951 and #2952 branch from it, and the head of #2952, `c265cd8e`, contains both: merging the two heads gives the same tree. #3070 builds on vllm-torchtpu main; its source changes apply on top of `c265cd8e`, but its new tests in `tests/layers/core/test_attention_metadata.py` conflict with that file's older version, and the image then needs a vllm-torchtpu commit in a repository you control. |
| torch_tpu | `005903d0` | None in this list. |

```bash
# verl (this checkout): merge #8596. -X theirs settles the one conflict, the "Last updated" line of
# docs/advance/v1_async_trainer.md, with the date from #8596.
git fetch https://github.com/verl-project/verl.git pull/8596/head
git merge --no-edit -X theirs 36a2d85c2eeeea2eb9361326517fc62174333cf9

# verl-hardware-plugin: the tagged commit plus trainer-only libtpu flags
git -C ../verl-hardware-plugin fetch https://github.com/<OWNER>/verl-hardware-plugin.git feat/qwen35/trainer-libtpu-flags
git -C ../verl-hardware-plugin checkout 0f781bde69f43d0b878e29a676ea32a69a2e67dd
```

For the image, build with `--build-arg VLLM_TORCHTPU_COMMIT=c265cd8ea652e87a81e74724fcbacbf72c834948`, and point the
vllm-torchtpu `ADD` line of [`Dockerfile`](Dockerfile) at `https://github.com/<OWNER>/vllm-torchtpu.git`, where that
commit is the head of branch `feat/qwen35/deferred-logprobs`: Docker's `ADD` of a Git URL may not find a commit that
only a pull request reaches.

We checked with `git merge-tree` on 2026-11-10 that these merges apply. We have not built this image or run this
combination; our runs used these commits with earlier versions or equivalents of the changes (see
[Measured results](#measured-results)).

### On current mains

To try the pull requests as they are proposed, merge their heads onto the main of each repository, in the order below.
On 2026-11-10 we checked with `git merge-tree` that they merge onto these main commits as described. We have not run
the merged trees, and current mains need newer pins than the image above: vllm-torchtpu main, for example, builds
against vLLM 0.33.0.

| Repository | Main | Merge in this order | Conflicts |
|---|---|---|---|
| verl-project/verl-hardware-plugin | `511dce30` | #130, #131, #132, #135 | None |
| google-pytorch/torchtitan | `036ef100` | #168, #169, #170 | #170 and #168 change the same lines of `TPUSelectiveAC` in `torchtitan/experiments/tpu/components/activation_checkpoint.py`: keep all three flags, as the Dependencies section of #170 describes, and add `save_kda_residuals` to `_FLAGS` in the test that #168 adds |
| vllm-project/vllm-torchtpu | `229f2e7d` | #2951, #2952, #3070 | None; main's `tests/runner/test_min_tokens.py` needs the test-only change that #2951 and #2952 describe |
| verl-project/verl | `1ec15fed` | #8596, #8628, #8627 | #8627 and #8596 change the same "Last updated" line of `docs/advance/v1_async_trainer.md`; keep either date |

```bash
# verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git && cd verl-hardware-plugin
git checkout -b open-prs 511dce30298620cc5e6ae8f5d7a471f2547f905a
git fetch origin pull/130/head pull/131/head pull/132/head pull/135/head
for head in 729a3d3f407548be5053dd11d0125356e5cdad21 e845331d23a0fad4570878c6bd8ce7c220bfd28f \
    bd0d8fcc84ca68f231972b3f670abafb4a22af72 bd3f62798cc4d9838f17d3396f36e89e438ec162; do
  git merge --no-edit "${head}" || break
done
cd ..

# torchtitan: the merge of #170 stops at the conflict described above
git clone https://github.com/google-pytorch/torchtitan.git && cd torchtitan
git checkout -b open-prs 036ef100d7af56d1f1d3a8805914a2ffbc3e7e15
git fetch origin pull/168/head pull/169/head pull/170/head
for head in fac85b8b976bf7248edc3c33392de49ab2b64959 e30b6ae75cb335aa39725fcca04288f7011ab965 \
    8be36ba397282a9c8c0f4e9e5b21e877fa0126ca; do
  git merge --no-edit "${head}" || break
done
cd ..

# vllm-torchtpu
git clone https://github.com/vllm-project/vllm-torchtpu.git && cd vllm-torchtpu
git checkout -b open-prs 229f2e7d19ff1491ae98fbe41c6a049683178d6a
git fetch origin pull/2951/head pull/2952/head pull/3070/head
for head in 257d258c61431cefe16df008c0d58e71d7081946 c265cd8ea652e87a81e74724fcbacbf72c834948 \
    cc908901aef67b90cb55db8212aa02c81ec4eaac; do
  git merge --no-edit "${head}" || break
done
cd ..

# verl: the merge of #8627 stops at the "Last updated" line
git clone https://github.com/verl-project/verl.git verl-main && cd verl-main
git checkout -b open-prs 1ec15fed25f3fbdb79b15b8e4213292d9038e2d3
git fetch origin pull/8596/head pull/8627/head pull/8628/head
for head in 36a2d85c2eeeea2eb9361326517fc62174333cf9 22d5a8627c9bcc85321da7cf6b3fd9a262987c4a \
    1432991a98d447d66c5a25b800e02e4d2353099c; do
  git merge --no-edit "${head}" || break
done
```

### Turning them on

| Change | Setting for `run_qwen3_5_torchtitan.sh` |
|---|---|
| Fixed-shape micro-batches (#130; also in the tested plugin) | `FIXED_SHAPE_ROWS=8192:14336:16384` in the script's environment, see [README.md](README.md). The script then sets the plugin's `fixed_shape_*` options and the actor's `use_dynamic_bsz=True`. The example in #130 also sets `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.seq_bucket_size=2048`. |
| Tied input embedding and LM head (#131; also in the tested plugin) | `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tie_embeddings=True` |
| Per-step embedding and LM head gradient reduction (in the tested plugin; not a pull request yet) | `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.embed_head_accum=True` |
| No LM head in the weight sync (a tpu-sync option; #135 checks it) | `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.tie_word_embeddings=True`, only together with `tie_embeddings=True`, see [Notes](#notes) |
| Trainer-only libtpu flags (#132, or the fork branch above) | `VERL_TPU_TRAINER_LIBTPU_INIT_ARGS` in the script's environment, e.g. the flags in the example below |
| Replay-buffer poll interval (#8596) | `trainer.v1.sampler.poll_interval=0.1` |
| No top-k log-probs in vLLM (an option of this script) | `ROLLOUT_MAX_LOGPROBS=0` in the script's environment, see [README.md](README.md) |

For example, Qwen3.5-0.8B on the tested stack with the plugin's fork branch above and #8596 merged, submitted as in
[README.md](README.md). We have not run it exactly as written.

```bash
# The trainer libtpu flags that we use with libtpu 0.0.50, as listed in #132
flags=(
  --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0
  --xla_enable_async_all_gather=true
  --xla_tpu_prefer_async_allgather_to_allreduce=true
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_aggregator=true
  --xla_tpu_enable_latency_hiding_layer_scheduler=true
  --xla_tpu_dvfs_p_state=7
)
TRAINER_LIBTPU_FLAGS=$(IFS=,; echo "${flags[*]}")

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-0.8B ROLLOUT_MAX_LOGPROBS=0 FIXED_SHAPE_ROWS=8192:14336:16384 \
  VERL_TPU_TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_FLAGS}" \
  bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh \
  +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.model_kernels=none \
  +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.model_kernels=none \
  +critic.torchtitan.engine_kwargs.tpu.model_kernels=none \
  +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.seq_bucket_size=2048 \
  +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tie_embeddings=True \
  +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.embed_head_accum=True \
  +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.tie_word_embeddings=True \
  trainer.v1.sampler.poll_interval=0.1
```

### Notes

- #168 and #170 change `TPUSelectiveAC`, the policy that
  `actor_rollout_ref.actor.torchtitan.activation_checkpoint=selective` selects; this script uses `full`.
  verl-hardware-plugin does not set their flags yet: that is item 5 of its tracking issue (fork branch
  `feat/qwen35/tpu-sac-flags`). Against TorchTitan main, the plugin's import of `TPUSelectiveAC` also falls back to core
  `SelectiveAC` without notice, because `activation_checkpoint.py` moved to `components/` (see
  verl-project/verl-hardware-plugin#129).
- #170 matters only with Tokamax's KDA kernels for the GatedDeltaNet (`model_kernels=kda`), which come with
  google-pytorch/torchtitan#74 and its follow-ups on TorchTitan main; the tested TorchTitan does not have them. Our
  later runs used them with Tokamax 0.0.18.dev20261106 (a pre-release on PyPI), its scalar gate (item 8 of
  google-pytorch/torchtitan#167, which waits for a Tokamax release) and the backward precision change that the same
  issue describes (not in a Tokamax release; we plan to report it to openxla/tokamax).
- Set tpu-sync's `tie_word_embeddings` only when the trainer trains one tied tensor (`tie_embeddings=True`). With a
  separately trained LM head, the rollout would sample with the trained embedding in its place, and the check of #135
  cannot see that case.
- #132 changes how plugin main applies `VERL_TPU_TRAINER_LIBTPU_INIT_ARGS` (verl-project/verl-hardware-plugin#115).
  With plugin main alone, a comma-separated value stays one token of `LIBTPU_INIT_ARGS`, and the rollout workers get
  the trainer flags too. The tested plugin `1b9056f5` ignores the variable; the fork branch above reads it.
- `trainer.v1.sampler.poll_interval` exists only with #8596; without it, Hydra rejects the key.
- #3070 stays a draft until its same-host serving numbers on TPU are attached.

## Measured results

The step times, MFU values and GSM8K accuracies in the table below and in the paragraph after it come from our own
experiment harness, not from `run_qwen3_5_torchtitan.sh`, and we have not run this script with the open pull requests
above. The harness runs the same model (the [text-only checkpoint](#text-only-checkpoint) above) on the same GSM8K
revision, on 8 TPU v7x chips (one `2x2x1` trainer slice with 8 ranks, one `2x2x1` rollout slice), with the tested
stack's commits plus, depending on the row, earlier versions or equivalents of the changes above and the later items of
the tracking issues that so far exist only as fork branches. These fork-branch items account for most of the trainer
MFU gain from 5.97% to 10.95% below (the tracking issues give the per-change numbers): the TorchTitan GatedDeltaNet
changes (items 2, 5 and 6 of google-pytorch/torchtitan#167; branch `feat/qwen35/gdn-chunk-recurrence-vjp` of
`<OWNER>/torchtitan`), the plugin's selective-AC flags and deferred host syncs (items 5 and 6 of
verl-project/verl-hardware-plugin#129; branch `feat/qwen35/deferred-host-syncs` of `<OWNER>/verl-hardware-plugin`) and
verl's deferred host scalars (item 3 of verl-project/verl#8595; branch `feat/qwen35/deferred-host-scalars` of
`<OWNER>/verl`). In the last row of the table, the trainer also ran Tokamax's KDA kernels. The harness differs from this
script in the settings and the code that [Settings of the measured runs](#settings-of-the-measured-runs) lists, and,
where a row of the table says so, in further changes that are not proposed here. So:

- the GSM8K accuracies are scored with the harness's own prompt and reward function, and are not comparable with the
  accuracy that this script logs;
- each configuration ran with one seed, once or, where noted, twice;
- our environment was built separately with the package versions above, not from the image above; in the KDA runs,
  the trainer had Tokamax 0.0.18.dev20261106 instead of 0.0.17 (see [Notes](#notes)).

All MFU values in this section come from our own FLOP count, with compile time excluded. verl's `perf/mfu/actor` is 0 on
TPU at this verl commit, because `verl/utils/flops_counter.py` has no TPU entry yet (verl main has one since
verl-project/verl#8469).

| Configuration | GRPO step | Also measured |
|---|---|---|
| Starting point of the series | 68.7 s | |
| Earlier versions or equivalents of the open pull requests except #3070, #135 and #170, the fork-branch items above, and rollout-side settings and changes that are not proposed here (see [Settings of the measured runs](#settings-of-the-measured-runs)) | 23.2 s | Trainer MFU 10.95%; GSM8K validation (1,319 problems, greedy) 54.5% / 59.1% / 70.6% / 74.6% at steps 0 / 10 / 20 / 30 |
| A later configuration: Tokamax KDA kernels in the trainer with the flag of #170, and further settings that are not proposed here | 21.7 s (two runs) | Generation 18.2 s per step |

Generation dominates the step in these runs. Trainer MFU was 5.97% on the tested stack itself, against the 10.95% of
the second row; moving the trainer to Tokamax's KDA kernels later raised it from 10.8% to 16.1% (two runs each) without
a measurable change of the step time. With one TP8 rollout engine, #2951 shortened the decode step at 128 concurrent
sequences from 12.7 ms to 9.2 ms and #2952 then to 5.8 ms; with #2952 the GRPO step went from 50.4 s to 37.2 s. In a
separate single-change comparison, tpu-sync's `tie_word_embeddings`, with the trainer's embedding and LM head tied, cut
the weight-sync transfer from 0.70 s to 0.53 s per sync. The pull requests give the per-change measurements and their
setups. #2951 and #2952 also carry a reviewer's independent measurements (#2951's decode numbers on one host),
including a GRPO ablation on 16 + 16 chips. For the rollout changes, see also the [accuracy note](#open-pull-requests)
above.

### Settings of the measured runs

The harness runs above, in the table and in the paragraph after it, all used the training settings below. They are
standard verl, vLLM and TorchTitan options and public environment switches; where they differ between runs, the row
says so. The runs with one TP8 rollout engine, the script's layout, are the starting point of the series, the 5.97%
trainer MFU on the tested stack and the decode-step measurements. Next to each value is the default of
`run_qwen3_5_torchtitan.sh` at the tagged commit, and how to set our value with the script. We have not run the script
with these settings.

| Setting | Our measured runs | Script default | With the script |
|---|---|---|---|
| Prompts per step x samples per prompt | 32 x 16 (512 responses); PPO mini-batch 32, so one optimizer step per step | 16 x 4; mini-batch 16 | `TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=16` |
| Prompt / response length | 256 / 4,096 tokens | 512 / 2,048 | `MAX_PROMPT_LEN=256 MAX_RESPONSE_LEN=4096`. With the Qwen3.5 chat template, the longest prompt that [`gsm8k.py`](../../data_preprocess/gsm8k.py) writes has 239 tokens, so no problem is filtered out |
| Thinking (`enable_thinking` of the chat template) | Off | On | `++data.apply_chat_template_kwargs.enable_thinking=False` (`++`, because the script sets the key with `+`) |
| Learning rate | 2.5e-6, 10 warmup steps | 1e-6, 5 warmup steps | `LEARNING_RATE=2.5e-6 LR_WARMUP_STEPS=10` |
| LR schedule after warmup | Cosine, `min_lr_factor=0.84` | Cosine, `min_lr_factor=0.0` (verl's default) | `actor_rollout_ref.actor.optim.min_lr_factor=0.84`. The floor makes no difference here: the tested plugin passes no step count to TorchTitan's learning-rate scheduler, so the cosine spans TorchTitan's default of 10,000 steps, and the learning rate stays within 0.1% of its peak after warmup, in our 30-step runs and in the script's 100 steps |
| Weight decay | 0.0 | 0.01 (verl's default) | `actor_rollout_ref.actor.optim.weight_decay=0.0` |
| KL loss | Off | On: coefficient 0.001, `low_var_kl`, against a reference model | `actor_rollout_ref.actor.use_kl_loss=False` |
| Clipping | PPO clip 0.2 (low and high); no dual-clip (`clip_ratio_c=1.0e9`) | PPO clip 0.2; `clip_ratio_c=3.0` | `actor_rollout_ref.actor.clip_ratio_c=1.0e9` |
| Activation checkpointing | Selective (`TPUSelectiveAC`) | Full | `actor_rollout_ref.actor.torchtitan.activation_checkpoint=selective`; the flags of #168 and #170 need a plugin that sets them, see [Notes](#notes) |
| Async warmup batches (`trainer.v1.separate_async.num_warmup_batches`) | 0 | 2 | `NUM_WARMUP_BATCHES=0` |
| Dynamic batch size | On, with fixed-shape micro-batches: rows of 8,192, 14,336 and 16,384 tokens, at most 64 sequences per row, `seq_bucket_size=2048` | Off | `FIXED_SHAPE_ROWS=8192:14336:16384`, which turns on `use_dynamic_bsz`, and `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.seq_bucket_size=2048` |
| vLLM `max_num_seqs`, `gpu_memory_utilization` | 256 and 0.8 in the runs with one TP8 engine; 128 and 0.6 in the other runs | 256, 0.6 | `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION` |
| vLLM `max_logprobs`; engine seeds | 0; one seed per replica | vLLM's default (20); `SEED` for every replica | `ROLLOUT_MAX_LOGPROBS=0 ROLLOUT_ENGINE_SEED=null` |
| Rollout environment | `VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1` (a vllm-torchtpu switch, off by default), except in the runs with one TP8 engine; `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1` | `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1` | Set `VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1` in the environment of the TPU hosts before `ray start` |
| Rollout dumps (`trainer.rollout_data_dir`) | On in the 23.2 s run, off in the other runs | On, every step | `trainer.rollout_data_dir=null` turns them off |
| Steps and validation | 30 steps, validation on all 1,319 test problems at step 0 and every 10 steps; the starting point of the series, the 5.97% trainer MFU run on the tested stack and some single-change measurements ran 8 steps, with validation at step 8 | 100 steps; 128 test problems at step 0 and every 20 steps | `TOTAL_TRAINING_STEPS=30 TEST_FREQ=10 VAL_MAX_SAMPLES=-1 VAL_BATCH_SIZE=1319` |

These settings are the script's too: AdamW with betas (0.9, 0.999), eps 1e-8 and gradient clipping 1.0; rollout
temperature 1.0 and top-p 1.0; greedy validation; bypass-mode rollout correction with token-level importance sampling
(threshold 3.0); token-mean loss aggregation; no entropy bonus. All runs also set the tested plugin's
`tie_embeddings=True` and `embed_head_accum=True` (see [Turning them on](#turning-them-on)).

Besides these settings, our runs used our own prompt, reward and advantage code, and a rollout layout that is not
public yet.

### Measurements of this script

The only end-to-end measurement of this script itself that we have published, with its own prompt and verl's GSM8K
reward, is the check we ran when we tagged the stack: on the tested stack, with 32 chips (the 16 + 16 layout), settings
other than the script's defaults and none of the open pull requests beyond the earlier versions that the tested stack
contains, 30 steps reached 71.34% GSM8K accuracy on the full test set (71.72% for the upstream base commits), at a
trainer MFU of 5.58% (1.54%).

Not measured: this script with any of the open pull requests or with the settings of our measured runs; the merged
trees on current mains; an image built from [`Dockerfile`](Dockerfile) with the [Dockerfile delta](#dockerfile-delta);
the accuracy effect of the rollout changes over more than one seed.
