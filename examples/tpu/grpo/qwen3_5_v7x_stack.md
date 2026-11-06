# Software stack for `run_qwen3_5_torchtitan.sh` on TPU v7x

[`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) is tested on TPU v7x with the software stack below. The forks of
the five repositories tag it `stack/qwen35/20261104`. They belong to the GitHub account that hosts this verl fork,
written `<OWNER>` below. Check out each fork at its tagged commit, and build the image from [`Dockerfile`](Dockerfile)
with the [Dockerfile delta](#dockerfile-delta) applied.

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
