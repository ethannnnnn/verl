# Software stack for `run_qwen3_5_torchtitan.sh` on TPU v7x

[`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) is tested on TPU v7x with the software stack below. Build the
image from [`Dockerfile`](Dockerfile) with the [Dockerfile delta](#dockerfile-delta) applied, and check out the
repositories at the listed commits.

## Repositories

| Repository | Commit | Notes |
|---|---|---|
| [verl-project/verl](https://github.com/verl-project/verl) | `31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe` | Base of this recipe |
| [verl-project/verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) | `56d2cd370a6c957aa78e7da507e86c8b354b32e9` | Shipped with the job (`py_modules`), see [README.md](README.md) |
| [google-pytorch/torchtitan](https://github.com/google-pytorch/torchtitan) | `8ad0bfcdebd927c2fedef8c1e2155bd859c280f5` | `/torchtitan` in the image (`TORCHTITAN_COMMIT`) |
| [vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) | `732c43464b5d4601d2529f2ee42af20c3be11c99` | `/vllm-torchtpu` in the image (`VLLM_TORCHTPU_COMMIT`) |
| [google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu) | `005903d0ef0dde919b832530451e568912c6f464` | Tag `v0.2.0`; installed as the `torch-tpu==0.2.0` wheel |

## Base image

`rayproject/ray:2.56.0-py312` (Python 3.12, Ray 2.56.0), image index digest
`sha256:2951c07de396a8b746f9c678b52c6e2282e614e00f80e6846a9ccd12945ae6b0`.

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
| `TransferQueue` | 0.1.11 | PyPI, hash-pinned overlay (see below) |
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

Relative to [`Dockerfile`](Dockerfile) at verl `31cfccb4`, which pins jax/jaxlib 0.11.2, tokamax 0.0.14,
TransferQueue 0.1.8 and torchtitan `89ef29b5`. torchtitan `8ad0bfcd` pins jax/jaxlib 0.12.0.1 and tokamax 0.0.17 in
`torchtitan/experiments/tpu/requirements.txt`, and verl's `pyproject.toml` requires `TransferQueue>=0.1.11` (V1 async
checkpoints use its checkpoint API). The overlay changes only these four packages. vllm-torchtpu `732c4346` still
declares `jax==0.11.2` and `jaxlib==0.11.2`, so `pip check` reports those two requirements.

```diff
diff --git a/examples/tpu/grpo/Dockerfile b/examples/tpu/grpo/Dockerfile
index 71c1e24..07f2366 100644
--- a/examples/tpu/grpo/Dockerfile
+++ b/examples/tpu/grpo/Dockerfile
@@ -5,8 +5,8 @@ FROM rayproject/ray:2.56.0-py312

 ARG TORCH_TPU_VERSION=0.2.0
 ARG TPU_SYNC_VERSION=0.0.1.dev20261012084241
-# 10/25/2026 HEAD of torchtitan
-ARG TORCHTITAN_COMMIT=89ef29b5a0d03a6883079ab37ff1bdc498fe3485
+# 11/03/2026 HEAD of torchtitan
+ARG TORCHTITAN_COMMIT=8ad0bfcdebd927c2fedef8c1e2155bd859c280f5
 # Optimized Qwen3.5 (google-pytorch/torchtitan#80, last of the TPU Qwen3.5 perf series #73-#80):
 # ARG TORCHTITAN_COMMIT=3ebfb20abc89613b29b38634b464fbca0a24c96b

@@ -65,6 +65,17 @@ pip install --no-deps "torch==${torch}+cpu" "torchvision==${torchvision}+cpu" --
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
