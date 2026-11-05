# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the optional settings of the Qwen3.5 TPU GRPO example against verl's real Hydra config.

The example (``examples/tpu/grpo/run_qwen3_5_torchtitan.sh``) runs with a ``python3`` on ``PATH`` that records its
arguments instead of starting ``verl.trainer.main_ppo``; the recorded overrides are then composed with verl's trainer
config. No TPU, Ray or verl-hardware-plugin is needed.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "tpu" / "grpo" / "run_qwen3_5_torchtitan.sh"
CONFIG_DIR = REPO_ROOT / "verl" / "trainer" / "config"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the example is a bash script")


def run_example(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    """Runs the example with the given environment; its ``python3`` arguments go to ``tmp_path / "args.json"``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    recorder = bin_dir / "python3"
    record = 'import json, os, sys; json.dump(sys.argv[1:], open(os.environ["ARGS_FILE"], "w"))'
    recorder.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(record)} "$@"\n')
    recorder.chmod(0o755)
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "ARGS_FILE": str(tmp_path / "args.json"),
        **env,
    }
    return subprocess.run(["bash", str(EXAMPLE)], env=full_env, cwd=tmp_path, capture_output=True, text=True)


def compose_example(tmp_path: Path, **env: str) -> DictConfig:
    """Composes the config that the example passes to ``verl.trainer.main_ppo``."""
    result = run_example(tmp_path, **env)
    assert result.returncode == 0, result.stderr
    argv = json.loads((tmp_path / "args.json").read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    # hydra.* overrides configure Hydra's run directory, not the job config.
    overrides = [arg for arg in argv[2:] if not arg.startswith("hydra.")]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_defaults_are_unchanged(tmp_path):
    config = compose_example(tmp_path)
    rollout = config.actor_rollout_ref.rollout
    assert rollout.seed == 1
    assert OmegaConf.to_container(rollout.engine_kwargs.vllm) == {"seed": 1, "no_enable_prefix_caching": True}
    actor = config.actor_rollout_ref.actor
    assert actor.use_dynamic_bsz is False
    assert actor.ppo_max_token_len_per_gpu == 8192
    assert not any(key.startswith("fixed_shape_") for key in actor.torchtitan.engine_kwargs.tpu)


def test_null_engine_seed_keeps_per_replica_seeds(tmp_path):
    config = compose_example(tmp_path, SEED="7", ROLLOUT_ENGINE_SEED="null")
    rollout = config.actor_rollout_ref.rollout
    # The vLLM server drops engine kwargs set to None, so replica r keeps its engine seed rollout.seed + r.
    assert rollout.engine_kwargs.vllm.seed is None
    assert rollout.seed == 7
    assert config.data.seed == 7


def test_max_logprobs(tmp_path):
    config = compose_example(tmp_path, ROLLOUT_MAX_LOGPROBS="0")
    rollout = config.actor_rollout_ref.rollout
    assert rollout.engine_kwargs.vllm.max_logprobs == 0
    assert rollout.engine_kwargs.vllm.seed == 1
    # The rollout worker config passes it to vLLM unchanged (it is raised only for rollout.topk_log_probs).
    OmegaConf.resolve(config)  # as verl.trainer.main_ppo does before building the workers
    rollout_config = omega_conf_to_dataclass(config.actor_rollout_ref.rollout, RolloutConfig)
    assert rollout_config.engine_kwargs["vllm"]["max_logprobs"] == 0


def test_fixed_shape_rows(tmp_path):
    config = compose_example(tmp_path, FIXED_SHAPE_ROWS="8192:16384:14336", MAX_RESPONSE_LEN="4096")
    actor = config.actor_rollout_ref.actor
    tpu = actor.torchtitan.engine_kwargs.tpu
    assert tpu.fixed_shape_rows == "8192:16384:14336"
    assert tpu.fixed_shape_max_seqs == 64
    assert tpu.fixed_shape_response_len == 4096
    assert tpu.fixed_shape_prewarm is True
    # The fixed layout replaces the dynamic-batch partition, with the largest row as the token budget.
    assert actor.use_dynamic_bsz is True
    assert actor.ppo_max_token_len_per_gpu == 16384
    # The reference model keeps its static micro-batches. The rollout log-prob setting follows the actor, as the
    # engine workers require.
    ref = config.actor_rollout_ref.ref
    assert ref.log_prob_use_dynamic_bsz is False
    assert not any(key.startswith("fixed_shape_") for key in ref.torchtitan.engine_kwargs.tpu)
    assert config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz is True


def test_fixed_shape_max_seqs(tmp_path):
    config = compose_example(tmp_path, FIXED_SHAPE_ROWS="4096:8192", FIXED_SHAPE_MAX_SEQS="32")
    assert config.actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.fixed_shape_max_seqs == 32
    assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 8192


@pytest.mark.parametrize("rows", ["2048", "2048:2048", "8192:x", "8192::16384", "-8192", "08192"])
def test_invalid_fixed_shape_rows_fail_before_launch(tmp_path, rows):
    # The default MAX_PROMPT_LEN + MAX_RESPONSE_LEN is 2560 tokens, so a largest row of 2048 cannot hold a sequence.
    result = run_example(tmp_path, FIXED_SHAPE_ROWS=rows)
    assert result.returncode != 0
    assert "FIXED_SHAPE_ROWS" in result.stderr
    assert not (tmp_path / "args.json").exists()
