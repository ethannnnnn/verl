# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
