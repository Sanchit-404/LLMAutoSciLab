"""
JSONL worker for ChemGymRL experiments.

Protocol:
  request: {"cmd":"run","action":[...],"seed":123}
  request: {"cmd":"batch","actions":[[...], [...]],"seed":123}
  response: {"ok":true,"measurement":float,"reward":float}
"""
from __future__ import annotations

import json
import sys

import gymnasium as gym
import chemistrylab  # noqa: F401  # Needed to register environments


def _run_once(env_id: str, action: list[float], seed: int = 123) -> tuple[float, float]:
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
        _, reward, _, _, _ = env.step(action)
        # Keep measurement positive for log-space models used in AutoSciLab.
        measurement = float(reward + 2.0)
        if measurement <= 1e-12:
            measurement = 1e-12
        return measurement, float(reward)
    finally:
        env.close()


def main() -> None:
    env_id = sys.argv[1] if len(sys.argv) > 1 else "FictReactBandit-v0"

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            seed = int(req.get("seed", 123))

            if cmd == "run":
                measurement, reward = _run_once(env_id, req["action"], seed=seed)
                out = {"ok": True, "measurement": measurement, "reward": reward}
            elif cmd == "batch":
                ys = []
                rewards = []
                for action in req["actions"]:
                    m, r = _run_once(env_id, action, seed=seed)
                    ys.append(m)
                    rewards.append(r)
                out = {"ok": True, "measurements": ys, "rewards": rewards}
            else:
                out = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:
            out = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
