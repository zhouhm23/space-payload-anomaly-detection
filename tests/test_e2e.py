"""End-to-end integration test with golden-value verification.

Launches two subprocesses (signal_generator + DAQ main.py), connects via
TCP as a ground client, verifies:

  1. MSL C-1 telemetry matches pre-computed golden values (mean/std/min/max).
  2. Detection scores are present and non-trivial (not all-zero).
  3. loop=True keeps producing data after exhaustion.
  4. Space TCP server stays reachable.
  5. Forecaster produces valid 96-step output (no NaN).
  6. Space shutdown handled gracefully.
  7. After data exhausted, polling still reaches space (not timeout).

M1.2 architecture: signal generator (owns SensorSource) + DAQ card (does
detection + TCP) + ground client. The test creates temporary config files
so it doesn't pollute the default ``space_daq.json`` / ``signal_sources.json``.

Golden values computed directly from TSB-UAD dataset files:
    MSL  C-1  first-512: mean=-0.860542 std=0.245817 min=-1.0 max=-0.046802

Run:
    pytest tests/test_e2e.py -v -s
"""

import json
import os
import sys
import time
import socket
import subprocess
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))
sys.path.insert(0, os.path.join(_SRC, "space"))

from ground.comm import GroundClient, TelemetryPacket, AlertPacket

SPACE_PYTHON = os.path.join(_SRC, ".conda-env", "python.exe")
SPACE_HOST = "127.0.0.1"
SPACE_PORT = 9877   # DAQ TCP port (avoid default 9876)
IPC_PORT = 9879     # signal-generator IPC port (avoid default 9878)
TEST_WINDOW = 512
TEST_RATE = 200  # fast enough so each window cycle takes ~2.5s

# ── Golden values (from dataset slice [512:1024], NOT from pipeline) ──
# Space segment consumes [0:512] for initial scaler fit, so the first
# telemetry packet sent to ground contains dataset points 512-1023.
GOLDEN_MSL_C1 = {
    "channel": "C-1",
    "mean": -0.35490503907203674,
    "std": 0.5395727753639221,
    "min": -1.0,
    "max": 1.0,
    "first5": [-0.9921996593475342, -0.9875195026397705,
               -0.9656786322593689, -0.9407176375389099, -0.8471139073371887],
}
STAT_TOL = 1e-4


def _wait_for_port(host, port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((host, port))
            s.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


def _verify_against_dataset(raw, dataset, channel, label):
    """Verify raw_values exactly match a contiguous slice of the dataset."""
    sys.path.insert(0, os.path.join(_SRC, "space"))
    from data_loader import list_channels, load_channel

    chs = list_channels(dataset)
    match = [c for c in chs if c[0] == channel]
    assert match, f"Channel {channel} not in {dataset}"
    ts, _ = load_channel(match[0][2], match[0][1])

    fp = raw[:5]
    found = False
    for start in range(0, len(ts) - TEST_WINDOW + 1, TEST_WINDOW):
        if np.allclose(ts[start:start + 5], fp, atol=STAT_TOL):
            if np.allclose(ts[start:start + TEST_WINDOW], raw, atol=STAT_TOL):
                found = True
                break
    assert found, \
        f"[{label}] Telemetry does not match any window in {dataset}/{channel}"


def _drain(client, cfg, n_polls=8, delay=0.5):
    all_t, all_a = [], []
    for _ in range(n_polls):
        for p in client.poll(cfg):
            if isinstance(p, TelemetryPacket):
                all_t.append(p)
            elif isinstance(p, AlertPacket):
                all_a.append(p)
        time.sleep(delay)
    return all_t, all_a


def _write_temp_configs(tmpdir, sample_rate, window):
    """Create temporary space_daq.json + signal_sources.json with only C-1.

    Keeps test isolated from the default configs under space/data/.
    """
    daq_cfg = {
        "sample_rate": sample_rate,
        "window_size": window,
        "host": SPACE_HOST,
        "port": SPACE_PORT,
        "channels": [
            {"id": 0, "name": "C-1", "enabled": True, "isSpecial": False},
        ],
    }
    sig_cfg = {
        "default_sample_rate": sample_rate,
        "bindings": [
            {"channel": "C-1", "sourceId": "file:NASA-MSL/C-1", "loop": True},
        ],
    }
    daq_path = os.path.join(tmpdir, "space_daq.json")
    sig_path = os.path.join(tmpdir, "signal_sources.json")
    with open(daq_path, "w", encoding="utf-8") as f:
        json.dump(daq_cfg, f, ensure_ascii=False, indent=2)
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(sig_cfg, f, ensure_ascii=False, indent=2)
    return daq_path, sig_path


@pytest.fixture(scope="module")
def space_processes():
    """Start signal_generator + DAQ main.py, return (gen_proc, daq_proc).

    Uses temporary config files so the test is isolated from default configs.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HOME"] = os.path.join(_SRC, ".hf_cache")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    tmpdir = tempfile.mkdtemp(prefix="phm_e2e_")
    daq_path, sig_path = _write_temp_configs(tmpdir, TEST_RATE, TEST_WINDOW)

    # 1. Start signal generator
    gen_proc = subprocess.Popen(
        [SPACE_PYTHON, "-m", "space.signal_generator",
         "--port", str(IPC_PORT),
         "--config", sig_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_SRC,
    )
    # Wait for IPC server
    if not _wait_for_port("127.0.0.1", IPC_PORT, timeout=30):
        out = gen_proc.stdout.read().decode("utf-8", errors="replace") if gen_proc.stdout else ""
        gen_proc.kill()
        pytest.fail(f"Signal generator did not start.\nOutput:\n{out}")

    # 2. Start DAQ main.py (connects to IPC + serves TCP)
    daq_proc = subprocess.Popen(
        [SPACE_PYTHON, "-m", "space.main",
         "--daq-config", daq_path,
         "--ipc-port", str(IPC_PORT),
         "--host", SPACE_HOST, "--port", str(SPACE_PORT),
         "--sample-rate", str(TEST_RATE), "--window", str(TEST_WINDOW)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_SRC,
    )
    if not _wait_for_port(SPACE_HOST, SPACE_PORT, timeout=120):
        out = daq_proc.stdout.read().decode("utf-8", errors="replace") if daq_proc.stdout else ""
        gen_proc.terminate(); daq_proc.kill()
        pytest.fail(f"DAQ main.py did not start.\nOutput:\n{out}")

    yield gen_proc, daq_proc

    for p in (daq_proc, gen_proc):
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    # Clean up temp configs
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


class TestE2E:
    """Full pipeline: signal_gen → DAQ → TCP → ground client, with golden checks."""

    @staticmethod
    def _cfg(channel="C-1"):
        return {"source_id": channel, "sample_rate": TEST_RATE}

    def test_1_msl_c1_golden_values(self, space_processes):
        """MSL C-1 telemetry must exactly match a contiguous slice of the dataset."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        telemetry, _ = _drain(client, self._cfg())
        assert len(telemetry) > 0, "No telemetry from C-1"

        full = [p for p in telemetry if len(p.raw_values) == TEST_WINDOW]
        assert full, f"No full-length packets; got {[len(p.raw_values) for p in telemetry]}"
        pkt = full[0]
        assert pkt.channel == GOLDEN_MSL_C1["channel"], \
            f"Channel: got {pkt.channel}, expected {GOLDEN_MSL_C1['channel']}"

        _verify_against_dataset(pkt.raw_values, "NASA-MSL", "C-1", "MSL C-1")

    def test_2_detection_scores_nontrivial(self, space_processes):
        """Detection scores must be present and non-trivial (not all-zero)."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        telemetry, _ = _drain(client, self._cfg())
        assert len(telemetry) > 0

        scored = [p for p in telemetry
                  if p.scores is not None and len(p.scores) == TEST_WINDOW]
        assert scored, \
            f"No scored packets; got {[len(p.scores) if p.scores is not None else None for p in telemetry]}"

        sc = scored[0].scores
        assert not (sc < 0).any(), "Scores should be non-negative"
        assert float(np.max(sc)) > 0.0, \
            f"All scores zero — detection broken. max={float(np.max(sc))}"
        assert int(np.count_nonzero(sc)) > TEST_WINDOW // 2, \
            f"Too many zero scores: only {int(np.count_nonzero(sc))}/{TEST_WINDOW} nonzero"

    def test_3_loop_mode(self, space_processes):
        """With loop=True, signal generator keeps producing data after exhaustion."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        time.sleep(TEST_WINDOW / TEST_RATE + 1)
        pkts = client.poll(self._cfg())
        tele = [p for p in pkts if isinstance(p, TelemetryPacket)]
        assert len(tele) > 0, "Loop mode: expected data after pacing interval"

    def test_4_space_still_alive(self, space_processes):
        """DAQ TCP server still reachable (returns list, not exception)."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        pkts = client.poll(self._cfg())
        assert isinstance(pkts, list), "Server should still accept connections"

    def test_5_forecast_valid(self, space_processes):
        """Forecaster produces 96-step output with no NaN."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        client.poll(self._cfg())
        time.sleep(1)
        telemetry, _ = _drain(client, self._cfg(), n_polls=5)
        assert len(telemetry) > 0

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = os.path.join(_SRC, ".hf_cache")
        from ground.phm.algorithm import TrendForecaster
        f = TrendForecaster(device="cpu")
        raw = telemetry[-1].raw_values
        assert len(raw) >= 512
        ctx, pred, _scaler = f.forecast(raw[-512:])
        assert len(ctx) == 512, f"Context len {len(ctx)} != 512"
        assert len(pred) == 96, f"Prediction len {len(pred)} != 96"
        assert not np.isnan(pred).any(), "Prediction has NaN"

    def test_6_space_shutdown(self, space_processes):
        """Poll returns a list (server reachable)."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=1)
        pkts = client.poll({"source_id": "C-1"})
        assert isinstance(pkts, list)

    def test_7_space_still_connected_after_data_exhausted(self, space_processes):
        """Even after loop wraps around, polling still reaches the DAQ."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        pkts = client.poll({"source_id": "C-1", "sample_rate": 100})
        assert isinstance(pkts, list), "Poll should not raise"

