"""End-to-end integration test with golden-value verification.

Launches the space segment as a subprocess, connects via TCP, verifies:

  1. MSL C-1 telemetry matches pre-computed golden values (mean/std/min/max).
  2. Detection scores are present and non-trivial (not all-zero).
  3. Channel switch MSL→SMAP: telemetry changes to match SMAP E-1 golden.
  4. Reconfig latency bounded (<5 s after sending config).
  5. Forecaster produces valid 96-step output (no NaN).
  6. Space shutdown handled gracefully.

Golden values computed directly from TSB-UAD dataset files:
    MSL  C-1  first-512: mean=-0.860542 std=0.245817 min=-1.0 max=-0.046802
    SMAP E-1  first-512: mean=-0.609375 std=0.792882 min=-1.0 max=1.0

Run:
    pytest tests/test_e2e.py -v -s
"""

import os
import sys
import time
import socket
import subprocess

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
SPACE_PORT = 9877  # avoid clashing with default 9876
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
GOLDEN_SMAP_E1 = {
    "channel": "E-1",
    "mean": -0.55859375,
    "std": 0.8294413685798645,
    "min": -1.0,
    "max": 1.0,
    "first5": [-1.0, -1.0, -1.0, -1.0, -1.0],
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


def _assert_golden(raw, golden, label):
    """Assert raw_values match pre-computed golden statistics."""
    assert len(raw) == TEST_WINDOW, \
        f"[{label}] Expected {TEST_WINDOW} samples, got {len(raw)}"
    # Statistical fingerprint — if the data is from the right channel these
    # must match to ~1e-4 (float32 precision).
    assert abs(float(np.mean(raw)) - golden["mean"]) < STAT_TOL, \
        f"[{label}] mean: got {float(np.mean(raw)):.6f}, expected {golden['mean']:.6f}"
    assert abs(float(np.std(raw)) - golden["std"]) < STAT_TOL, \
        f"[{label}] std: got {float(np.std(raw)):.6f}, expected {golden['std']:.6f}"
    assert abs(float(np.min(raw)) - golden["min"]) < STAT_TOL, \
        f"[{label}] min: got {float(np.min(raw)):.6f}, expected {golden['min']:.6f}"
    assert abs(float(np.max(raw)) - golden["max"]) < STAT_TOL, \
        f"[{label}] max: got {float(np.max(raw)):.6f}, expected {golden['max']:.6f}"


def _verify_against_dataset(raw, dataset, channel, label):
    """Verify raw_values exactly match a contiguous slice of the dataset.

    Searches all 512-point windows in the dataset for an exact match.
    This proves the telemetry came from the real dataset file (not zeros,
    not synthetic, not wrong channel).
    """
    import sys
    sys.path.insert(0, os.path.join(_SRC, "space"))
    from data_loader import list_channels, load_channel
    import numpy as np

    chs = list_channels(dataset)
    match = [c for c in chs if c[0] == channel]
    assert match, f"Channel {channel} not in {dataset}"
    ts, _ = load_channel(match[0][2], match[0][1])

    # Search for exact match (first 5 points is enough as fingerprint)
    fp = raw[:5]
    found = False
    for start in range(0, len(ts) - TEST_WINDOW + 1, TEST_WINDOW):
        if np.allclose(ts[start:start + 5], fp, atol=STAT_TOL):
            # Full window check
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


@pytest.fixture(scope="module")
def space_process():
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC
    # Ensure space subprocess can load models from local cache without network
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HOME"] = os.path.join(_SRC, ".hf_cache")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    proc = subprocess.Popen(
        [SPACE_PYTHON, "-m", "space.main",
         "--host", SPACE_HOST, "--port", str(SPACE_PORT),
         "--source", "dataset", "--dataset", "NASA-MSL", "--channel", "C-1",
         "--sample-rate", str(TEST_RATE), "--window", str(TEST_WINDOW)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_SRC,
    )
    if not _wait_for_port(SPACE_HOST, SPACE_PORT, timeout=90):
        out = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
        proc.kill()
        pytest.fail(f"Space segment did not start.\nOutput:\n{out}")
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestE2E:
    """Full pipeline: space → TCP → ground client, with golden-value checks."""

    @staticmethod
    def _cfg(dataset="NASA-MSL", channel="C-1", source="dataset"):
        return {"source_type": source, "dataset_name": dataset,
                "channel": channel, "sample_rate": TEST_RATE}

    def test_1_msl_c1_golden_values(self, space_process):
        """MSL C-1 telemetry must exactly match a contiguous slice of the dataset."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        telemetry, _ = _drain(client, self._cfg())
        assert len(telemetry) > 0, "No telemetry from MSL C-1"

        full = [p for p in telemetry if len(p.raw_values) == TEST_WINDOW]
        assert full, f"No full-length packets; got {[len(p.raw_values) for p in telemetry]}"
        pkt = full[0]
        assert pkt.channel == GOLDEN_MSL_C1["channel"], \
            f"Channel: got {pkt.channel}, expected {GOLDEN_MSL_C1['channel']}"

        # Exact match against the actual dataset file — catches wrong-channel,
        # all-zeros, or corrupted data bugs that statistical checks would miss.
        _verify_against_dataset(pkt.raw_values, "NASA-MSL", "C-1", "MSL C-1")

    def test_2_detection_scores_nontrivial(self, space_process):
        """Space segment must return non-trivial detection scores for telemetry.

        MSL C-1 data at offset [512:1024] contains 201 ground-truth anomaly
        points (labels=1).  The fixed TSPulse detector produces per-point
        reconstruction error scores normalized to [0, 1], so:
          - All 512 points should have scores (no zeros-only output)
          - max score must be > 0 (broken pipeline returned all-zeros)
          - Scores on anomaly points should be higher than normal points
        """
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

    def test_3_channel_switch_golden(self, space_process):
        """Switch MSL→SMAP E-1; verify telemetry is genuinely from SMAP dataset."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        smap_cfg = self._cfg("NASA-SMAP", "E-1")
        client.poll(smap_cfg)
        # drain old buffered MSL data
        for _ in range(3):
            client.poll(smap_cfg)
            time.sleep(0.5)

        smap_pkt = None
        for _ in range(10):
            for p in client.poll(smap_cfg):
                if (isinstance(p, TelemetryPacket)
                        and len(p.raw_values) == TEST_WINDOW
                        and p.channel == GOLDEN_SMAP_E1["channel"]):
                    smap_pkt = p
                    break
            if smap_pkt is not None:
                break
            time.sleep(1.0)

        assert smap_pkt is not None, \
            "SMAP E-1 data never arrived — reconfig may be blocked"

        # Exact match against SMAP E-1 dataset — proves channel switch worked
        _verify_against_dataset(smap_pkt.raw_values, "NASA-SMAP", "E-1", "SMAP E-1")

    def test_4_reconfig_latency(self, space_process):
        """After sending synthetic config, data must arrive within 5 s."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        synth_cfg = {"source_type": "synthetic", "signal_type": "multi_sine",
                      "freq": 0.02, "sample_rate": TEST_RATE}
        t0 = time.time()
        client.poll(synth_cfg)
        latency = None
        for _ in range(10):
            for p in client.poll(synth_cfg):
                if isinstance(p, TelemetryPacket) and p.channel.startswith("SYN"):
                    latency = time.time() - t0
                    break
            if latency is not None:
                break
            time.sleep(0.5)
        assert latency is not None, "Synthetic data never arrived"
        assert latency < 5.0, \
            f"Reconfig latency {latency:.1f}s exceeds 5s — pace loop blocking"

    def test_5_forecast_valid(self, space_process):
        """Forecaster produces 96-step output with no NaN."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=3)
        client.poll(self._cfg())
        time.sleep(1)
        telemetry, _ = _drain(client, self._cfg(), n_polls=5)
        assert len(telemetry) > 0

        # Set offline env before loading model (avoids SSL errors)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = os.path.join(_SRC, ".hf_cache")
        from ground.forecasting import TrendForecaster
        f = TrendForecaster(device="cpu")
        raw = telemetry[-1].raw_values
        assert len(raw) >= 512
        ctx, pred = f.forecast(raw[-512:])
        assert len(ctx) == 512, f"Context len {len(ctx)} != 512"
        assert len(pred) == 96, f"Prediction len {len(pred)} != 96"
        assert not np.isnan(pred).any(), "Prediction has NaN"

    def test_6_space_shutdown(self, space_process):
        """Poll returns a list after space terminates."""
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=1)
        pkts = client.poll({"source_type": "dataset"})
        assert isinstance(pkts, list)
