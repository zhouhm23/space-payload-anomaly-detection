"""Score-direction calibration for reconstruction-based detectors.

TSPulse reconstructs the input and flags large reconstruction error as
anomalous.  On channels with a high anomaly ratio in the scored window this
inverts: the model has learned to reconstruct the (dominant) anomalous
pattern well, so **normal** points get the higher error.  The resulting AUC
drops below 0.5 — the score is systematically backwards.

The fix is a per-channel direction flip (``score → 1 - score``).  Whether to
flip is decided **offline** with a few-shot label probe
(:meth:`DirectionCalibrator.fit`); at runtime the cascade only needs the
boolean verdict, applied via :meth:`DirectionCalibrator.flip`.

Validation (NASA-MSL, ``experiments/tspulse_eval/run_with_direction_calibration.py:132-197``):
the N=3 few-shot judge matches the oracle (full-label AUC<0.5) on 96.3% of
channels.  Constants and logic below are ported verbatim from that script.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DirectionCalibrator"]


# Few-shot direction-judge defaults — validated in
# experiments/tspulse_eval/run_with_direction_calibration.py:60-66.
N_SAMPLE = 3
N_TRIALS = 20
ANOM_THRESHOLD = 0.15  # window anomaly-rate above which a window is "anomalous"
WINDOW_SIZE = 512      # window size (matches TSPulse context length)


class DirectionCalibrator:
    """Stateless score-direction calibrator.

    The class has no instance state — it groups two related operations:

    * :meth:`fit` (offline, needs labels) — decide whether a channel's
      score direction is inverted.  Returns a bool to store in
      ``channel_calibration.json``.
    * :meth:`flip` (online, no labels) — apply the stored decision to a
      score array.

    The input to :meth:`flip` **must already be normalised to ``[0,1]``**
    (TSPulse clip-normalises; the frequency score is mapped via its training-
    segment z-score reference); flipping a raw-score array whose range is
    unknown produces a meaningless result.  :class:`AnomalyDetector` clips
    before returning, and :class:`CascadeDetector` applies the flip right
    after, so this contract holds in the default pipeline.
    """

    # ------------------------------------------------------------------
    # Online transform
    # ------------------------------------------------------------------

    @staticmethod
    def flip(score: np.ndarray, do_flip: bool) -> np.ndarray:
        """Apply a stored flip decision to a normalised ``[0,1]`` score array.

        Args:
            score: 1-D float array in ``[0, 1]``.
            do_flip: the boolean verdict from :meth:`fit` (read from the
                channel calibration config at runtime).

        Returns:
            1-D float32 array of the same length.  If ``do_flip`` is False
            the array is returned unchanged (as float32); otherwise each
            element becomes ``1 - x``.
        """
        score = np.asarray(score, dtype=np.float32).ravel()
        if not do_flip:
            return score
        return (1.0 - score).astype(np.float32)

    # ------------------------------------------------------------------
    # Offline fit (needs labels)
    # ------------------------------------------------------------------

    @staticmethod
    def fit(
        score: np.ndarray,
        labels: np.ndarray,
        n_sample: int = N_SAMPLE,
        n_trials: int = N_TRIALS,
        anom_threshold: float = ANOM_THRESHOLD,
        window_size: int = WINDOW_SIZE,
        seed: int = 42,
    ) -> tuple[bool, dict]:
        """Few-shot N=3 direction judge — decide if a channel needs flipping.

        Slices the sequence into windows, samples ``n_sample`` labelled
        windows per trial (forcing at least one normal + one anomalous),
        and votes on whether anomalous windows have a *lower* mean score
        than normal windows (the signature of an inverted direction).

        Args:
            score: 1-D anomaly score array (MinMax-normalised recommended
                but not required — only the ordering matters here).
            labels: 1-D 0/1 ground-truth array, same length as ``score``.
            n_sample: windows sampled per trial (default 3).
            n_trials: number of sampling trials (default 20).
            anom_threshold: a window counts as anomalous if its label
                mean ≥ this value (default 0.15).
            window_size: window length in samples (default 512).
            seed: RNG seed for reproducibility.

        Returns:
            ``(flip, detail)`` where ``flip`` is True if the score direction
            appears inverted, and ``detail`` carries vote tallies for
            diagnostics.
        """
        rng = np.random.RandomState(seed)
        score = np.asarray(score, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        n = len(score)
        n_windows = max(1, n // window_size)

        # Slice into windows; compute each window's mean score + anomaly rate.
        window_scores: list[float] = []
        window_anom_rates: list[float] = []
        for i in range(n_windows):
            s = i * window_size
            e = min(s + window_size, n)
            if e - s < window_size // 2:
                break
            window_scores.append(float(np.mean(score[s:e])))
            window_anom_rates.append(float(np.mean(labels[s:e])))

        window_labels = [r >= anom_threshold for r in window_anom_rates]
        n_anom_win = sum(window_labels)
        n_norm_win = len(window_labels) - n_anom_win

        if n_anom_win == 0 or n_norm_win == 0:
            return False, {
                "reason": "no_both_classes",
                "n_anom_win": n_anom_win,
                "n_norm_win": n_norm_win,
            }

        anom_idx = [i for i, a in enumerate(window_labels) if a]
        norm_idx = [i for i, a in enumerate(window_labels) if not a]

        votes_flip = votes_noop = valid_trials = 0
        for _ in range(n_trials):
            picked: list[int] = []
            if n_sample >= 2 and anom_idx and norm_idx:
                picked.append(int(rng.choice(anom_idx)))
                picked.append(int(rng.choice(norm_idx)))
                remaining = [i for i in range(len(window_labels)) if i not in picked]
                rng.shuffle(remaining)
                picked.extend(remaining[: max(0, n_sample - 2)])
            else:
                pool = list(range(len(window_labels)))
                rng.shuffle(pool)
                picked = pool[:n_sample]

            ns = [window_scores[i] for i in picked if not window_labels[i]]
            as_ = [window_scores[i] for i in picked if window_labels[i]]
            if not ns or not as_:
                continue
            if float(np.mean(as_)) < float(np.mean(ns)):
                votes_flip += 1
            else:
                votes_noop += 1
            valid_trials += 1

        if valid_trials == 0:
            return False, {"reason": "no_valid_trials"}

        flip = votes_flip > votes_noop
        return flip, {
            "votes_flip": votes_flip,
            "votes_noop": votes_noop,
            "valid_trials": valid_trials,
            "n_anom_win": n_anom_win,
            "n_norm_win": n_norm_win,
        }
