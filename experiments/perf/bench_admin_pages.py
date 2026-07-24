"""Admin-page response-time benchmark (issue #6 — slow pages diagnosis).

Logs into the Django test client as a staff user, then GETs every admin
page that the user reported as "slow", printing min / mean / max wall time
across N runs per page.  No browser needed — this measures server-side
render cost only (template + DB + Python), which is the dominant factor
the user is seeing.

Run::

    ./.conda-env/python.exe experiments/perf/bench_admin_pages.py

Reads the live phm.db / channel_calibration.json / device_config.json
(via the real Container), so numbers reflect the actual 2.6 GB DB the
user is hitting.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# Bootstrap Django (mirror experiments/metrics/run_leakfree_metrics.py layout).
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent  # .../src
sys.path.insert(0, str(_SRC / "ground"))
sys.path.insert(0, str(_SRC / "ground" / "django_phm"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_phm.settings")

import django  # noqa: E402

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import Client  # noqa: E402
from django.urls import reverse  # noqa: E402


# (label, reverse name, query string) — query strings mirror typical usage.
PAGES: list[tuple[str, str, dict]] = [
    ("仪表盘 (dashboard)",          "phm_admin_dashboard",  {}),
    ("仪表盘 ?auto=1",              "phm_admin_dashboard",  {"auto": "1"}),
    ("告警管理 (alert)",            "phm_admin_alert",      {}),
    ("遥测数据 (telemetry)",        "phm_admin_telemetry",  {"channel": "D-14"}),
    ("遥测数据 (无 channel)",       "phm_admin_telemetry",  {}),
    ("算法库 L1 (library)",         "phm_admin_library",    {"cat": "l1"}),
    ("算法库 L2",                   "phm_admin_library",    {"cat": "l2"}),
    ("算法库 预测",                 "phm_admin_library",    {"cat": "forecast"}),
    ("回收站 告警",                 "phm_admin_recycle",    {}),
    ("回收站 遥测",                 "phm_admin_recycle",    {"type": "telemetry", "channel": "D-14"}),
    ("设备树",                      "phm_admin_device_tree", {}),
    ("权限管理",                    "phm_admin_permissions", {}),
    ("系统设置",                    "phm_admin_settings",    {}),
]

RUNS = 5


def main() -> int:
    User = get_user_model()
    # Reuse the staff user if one exists, else create one (test-only creds).
    staff = User.objects.filter(is_staff=True).first()
    if staff is None:
        staff = User.objects.create_user(
            username="perf_bench", password="perf_bench_pw", is_staff=True
        )
        print(f"[setup] created staff user {staff.username}")
    else:
        print(f"[setup] reusing staff user {staff.username}")

    client = Client()
    client.force_login(staff)

    # Warm-up: first request per page primes caches (config load, registry
    # import, sqlite_master scan).  We discard its timing.
    print(f"\n{'page':<32} {'min':>8} {'mean':>8} {'max':>8} {'p50':>8}   status")
    print("-" * 82)
    slow: list[tuple[str, float]] = []
    for label, url_name, qs in PAGES:
        url = reverse(url_name)
        try:
            # Warm-up
            client.get(url, qs)
        except Exception as e:
            print(f"{label:<32}   WARMUP ERROR: {e!r}")
            continue

        timings: list[float] = []
        last_status = 0
        for _ in range(RUNS):
            t0 = time.perf_counter()
            resp = client.get(url, qs)
            t1 = time.perf_counter()
            last_status = resp.status_code
            timings.append(t1 - t0)
        mean_ms = statistics.mean(timings) * 1000
        row = (
            f"{label:<32} "
            f"{min(timings) * 1000:>8.1f} "
            f"{mean_ms:>8.1f} "
            f"{max(timings) * 1000:>8.1f} "
            f"{statistics.median(timings) * 1000:>8.1f}   {last_status}"
        )
        print(row)
        if mean_ms > 1500:
            slow.append((label, mean_ms))

    print("-" * 82)
    if slow:
        print(f"\nSLOW PAGES (>1.5s mean):")
        for label, ms in sorted(slow, key=lambda x: -x[1]):
            print(f"  {label:<32} {ms:>8.1f} ms")
    else:
        print("\nNo page exceeded the 1.5s mean threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
