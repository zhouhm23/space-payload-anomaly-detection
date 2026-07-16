"""Django views for PHM API endpoints (23 total, paths identical to FastAPI).

All /api/* views are CSRF-exempt (called by frontend JS via fetch).
Services are accessed through services_bridge.get_container().
"""

from __future__ import annotations

import json

from django.http import HttpRequest, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import VerdictForm, AlertVerdictForm, AlertStatusForm, DiagnosisForm
from .services_bridge import get_container


# ── poll / forecast / config / reset / health / sensors ─────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def poll_view(request: HttpRequest):
    body = json.loads(request.body or "{}")
    source_id = body.get("source_id", "file:NASA-MSL/C-1")
    sample_rate = float(body.get("sample_rate", 50.0))
    block_size = int(body.get("block_size", 512))
    c = get_container()
    result = c.telemetry.poll(source_id, sample_rate, block_size)
    ingested = result.pop("_ingested", {})
    for ch in ingested:
        try:
            c.warning_service.evaluate_channel(ch, block_size)
        except Exception:
            pass
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def forecast_view(request: HttpRequest):
    body = json.loads(request.body or "{}")
    values = body.get("values", [])
    c = get_container()
    result = c.forecast.forecast(values)
    status = 400 if "error" in result else 200
    return JsonResponse(result, status=status)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def config_view(request: HttpRequest):
    """GET /api/config — load device tree; POST /api/config — save device tree."""
    c = get_container()
    if request.method == "GET":
        return JsonResponse(c.config.load())
    # POST
    body = json.loads(request.body or "{}")
    return JsonResponse(c.config.save(body))


@csrf_exempt
@require_http_methods(["POST"])
def reset_view(request: HttpRequest):
    c = get_container()
    c.ring.clear()
    c.alerts.clear()
    c.warnings.clear()
    return JsonResponse({"status": "ok"})


@csrf_exempt
@require_http_methods(["GET"])
def health_view(request: HttpRequest):
    block_size = min(max(int(request.GET.get('block_size', 20000)), 1), 20000)
    c = get_container()
    return JsonResponse(c.health.system_health(block_size))


@csrf_exempt
@require_http_methods(["GET"])
def sensors_view(request: HttpRequest):
    c = get_container()
    latest = c.ring.latest_metrics()
    health = c.health.system_health()
    sensors = []
    for ch, m in latest.items():
        sensors.append({
            "channel": ch,
            "latest_raw": m["raw"],
            "latest_score": m["score"],
            "points": m["points"],
            "received_at": m["received_at"],
            "health": health["channels"].get(ch, 100.0),
        })
    return JsonResponse({"sensors": sensors, "system_health": health["system"]})


# ── alerts (4 endpoints) ────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["GET"])
def alerts_view(request: HttpRequest):
    limit = min(max(int(request.GET.get('limit', 50)), 1), 500)
    c = get_container()
    alerts = c.alert_service.list(limit)
    # The in-memory deque has no human_verdict/llm_verdict (those live in the
    # DB). Merge them from alert_records so the frontend verdict buttons reflect
    # the persisted state after a page refresh or re-fetch.  Keyed by
    # (channel, created_at≈time) — the same key used by update_alert_verdict.
    try:
        db_rows = c.sqlite.query_alerts(limit=limit * 2)
        verdict_map = {}
        for r in db_rows:
            key = (r.get("channel"), round(float(r.get("created_at", 0)), 3))
            verdict_map[key] = {
                "human_verdict": r.get("human_verdict"),
                "llm_verdict": r.get("llm_verdict"),
                "final_status": r.get("final_status"),
            }
        for a in alerts:
            key = (a.get("channel"), round(float(a.get("time", 0)), 3))
            v = verdict_map.get(key)
            if v:
                a["human_verdict"] = v["human_verdict"]
                a["llm_verdict"] = v["llm_verdict"]
                a["final_status"] = v["final_status"]
    except Exception:
        pass
    return JsonResponse({
        "alerts": alerts,
        "threshold": c.alert_service.threshold,
    })


@csrf_exempt
@require_http_methods(["GET"])
def alerts_history_view(request: HttpRequest):
    limit = min(max(int(request.GET.get('limit', 50)), 1), 500)
    from .models import AlertRecord
    qs = AlertRecord.objects.filter(is_deleted=0).order_by('-created_at')[:limit]
    alerts = []
    for a in qs:
        alerts.append({
            "id": a.id, "channel": a.channel, "alert_type": a.alert_type,
            "score": a.score, "message": a.message, "created_at": a.created_at,
            "status": a.status, "verified_at": a.verified_at,
            "llm_verdict": a.llm_verdict, "human_verdict": a.human_verdict,
            "final_status": a.final_status,
        })
    c = get_container()
    return JsonResponse({"alerts": alerts, "threshold": c.alert_service.threshold})


@csrf_exempt
@require_http_methods(["PATCH"])
def patch_alert_view(request: HttpRequest, alert_id: int):
    body = json.loads(request.body or "{}")
    form = AlertStatusForm(body)
    if not form.is_valid():
        return JsonResponse(form.errors, status=422)
    c = get_container()
    ok = c.sqlite.update_alert_status(alert_id, form.cleaned_data['status'])
    if not ok:
        return JsonResponse({"ok": False, "error": "not_found_or_invalid_status"}, status=404)
    return JsonResponse({"ok": True, "id": alert_id, "status": form.cleaned_data['status']})


@csrf_exempt
@require_http_methods(["POST"])
def alert_verdict_view(request: HttpRequest):
    body = json.loads(request.body or "{}")
    form = AlertVerdictForm(body)
    if not form.is_valid():
        return JsonResponse(form.errors, status=422)
    d = form.cleaned_data
    c = get_container()
    ok = c.sqlite.update_alert_verdict(d['channel'], d['alert_ts'], d['human_verdict'])
    if not ok:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    return JsonResponse({"ok": True, "human_verdict": d['human_verdict']})


# ── warnings (3 endpoints) ──────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["GET"])
def warnings_view(request: HttpRequest):
    limit = min(max(int(request.GET.get('limit', 50)), 1), 500)
    c = get_container()
    return JsonResponse({"warnings": c.warning_service.list(limit)})


@csrf_exempt
@require_http_methods(["GET"])
def predict_scores_view(request: HttpRequest):
    channel = request.GET.get('channel', '')
    c = get_container()
    data = c.warning_service.get_latest_predict_scores(channel)
    if data is None:
        return JsonResponse({"timestamps": [], "scores": [], "predict_start": 0, "predict_end": 0})
    return JsonResponse(data)


@csrf_exempt
@require_http_methods(["POST"])
def warning_verdict_view(request: HttpRequest, warning_id: int):
    body = json.loads(request.body or "{}")
    form = VerdictForm(body)
    if not form.is_valid():
        return JsonResponse(form.errors, status=422)
    c = get_container()
    ok = c.warning_service.warnings.set_verdict(warning_id, "human", form.cleaned_data['human_verdict'])
    if not ok:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    return JsonResponse({"ok": True, "id": warning_id, "human_verdict": form.cleaned_data['human_verdict']})


# ── history / detection / db-stats / window / export ────────────────────────

@csrf_exempt
@require_http_methods(["GET", "DELETE"])
def history_view(request: HttpRequest):
    """GET /api/history — query; DELETE /api/history — delete with confirm guard."""
    channel = request.GET.get('channel') or None
    start = float(request.GET['start']) if 'start' in request.GET else None
    end = float(request.GET['end']) if 'end' in request.GET else None
    if request.method == "GET":
        limit = min(max(int(request.GET.get('limit', 1000)), 1), 10000)
        c = get_container()
        rows = c.sqlite.query_history(channel=channel, start_time=start, end_time=end, limit=limit)
        return JsonResponse({"count": len(rows), "data": rows})
    # DELETE
    confirm = request.GET.get('confirm', 'false').lower() == 'true'
    if not channel and start is None and end is None and not confirm:
        return JsonResponse({"deleted": 0, "error": "confirm_required"}, status=400)
    c = get_container()
    deleted = c.sqlite.delete_history(channel=channel, start_time=start, end_time=end)
    return JsonResponse({"deleted": deleted})


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
def detection_view(request: HttpRequest):
    """GET /api/detection — query; DELETE /api/detection — delete with confirm guard."""
    channel = request.GET.get('channel') or None
    start = float(request.GET['start']) if 'start' in request.GET else None
    end = float(request.GET['end']) if 'end' in request.GET else None
    if request.method == "GET":
        limit = min(max(int(request.GET.get('limit', 50)), 1), 500)
        from .models import DetectionResult
        qs = DetectionResult.objects.filter(is_deleted=0).order_by('-timestamp')[:limit] if channel is None \
            else DetectionResult.objects.filter(channel=channel, is_deleted=0).order_by('-timestamp')[:limit]
        rows = []
        for d in qs:
            # l1_detail / l3_rules are stored as JSON TEXT in SQLiteStore;
            # parse to objects so the frontend can use them directly (matches
            # SQLiteStore.query_detection behaviour).
            try:
                l1_detail = json.loads(d.l1_detail) if d.l1_detail else {}
            except (json.JSONDecodeError, TypeError):
                l1_detail = {}
            try:
                l3_rules = json.loads(d.l3_rules) if d.l3_rules else []
            except (json.JSONDecodeError, TypeError):
                l3_rules = []
            rows.append({
                "id": d.id, "channel": d.channel, "timestamp": d.timestamp,
                "l1_decision": d.l1_decision, "l1_score": d.l1_score,
                "l1_detail": l1_detail, "l2_score": d.l2_score,
                "l3_score": d.l3_score, "l3_rules": l3_rules,
                "final_score": d.final_score,
            })
        latest = None
        if channel is not None:
            c = get_container()
            cascade = c.warning_service.get_latest_cascade(channel)
            if cascade is not None:
                latest = cascade.to_dict(max_detail=True)
        return JsonResponse({"count": len(rows), "data": rows, "latest": latest})
    # DELETE
    confirm = request.GET.get('confirm', 'false').lower() == 'true'
    if not channel and start is None and end is None and not confirm:
        return JsonResponse({"deleted": 0, "error": "confirm_required"}, status=400)
    c = get_container()
    deleted = c.sqlite.delete_detection(channel=channel, start_time=start, end_time=end)
    return JsonResponse({"deleted": deleted})


@csrf_exempt
@require_http_methods(["GET"])
def db_stats_view(request: HttpRequest):
    c = get_container()
    return JsonResponse(c.sqlite.stats())


@csrf_exempt
@require_http_methods(["GET"])
def window_view(request: HttpRequest):
    channel = request.GET.get('channel', '')
    count = min(max(int(request.GET.get('count', 512)), 1), 10000)
    end_ts = float(request.GET['end_ts']) if 'end_ts' in request.GET else None
    c = get_container()
    result = c.sqlite.query_window(channel=channel, count=count, end_ts=end_ts)
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["GET"])
def export_view(request: HttpRequest):
    import csv
    import io
    from datetime import datetime, timezone
    from django.http import StreamingHttpResponse

    channels = request.GET.get('channels', '')
    start = float(request.GET['start'])
    end = float(request.GET['end'])
    fmt = request.GET.get('fmt', 'csv')
    c = get_container()
    ch_list = [ch.strip() for ch in channels.split(",") if ch.strip()]
    all_rows = []
    for ch in ch_list:
        rows = c.sqlite.query_history(channel=ch, start_time=start, end_time=end, limit=100000)
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: r.get("received_at", 0))

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_channels = "_".join(ch_list[:3])
    if len(ch_list) > 3:
        safe_channels += f"_plus{len(ch_list) - 3}"
    filename = f"telemetry_{safe_channels}_{ts_str}.csv"

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["channel", "timestamp", "raw_value", "anomaly_score", "received_at_iso"])
    for r in all_rows:
        ts = r.get("received_at")
        iso_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts is not None else ""
        writer.writerow([r.get("channel", ""), ts if ts is not None else "",
                         r.get("raw", ""), r.get("score", ""), iso_str])
    content = output.getvalue().encode("utf-8-sig")

    resp = StreamingHttpResponse(io.BytesIO(content), content_type="text/csv; charset=utf-8")
    resp['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp


# ── diagnosis (4 endpoints) ─────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def diagnosis_view(request: HttpRequest):
    body = json.loads(request.body or "{}")
    form = DiagnosisForm(body)
    if not form.is_valid():
        return JsonResponse(form.errors, status=422)
    d = form.cleaned_data
    c = get_container()
    if not c.diagnosis.enabled:
        return JsonResponse({
            "error": "LLM diagnosis not configured",
            "detail": "Set OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL environment variables.",
        }, status=503)
    result = c.diagnosis.diagnose(
        d['channel'], alert_type=d['alert_type'], alert_ts=d['alert_ts'],
        force_refresh=bool(body.get('force_refresh', False)),
    )
    if result.get("error"):
        status = 502 if "LLM" in result["error"] else 404
        return JsonResponse(result, status=status)
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["GET"])
def diagnosis_done_view(request: HttpRequest):
    limit = min(max(int(request.GET.get('limit', 200)), 1), 1000)
    c = get_container()
    if c.sqlite is None:
        return JsonResponse({"done": []})
    items = c.sqlite.list_diagnosis_keys(limit=limit)
    return JsonResponse({"done": items})


@csrf_exempt
@require_http_methods(["POST"])
def diagnosis_auto_view(request: HttpRequest):
    c = get_container()
    if not c.diagnosis:
        return JsonResponse({"error": "diagnosis service not available"}, status=503)
    result = c.diagnosis.auto_diagnose_all()
    if result.get("error"):
        return JsonResponse(result, status=409)
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["GET"])
def diagnosis_auto_status_view(request: HttpRequest):
    c = get_container()
    if not c.diagnosis:
        return JsonResponse({"running": False, "done": 0, "total": 0, "errors": 0})
    return JsonResponse(c.diagnosis.auto_status)


# ── RUL degradation prediction ──────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["GET"])
def rul_view(request: HttpRequest):
    """Return RUL predictions for channels tagged ``@rul:fd00X``.

    With no ``channel`` query param, advances the C-MAPSS playback by one
    cycle and predicts for every tagged channel (the front-end polls this).
    With ``?channel=xxx`` returns a single channel without advancing.
    """
    c = get_container()
    if c.rul is None:
        return JsonResponse({
            "status": "disabled",
            "message": "RUL 服务未启用（C-MAPSS 数据 / FD001 权重 / scaler 缺失）",
        }, status=503)
    channel = request.GET.get('channel')
    if channel:
        result = c.rul.predict(channel)
        if result is None:
            return JsonResponse({
                "status": "ok",
                "data": None,
                "message": "通道未启用 RUL 或无足够数据",
            })
        return JsonResponse({"status": "ok", "data": result})
    data = c.rul.predict_all()
    return JsonResponse({"status": "ok", "data": data})


# ── monitor page ────────────────────────────────────────────────────────────

def monitor_view(request: HttpRequest):
    """Render the real-time monitoring page (migrated from dashboard.html)."""
    from django.shortcuts import render
    return render(request, 'phm_site/monitor.html')
