"""Tests for Django API views (22 endpoints, URL paths identical to FastAPI)."""
import json
from unittest.mock import patch

import pytest
from django.test import Client, TestCase

from phm_site import services_bridge


@pytest.mark.django_db
class TestPollForecastConfigResetHealthSensors(TestCase):
    """First batch: poll, forecast, config, reset, health, sensors."""

    def setUp(self):
        # apps.py ready() only fires under runserver (RUN_MAIN=true),
        # so under pytest the container must be started explicitly.
        services_bridge.start()
        self.client = Client()

    def test_get_config(self):
        resp = self.client.get('/api/config')
        assert resp.status_code == 200
        data = resp.json()
        assert 'device_tree' in data

    def test_post_config(self):
        # Non-empty tree — empty trees are refused by ConfigService.save
        # safety guard (prevents accidental wipe of device config).
        #
        # IMPORTANT: mock ConfigService.save so the test does NOT overwrite
        # the real device_config.json on disk.  The view's behaviour (status
        # code + response shape) is still fully exercised; only the file
        # write is intercepted.
        body = {"device_tree": [
            {"id": "t1", "name": "S1", "type": "sensor",
             "sourceId": "virtual:sine", "channelName": "VS-sine", "blockSize": 512},
        ], "aggregation_strategy": "min"}
        with patch.object(services_bridge.get_container().config, 'save',
                          return_value={"status": "ok"}) as mock_save:
            resp = self.client.post('/api/config', data=json.dumps(body), content_type='application/json')
            assert resp.status_code == 200
            assert resp.json()['status'] == 'ok'
            mock_save.assert_called_once()

    def test_post_empty_config_refused(self):
        """Empty device_tree must be refused — safety guard against config wipe."""
        body = {"device_tree": [], "aggregation_strategy": "min"}
        resp = self.client.post('/api/config', data=json.dumps(body), content_type='application/json')
        assert resp.status_code == 200
        assert resp.json()['status'] == 'error'

    def test_config_method_not_allowed(self):
        resp = self.client.delete('/api/config')
        assert resp.status_code == 405

    def test_reset(self):
        resp = self.client.post('/api/reset')
        assert resp.status_code == 200
        assert resp.json()['status'] == 'ok'

    def test_health(self):
        resp = self.client.get('/api/health')
        assert resp.status_code == 200
        data = resp.json()
        assert 'system' in data
        assert 'channels' in data

    def test_sensors(self):
        resp = self.client.get('/api/sensors')
        assert resp.status_code == 200
        data = resp.json()
        assert 'sensors' in data
        assert 'system_health' in data


@pytest.mark.django_db
class TestAlertsAndWarnings(TestCase):
    """Second batch: alerts (4) + warnings (3)."""

    def setUp(self):
        # apps.py ready() only fires under runserver (RUN_MAIN=true),
        # so under pytest the container must be started explicitly.
        services_bridge.start()
        self.client = Client()

    def test_get_alerts(self):
        resp = self.client.get('/api/alerts')
        assert resp.status_code == 200
        data = resp.json()
        assert 'alerts' in data
        assert 'threshold' in data

    def test_get_alerts_history(self):
        resp = self.client.get('/api/alerts/history')
        assert resp.status_code == 200
        data = resp.json()
        assert 'alerts' in data

    def test_alert_verdict_422_on_invalid(self):
        body = {"channel": "C-1", "alert_ts": 1700000000.0, "human_verdict": "bad"}
        resp = self.client.post(
            '/api/alerts/verdict', data=json.dumps(body), content_type='application/json'
        )
        assert resp.status_code == 422

    def test_warning_verdict_422_on_invalid(self):
        body = {"human_verdict": "bad"}
        resp = self.client.post(
            '/api/warnings/1/verdict', data=json.dumps(body), content_type='application/json'
        )
        assert resp.status_code == 422

    def test_get_warnings(self):
        resp = self.client.get('/api/warnings')
        assert resp.status_code == 200
        assert 'warnings' in resp.json()

    def test_get_predict_scores(self):
        resp = self.client.get('/api/predict-scores', {'channel': 'C-1'})
        assert resp.status_code == 200
        data = resp.json()
        assert 'timestamps' in data
        assert 'scores' in data


@pytest.mark.django_db
class TestHistoryWindowExport(TestCase):
    """Third batch: history (5) + window (1) + export (1)."""

    def setUp(self):
        # apps.py ready() only fires under runserver (RUN_MAIN=true),
        # so under pytest the container must be started explicitly.
        services_bridge.start()
        self.client = Client()

    def test_get_history(self):
        resp = self.client.get('/api/history')
        assert resp.status_code == 200
        assert 'count' in resp.json()

    def test_delete_history_confirm_guard(self):
        resp = self.client.delete('/api/history')
        assert resp.status_code == 400
        assert resp.json()['error'] == 'confirm_required'

    def test_get_detection(self):
        resp = self.client.get('/api/detection')
        assert resp.status_code == 200
        assert 'count' in resp.json()

    def test_delete_detection_confirm_guard(self):
        resp = self.client.delete('/api/detection')
        assert resp.status_code == 400
        assert resp.json()['error'] == 'confirm_required'

    def test_get_db_stats(self):
        resp = self.client.get('/api/db-stats')
        assert resp.status_code == 200

    def test_get_window(self):
        resp = self.client.get('/api/window', {'channel': 'C-1', 'count': 10})
        assert resp.status_code == 200


@pytest.mark.django_db
class TestDiagnosis(TestCase):
    """Fourth batch: diagnosis (4)."""

    def setUp(self):
        # apps.py ready() only fires under runserver (RUN_MAIN=true),
        # so under pytest the container must be started explicitly.
        services_bridge.start()
        self.client = Client()

    def test_diagnosis_done(self):
        resp = self.client.get('/api/diagnosis/done')
        assert resp.status_code == 200
        assert 'done' in resp.json()

    def test_diagnosis_auto_status(self):
        resp = self.client.get('/api/diagnosis/auto/status')
        assert resp.status_code == 200
        data = resp.json()
        assert 'running' in data
        assert 'done' in data
        assert 'total' in data

    def test_diagnosis_422_on_invalid(self):
        body = {"channel": ""}  # missing required channel
        resp = self.client.post(
            '/api/diagnosis', data=json.dumps(body), content_type='application/json'
        )
        assert resp.status_code == 422


@pytest.mark.django_db
class TestRul(TestCase):
    """Fifth batch: RUL degradation prediction."""

    def setUp(self):
        services_bridge.start()
        self.client = Client()

    def test_get_rul_returns_200_or_503(self):
        # Status depends on whether C-MAPSS data + weights are present in the
        # test environment.  Either is acceptable — the contract is:
        #   200 {"status":"ok","data":[...]} when enabled
        #   503 {"status":"disabled",...} when assets missing
        resp = self.client.get('/api/rul')
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data['status'] in ('ok', 'disabled')

    def test_get_rul_503_when_service_none(self):
        """When c.rul is None the view returns 503 with a helpful message."""
        c = services_bridge.get_container()
        if c.rul is not None:
            pytest.skip("RUL service is enabled — 503 path not exercisable here")
        resp = self.client.get('/api/rul')
        assert resp.status_code == 503
        assert resp.json()['status'] == 'disabled'

    def test_get_rul_data_shape_when_enabled(self):
        """When RUL is enabled, /api/rul returns a list of per-channel dicts."""
        c = services_bridge.get_container()
        if c.rul is None:
            pytest.skip("RUL service disabled — data-shape test needs assets")
        resp = self.client.get('/api/rul')
        assert resp.status_code == 200
        data = resp.json()['data']
        assert isinstance(data, list)
        if data:
            r = data[0]
            for key in ('channel', 'rul', 'max_rul', 'unit', 'model', 'source', 'history'):
                assert key in r, f"missing field {key}"

    def test_get_rul_single_channel(self):
        """?channel=xxx returns a single result (or null if not tagged)."""
        c = services_bridge.get_container()
        if c.rul is None:
            pytest.skip("RUL service disabled")
        resp = self.client.get('/api/rul', {'channel': 'CMAPSS_FD001_1'})
        assert resp.status_code == 200
        data = resp.json()['data']
        assert data is None or isinstance(data, dict)

    def test_rul_method_not_allowed(self):
        resp = self.client.post('/api/rul', data='{}', content_type='application/json')
        assert resp.status_code == 405
