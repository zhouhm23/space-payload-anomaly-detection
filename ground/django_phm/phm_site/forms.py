"""Django forms for request body validation (replaces Pydantic models)."""

from __future__ import annotations

from django import forms

_VALID_VERDICTS = {"real", "false_alarm", "uncertain"}
_VALID_STATUSES = {"active", "pending", "confirmed", "false"}


class VerdictForm(forms.Form):
    """Body for POST /api/warnings/{id}/verdict."""
    human_verdict = forms.CharField(max_length=16)

    def clean_human_verdict(self):
        v = self.cleaned_data['human_verdict']
        if v not in _VALID_VERDICTS:
            raise forms.ValidationError(
                f"human_verdict must be one of {sorted(_VALID_VERDICTS)}"
            )
        return v


class AlertVerdictForm(forms.Form):
    """Body for POST /api/alerts/verdict."""
    channel = forms.CharField(max_length=64)
    alert_ts = forms.FloatField()
    human_verdict = forms.CharField(max_length=16)

    def clean_human_verdict(self):
        v = self.cleaned_data['human_verdict']
        if v not in _VALID_VERDICTS:
            raise forms.ValidationError(
                f"human_verdict must be one of {sorted(_VALID_VERDICTS)}"
            )
        return v


class AlertStatusForm(forms.Form):
    """Body for PATCH /api/alerts/{id}."""
    status = forms.CharField(max_length=16)

    def clean_status(self):
        s = self.cleaned_data['status']
        if s not in _VALID_STATUSES:
            raise forms.ValidationError(
                f"status must be one of {sorted(_VALID_STATUSES)}"
            )
        return s


class DiagnosisForm(forms.Form):
    """Body for POST /api/diagnosis."""
    channel = forms.CharField(max_length=64)
    alert_type = forms.CharField(max_length=16, initial="measured")
    alert_ts = forms.FloatField(required=False)
