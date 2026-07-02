"""Unit tests for i18n module — no model dependency, fast."""

from core.i18n import t, LANGS


class TestTranslations:
    """Tests for the translation system."""

    def test_langs_contains_zh_and_en(self):
        assert "zh" in LANGS
        assert "en" in LANGS

    def test_get_chinese(self):
        result = t("title", "zh")
        assert "空间有效载荷" in result

    def test_get_english(self):
        result = t("title", "en")
        assert "Space Payload" in result

    def test_format_args(self):
        result = t("tspulse_loaded", "zh").format(1.1)
        assert "1.1" in result

    def test_format_args_en(self):
        result = t("tspulse_loaded", "en").format(1.1)
        assert "1.1" in result

    def test_missing_key_returns_key(self):
        result = t("nonexistent_key_12345", "zh")
        assert result == "nonexistent_key_12345"

    def test_all_keys_have_both_languages(self):
        """Every key in the dictionary must have both zh and en."""
        from core.i18n import _STRINGS

        missing = []
        for key, entry in _STRINGS.items():
            if "zh" not in entry:
                missing.append(f"{key}: missing zh")
            if "en" not in entry:
                missing.append(f"{key}: missing en")
        assert not missing, f"Keys with missing translations: {missing}"

    def test_channel_info_format(self):
        """The channel_info_fmt key should accept 3 format args."""
        result = t("channel_info_fmt", "zh").format("C-1", "NASA-MSL", 2048)
        assert "C-1" in result
        assert "NASA-MSL" in result
        assert "2048" in result
