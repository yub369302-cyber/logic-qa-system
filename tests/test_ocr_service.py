"""图片 OCR 与用户校正服务的回归测试。"""

import base64

import pytest

from logic_qa.ocr_service import (
    OcrCorrectionService,
    OcrService,
    OcrUnavailableError,
)

_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


class StubOcrProvider:
    """用于验证服务编排的确定性 OCR Provider。"""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def name(self) -> str:
        return "stub"

    def extract_text(self, image_bytes: bytes) -> str:
        assert image_bytes.startswith(_PNG_HEADER)
        return self._text


def test_extracts_text_and_marks_logic_sensitive_terms() -> None:
    """识别结果应保留原文并提示用户核对关键逻辑词。"""
    service = OcrService(provider=StubOcrProvider("只有甲参加，乙才通过。"))

    result = service.extract(_PNG_HEADER + b"image")

    assert result.provider == "stub"
    assert result.image_type == "png"
    assert result.text == "只有甲参加，乙才通过。"
    assert "只有" in result.critical_terms
    assert "才" in result.critical_terms
    assert any("请核对关键逻辑词" in warning for warning in result.warnings)


def test_accepts_base64_data_url() -> None:
    """接口层可安全传递标准 Data URL，而不需要写入用户指定路径。"""
    encoded = base64.b64encode(_PNG_HEADER + b"image").decode("ascii")
    service = OcrService(provider=StubOcrProvider("甲参加。"))

    result = service.extract_from_base64(f"data:image/png;base64,{encoded}")

    assert result.text == "甲参加。"


@pytest.mark.parametrize(
    "image_bytes",
    [b"", b"plain text", b"GIF89a"],
)
def test_rejects_empty_or_unsupported_images(image_bytes: bytes) -> None:
    """非支持格式的数据不得交给 OCR Provider。"""
    service = OcrService(provider=StubOcrProvider("不应调用"))

    with pytest.raises(ValueError):
        service.extract(image_bytes)


def test_rejects_invalid_base64() -> None:
    """Base64 格式错误时返回明确异常。"""
    service = OcrService(provider=StubOcrProvider("不应调用"))

    with pytest.raises(ValueError, match="Base64"):
        service.extract_from_base64("not-a-base64-image")


def test_correction_requires_confirmation_when_logic_terms_change() -> None:
    """用户替换关键逻辑词时必须要求二次确认。"""
    result = OcrCorrectionService().review(
        original_text="只有甲参加，乙才通过。",
        corrected_text="如果甲参加，那么乙通过。",
    )

    assert result.corrected_text == "如果甲参加，那么乙通过。"
    assert result.requires_confirmation
    assert "只有" in result.changed_critical_terms
    assert "如果" in result.changed_critical_terms


def test_correction_allows_non_logic_text_edits_without_extra_confirmation() -> None:
    """不涉及逻辑词的文字修正不应触发额外确认。"""
    result = OcrCorrectionService().review(
        original_text="甲参加。",
        corrected_text="甲参与。",
    )

    assert result.requires_confirmation is False
    assert result.changed_critical_terms == ()


def test_local_provider_reports_unavailable_when_tesseract_is_missing() -> None:
    """开发环境缺少本地 OCR 引擎时应明确降级，而不是返回虚假文本。"""
    service = OcrService()

    with pytest.raises(OcrUnavailableError, match="本地 OCR 引擎未安装"):
        service.extract(_PNG_HEADER + b"image")
