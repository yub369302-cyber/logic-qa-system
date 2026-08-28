"""OCR 接口与校正接口的集成测试。"""

from fastapi.testclient import TestClient

from logic_qa.api import app

client = TestClient(app)


def test_ocr_api_reports_unavailable_local_provider() -> None:
    """未配置本地 Tesseract 时 API 应返回可解释的服务不可用状态。"""
    response = client.post(
        "/v1/questions/ocr",
        json={"image_base64": "iVBORw0KGgo="},
    )

    assert response.status_code == 503
    assert "本地 OCR 引擎未安装" in response.json()["detail"]


def test_ocr_api_rejects_unsupported_image_payload() -> None:
    """无效图片数据必须在调用 OCR 引擎前被拒绝。"""
    response = client.post(
        "/v1/questions/ocr",
        json={"image_base64": "cGxhaW4gdGV4dA=="},
    )

    assert response.status_code == 422
    assert "仅支持 PNG、JPEG 或 WebP" in response.json()["detail"]


def test_ocr_correction_api_flags_sensitive_logic_change() -> None:
    """OCR 校正接口需要标记影响逻辑方向的关键字改动。"""
    response = client.post(
        "/v1/questions/ocr/correct",
        json={
            "original_text": "只有甲参加，乙才通过。",
            "corrected_text": "如果甲参加，那么乙通过。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requires_confirmation"] is True
    assert "只有" in payload["changed_critical_terms"]
    assert "如果" in payload["changed_critical_terms"]


def test_ocr_correction_api_rejects_empty_result() -> None:
    """用户提交空校正文案时应返回输入校验错误。"""
    response = client.post(
        "/v1/questions/ocr/correct",
        json={"original_text": "甲参加。", "corrected_text": "   "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "校正后的题干不能为空"
