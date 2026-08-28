"""题目图片 OCR 与用户校正的可替换服务层。

服务只接收内存中的 PNG、JPEG 或 WebP 字节，使用固定参数调用本地 OCR 引擎，
不执行用户提供的文件路径或命令。部署环境可注入其他经过审核的 OCR Provider。
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

_LOGIC_SENSITIVE_TERMS = (
    "当且仅当",
    "只要",
    "只有",
    "如果",
    "除非",
    "否则",
    "至少",
    "至多",
    "恰好",
    "不是",
    "所有",
    "有些",
    "并且",
    "或者",
    "同时",
    "若",
    "不",
    "才",
)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class OcrUnavailableError(RuntimeError):
    """表示部署环境当前没有可用的 OCR Provider。"""


class OcrProvider(Protocol):
    """OCR 引擎适配协议，便于替换本地或经过审核的服务实现。"""

    @property
    def name(self) -> str:
        """返回给用户侧的 Provider 名称。"""

    def extract_text(self, image_bytes: bytes) -> str:
        """从已校验的图片字节中提取原始文本。"""


@dataclass(frozen=True, slots=True)
class OcrExtraction:
    """一次 OCR 提取的文本与需要用户关注的提示。"""

    provider: str
    text: str
    image_type: str
    warnings: tuple[str, ...]
    critical_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OcrCorrectionResult:
    """用户校正后的可用文本及关键逻辑词变更提示。"""

    corrected_text: str
    requires_confirmation: bool
    changed_critical_terms: tuple[str, ...]
    warnings: tuple[str, ...]


class TesseractCliProvider:
    """通过固定的 Tesseract CLI 参数进行本地 OCR。"""

    def __init__(
        self,
        language: str = "chi_sim+eng",
        timeout_seconds: int = 15,
    ) -> None:
        self._language = language
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        """标识当前本地 OCR 引擎。"""
        return "tesseract"

    def extract_text(self, image_bytes: bytes) -> str:
        """通过标准输入传图，避免创建临时文件和处理用户指定路径。"""
        binary = shutil.which("tesseract")
        if binary is None:
            raise OcrUnavailableError(
                "本地 OCR 引擎未安装，当前无法识别图片。请配置受支持的 OCR Provider。"
            )

        try:
            completed = subprocess.run(
                [
                    binary,
                    "stdin",
                    "stdout",
                    "-l",
                    self._language,
                    "--psm",
                    "6",
                ],
                input=image_bytes,
                capture_output=True,
                check=False,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            timeout_message = "图片识别超时，请改用更清晰或更小的图片。"
            raise OcrUnavailableError(timeout_message) from error

        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            fallback_message = "OCR 引擎未返回结果"
            raise OcrUnavailableError(f"图片识别失败：{message or fallback_message}")
        return completed.stdout.decode("utf-8", errors="replace").strip()


class OcrService:
    """完成图片校验、OCR 提取和关键逻辑词风险标记。"""

    def __init__(self, provider: OcrProvider | None = None) -> None:
        self._provider = provider or TesseractCliProvider()

    def extract_from_base64(self, image_base64: str) -> OcrExtraction:
        """解码 Base64 图片后进行 OCR；不接受任意文件路径。"""
        image_bytes = self._decode_base64_image(image_base64)
        return self.extract(image_bytes)

    def extract(self, image_bytes: bytes) -> OcrExtraction:
        """识别经过格式和大小校验的图片字节。"""
        image_type = self._detect_image_type(image_bytes)
        text = self._provider.extract_text(image_bytes)
        critical_terms = _find_critical_terms(text)
        warnings: list[str] = []
        if not text:
            warnings.append("未识别到文字，请上传更清晰的题目图片或手动输入题干。")
        if critical_terms:
            terms = "、".join(critical_terms)
            warnings.append(f"请核对关键逻辑词：{terms}。")
        return OcrExtraction(
            provider=self._provider.name,
            text=text,
            image_type=image_type,
            warnings=tuple(warnings),
            critical_terms=critical_terms,
        )

    @staticmethod
    def _decode_base64_image(image_base64: str) -> bytes:
        normalized = image_base64.strip()
        if normalized.startswith("data:"):
            try:
                _, normalized = normalized.split(",", maxsplit=1)
            except ValueError as error:
                raise ValueError("图片 Data URL 格式不完整") from error
        try:
            image_bytes = base64.b64decode(normalized, validate=True)
        except ValueError as error:
            raise ValueError("图片必须是有效的 Base64 编码") from error
        if not image_bytes:
            raise ValueError("图片内容不能为空")
        return image_bytes

    @staticmethod
    def _detect_image_type(image_bytes: bytes) -> str:
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError("图片不能超过 10MB")
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "webp"
        raise ValueError("仅支持 PNG、JPEG 或 WebP 格式的图片")


class OcrCorrectionService:
    """核对用户修订的 OCR 文本，突出逻辑敏感词的改动。"""

    def review(self, original_text: str, corrected_text: str) -> OcrCorrectionResult:
        """返回用户确认后的文本和需要再次确认的关键字变更。"""
        original = original_text.strip()
        corrected = corrected_text.strip()
        if not corrected:
            raise ValueError("校正后的题干不能为空")
        if len(corrected) > 100_000:
            raise ValueError("校正后的题干不能超过 100000 个字符")

        changed_terms = _changed_critical_terms(original, corrected)
        warnings: list[str] = []
        if changed_terms:
            terms = "、".join(changed_terms)
            warnings.append(f"已修改关键逻辑词：{terms}。请确认修改符合原图。")
        return OcrCorrectionResult(
            corrected_text=corrected,
            requires_confirmation=bool(changed_terms),
            changed_critical_terms=changed_terms,
            warnings=tuple(warnings),
        )


def _find_critical_terms(text: str) -> tuple[str, ...]:
    """按固定优先级提取文本中出现的关键逻辑词。"""
    return tuple(term for term in _LOGIC_SENSITIVE_TERMS if term in text)


def _changed_critical_terms(original: str, corrected: str) -> tuple[str, ...]:
    """比较校正前后的关键逻辑词次数，避免遗漏影响推理方向的修改。"""
    changed: list[str] = []
    for term in _LOGIC_SENSITIVE_TERMS:
        if original.count(term) != corrected.count(term):
            changed.append(term)
    return tuple(changed)
