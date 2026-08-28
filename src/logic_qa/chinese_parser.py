"""受控中文条件解析与复杂语义人工确认服务。

本模块只自动转换明确且可形式化的中文句式。对于析取、合取、除非、量词及
非标准否定等复杂语义，先返回待确认项；只有调用方提交结构化的人工确认结果后，
这些条件才会进入确定性逻辑内核。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from logic_qa.models import ImplicationRule, Literal

_SENTENCE_SPLITTER = re.compile(r"[。；;\n]+")
_CONDITIONAL_PATTERNS = (
    re.compile(r"^如果(?P<premise>[^，,]+)[，,]那么(?P<conclusion>.+)$"),
    re.compile(r"^若(?P<premise>[^，,]+)[，,]则(?P<conclusion>.+)$"),
    re.compile(r"^只要(?P<premise>[^，,]+)[，,]就(?P<conclusion>.+)$"),
)
_NECESSARY_PATTERN = re.compile(
    r"^只有(?P<necessary>[^，,]+)[，,](?P<subject>[^才]+)才(?P<state>.+)$"
)
_BICONDITIONAL_PATTERN = re.compile(
    r"^当且仅当(?P<left>[^，,]+)[，,](?P<right>.+)$"
)
_CONFIRMATION_MARKERS = (
    ("除非", "unless"),
    ("否则", "unless"),
    ("至少", "quantifier"),
    ("至多", "quantifier"),
    ("并且", "conjunction"),
    ("同时", "conjunction"),
    ("或者", "disjunction"),
    ("不是", "non_standard_negation"),
    ("未", "non_standard_negation"),
    ("无", "non_standard_negation"),
    ("非", "non_standard_negation"),
    ("且", "conjunction"),
    ("或", "disjunction"),
)
_CONFIRMATION_MESSAGES = {
    "unless": "“除非/否则”可能存在条件方向或范围歧义。",
    "quantifier": "数量或范围量词超出当前命题逻辑内核的自动表达范围。",
    "conjunction": "合取条件需要确认其是否应作为共同前提或独立事实。",
    "disjunction": "析取条件需要确认是包含性析取、互斥析取还是题目中的普通用语。",
    "non_standard_negation": "该否定表达不能安全映射为主体与状态间的单个“不”。",
}


class ChineseParseError(ValueError):
    """表示题干不符合自动解析或人工确认的输入要求。"""


@dataclass(frozen=True, slots=True)
class ParsedChineseText:
    """中文文本自动或人工确认后得到的事实、规则与原句映射。"""

    facts: tuple[Literal, ...]
    rules: tuple[ImplicationRule, ...]
    source_sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChineseConfirmationRequest:
    """需要人工将复杂中文语义转写为结构化条件的单个原句。"""

    source_sentence: str
    codes: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class ChineseAnalysis:
    """自动可解析内容与需要人工确认的内容分离结果。"""

    parsed: ParsedChineseText
    confirmation_requests: tuple[ChineseConfirmationRequest, ...]


@dataclass(frozen=True, slots=True)
class ConfirmedChineseSentence:
    """人工确认后提交的单句结构化事实与规则。"""

    source_sentence: str
    facts: tuple[Literal, ...] = ()
    rules: tuple[ImplicationRule, ...] = ()


class ControlledChineseParser:
    """将受控中文条件转换为命题逻辑对象，并隔离复杂语义。"""

    def analyze(self, text: str) -> ChineseAnalysis:
        """解析安全句式，并返回必须由人工转写确认的复杂条件。"""
        sentences = self._split_sentences(text)
        facts: list[Literal] = []
        rules: list[ImplicationRule] = []
        confirmation_requests: list[ChineseConfirmationRequest] = []

        for sentence in sentences:
            confirmation_request = (
                None
                if _BICONDITIONAL_PATTERN.fullmatch(sentence)
                else self._confirmation_request(sentence)
            )
            if confirmation_request is not None:
                confirmation_requests.append(confirmation_request)
                continue

            parsed_rules = self._parse_rules(sentence)
            if parsed_rules is not None:
                rules.extend(parsed_rules)
            else:
                facts.append(self._parse_literal(sentence))

        return ChineseAnalysis(
            parsed=ParsedChineseText(
                facts=tuple(facts),
                rules=tuple(rules),
                source_sentences=sentences,
            ),
            confirmation_requests=tuple(confirmation_requests),
        )

    def parse(self, text: str) -> ParsedChineseText:
        """仅解析不含复杂语义的中文条件，否则要求先完成人工确认。"""
        analysis = self.analyze(text)
        if analysis.confirmation_requests:
            request = analysis.confirmation_requests[0]
            raise ChineseParseError(
                f"需要人工确认：“{request.source_sentence}”。{request.message}"
            )
        return analysis.parsed

    def parse_with_confirmations(
        self,
        text: str,
        confirmations: tuple[ConfirmedChineseSentence, ...],
    ) -> ParsedChineseText:
        """将人工确认的结构化语义与自动解析内容合并为确定性输入。"""
        analysis = self.analyze(text)
        required_by_sentence = {
            request.source_sentence: request
            for request in analysis.confirmation_requests
        }
        confirmed_by_sentence: dict[str, ConfirmedChineseSentence] = {}
        for confirmation in confirmations:
            source_sentence = confirmation.source_sentence.strip()
            if source_sentence not in required_by_sentence:
                raise ChineseParseError(
                    f"“{source_sentence}”不是当前题干中需要确认的复杂条件。"
                )
            if source_sentence in confirmed_by_sentence:
                raise ChineseParseError(f"复杂条件不能重复确认：“{source_sentence}”。")
            if not confirmation.facts and not confirmation.rules:
                raise ChineseParseError(
                    f"人工确认必须提供至少一条事实或规则：“{source_sentence}”。"
                )
            confirmed_by_sentence[source_sentence] = confirmation

        missing_sentences = tuple(
            sentence
            for sentence in required_by_sentence
            if sentence not in confirmed_by_sentence
        )
        if missing_sentences:
            missing = "；".join(missing_sentences)
            raise ChineseParseError(f"以下复杂条件尚未人工确认：{missing}。")

        facts = list(analysis.parsed.facts)
        rules = list(analysis.parsed.rules)
        for sentence in analysis.parsed.source_sentences:
            confirmation = confirmed_by_sentence.get(sentence)
            if confirmation is None:
                continue
            facts.extend(confirmation.facts)
            rules.extend(
                ImplicationRule(
                    premise=rule.premise,
                    conclusion=rule.conclusion,
                    source_text=f"{sentence}（人工确认）",
                )
                for rule in confirmation.rules
            )

        return ParsedChineseText(
            facts=tuple(facts),
            rules=tuple(rules),
            source_sentences=analysis.parsed.source_sentences,
        )

    def parse_query(self, text: str) -> Literal:
        """解析单个查询命题，不接受复杂语义或句子分隔符。"""
        normalized = text.strip()
        if any(separator in normalized for separator in "。；;\n"):
            raise ChineseParseError("查询只能包含一个不带句末标点的命题。")
        confirmation_request = self._confirmation_request(normalized)
        if confirmation_request is not None:
            raise ChineseParseError(
                f"查询不能包含复杂语义：“{normalized}”。{confirmation_request.message}"
            )
        return self._parse_literal(normalized)

    @staticmethod
    def _split_sentences(text: str) -> tuple[str, ...]:
        fragments = _SENTENCE_SPLITTER.split(text)
        normalized_sentences = [fragment.strip() for fragment in fragments]
        sentences = tuple(sentence for sentence in normalized_sentences if sentence)
        if not sentences:
            raise ChineseParseError("题干不能为空，请输入至少一个完整的受支持条件。")
        return sentences

    @staticmethod
    def _confirmation_request(sentence: str) -> ChineseConfirmationRequest | None:
        matched_codes: list[str] = []
        for marker, code in _CONFIRMATION_MARKERS:
            if marker not in sentence:
                continue
            if marker == "非" and ("除非" in sentence or "否则" in sentence):
                continue
            matched_codes.append(code)
        codes = tuple(dict.fromkeys(matched_codes))
        if not codes:
            return None
        messages = "".join(_CONFIRMATION_MESSAGES[code] for code in codes)
        return ChineseConfirmationRequest(
            source_sentence=sentence,
            codes=codes,
            message=messages,
        )

    def _parse_rules(self, sentence: str) -> tuple[ImplicationRule, ...] | None:
        for pattern in _CONDITIONAL_PATTERNS:
            match = pattern.fullmatch(sentence)
            if match:
                return (
                    ImplicationRule(
                        premise=self._parse_literal(match.group("premise")),
                        conclusion=self._parse_literal(match.group("conclusion")),
                        source_text=sentence,
                    ),
                )

        match = _NECESSARY_PATTERN.fullmatch(sentence)
        if match:
            necessary = self._parse_literal(match.group("necessary"))
            result = self._parse_literal(
                f"{match.group('subject')}{match.group('state')}"
            )
            return (
                ImplicationRule(
                    premise=result,
                    conclusion=necessary,
                    source_text=sentence,
                ),
            )

        match = _BICONDITIONAL_PATTERN.fullmatch(sentence)
        if match:
            left = self._parse_literal(match.group("left"))
            right = self._parse_literal(match.group("right"))
            return (
                ImplicationRule(
                    premise=left,
                    conclusion=right,
                    source_text=sentence,
                ),
                ImplicationRule(
                    premise=right,
                    conclusion=left,
                    source_text=sentence,
                ),
            )

        if sentence.startswith(("如果", "若", "只要", "只有", "当且仅当")):
            raise ChineseParseError(
                f"条件句格式不完整：“{sentence}”。请使用 README 中列出的受支持句式。"
            )
        return None

    @staticmethod
    def _parse_literal(text: str) -> Literal:
        """解析一个原子命题，且仅识别主体与状态间的单个“不”。"""
        normalized = text.strip()
        invalid_characters = "，,。；;"
        if (
            not normalized
            or any(character.isspace() for character in normalized)
            or any(character in invalid_characters for character in normalized)
        ):
            raise ChineseParseError(
                f"无法识别命题：“{text}”。请使用不含空格和标点的简单短语。"
            )

        if "不" not in normalized:
            return Literal(symbol=normalized)

        if normalized.count("不") != 1:
            raise ChineseParseError(f"否定命题只能包含一个“不”：“{text}”。")

        subject, state = normalized.split("不", maxsplit=1)
        if not subject or not state:
            raise ChineseParseError(f"命题不完整：“{text}”。")
        return Literal(symbol=f"{subject}{state}", negated=True)
