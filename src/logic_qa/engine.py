"""命题逻辑的确定性推理内核。

内核只接受结构化命题，不负责把自然语言直接猜测成逻辑规则。这样可以确保
每个结论都能回溯到已给出的事实、规则或等价的逆否规则。
"""

from __future__ import annotations

from collections.abc import Iterable

from logic_qa.models import (
    ImplicationRule,
    Literal,
    ProofStep,
    VerificationResult,
    VerificationStatus,
)


class InferenceEngine:
    """使用前向链式推理和逆否规则推导可证明结论。"""

    def verify(
        self,
        facts: Iterable[Literal],
        rules: Iterable[ImplicationRule],
        query: Literal,
    ) -> VerificationResult:
        """验证查询是否能从事实与规则集中推导。

        每条 `P → Q` 会同时生成逆否规则 `¬Q → ¬P`，因为二者逻辑等价。闭包完成
        后会检测整个知识库是否存在互为否定的命题：一旦发现矛盾，任何查询都
        返回 INCONSISTENT，避免把建立在矛盾条件上的结论伪装为可靠结果。
        """
        normalized_rules = tuple(rules)
        all_rules = self._with_contrapositives(normalized_rules)
        proof_by_literal = self._seed_facts(facts)
        known = set(proof_by_literal)

        changed = True
        while changed:
            changed = False
            for rule, is_contrapositive in all_rules:
                if rule.premise not in known or rule.conclusion in known:
                    continue

                reason = "逆否推理" if is_contrapositive else "条件推理"
                proof_by_literal[rule.conclusion] = ProofStep(
                    derived=rule.conclusion,
                    reason=reason,
                    source_rule=rule,
                    dependencies=(rule.premise,),
                )
                known.add(rule.conclusion)
                changed = True

        conflict = self._detect_conflict(known, preferred=query)
        status = self._resolve_status(query, known, conflict)
        proof_steps = self._build_proof_trace(query, proof_by_literal, status, conflict)
        sorted_literals = tuple(sorted(known, key=lambda item: item.display()))
        return VerificationResult(
            query=query,
            status=status,
            proof_steps=proof_steps,
            known_literals=sorted_literals,
            conflict=conflict,
        )

    @staticmethod
    def _seed_facts(facts: Iterable[Literal]) -> dict[Literal, ProofStep]:
        proof_by_literal: dict[Literal, ProofStep] = {}
        for fact in facts:
            proof_by_literal.setdefault(
                fact,
                ProofStep(derived=fact, reason="题干事实"),
            )
        return proof_by_literal

    @staticmethod
    def _with_contrapositives(
        rules: tuple[ImplicationRule, ...],
    ) -> tuple[tuple[ImplicationRule, bool], ...]:
        pairs: list[tuple[ImplicationRule, bool]] = []
        for rule in rules:
            pairs.append((rule, False))
            pairs.append(
                (
                    ImplicationRule(
                        premise=rule.conclusion.opposite(),
                        conclusion=rule.premise.opposite(),
                        source_text=rule.source_text,
                    ),
                    True,
                )
            )
        return tuple(pairs)

    @staticmethod
    def _detect_conflict(
        known: set[Literal],
        preferred: Literal,
    ) -> tuple[Literal, Literal] | None:
        """检测知识库中是否存在互为否定的命题对。

        若查询本身冲突，优先返回该冲突对，便于用户理解当前结论为何不可靠；否则
        按展示名排序遍历并返回第一对全局冲突文字，保证响应结果确定。
        """
        if preferred in known and preferred.opposite() in known:
            return preferred, preferred.opposite()

        for literal in sorted(known, key=lambda item: item.display()):
            if literal.opposite() in known:
                return literal, literal.opposite()
        return None

    @staticmethod
    def _resolve_status(
        query: Literal,
        known: set[Literal],
        conflict: tuple[Literal, Literal] | None,
    ) -> VerificationStatus:
        if conflict is not None:
            return VerificationStatus.INCONSISTENT
        if query in known:
            return VerificationStatus.PROVED
        if query.opposite() in known:
            return VerificationStatus.DISPROVED
        return VerificationStatus.UNKNOWN

    def _build_proof_trace(
        self,
        query: Literal,
        proof_by_literal: dict[Literal, ProofStep],
        status: VerificationStatus,
        conflict: tuple[Literal, Literal] | None,
    ) -> tuple[ProofStep, ...]:
        if status is VerificationStatus.UNKNOWN:
            return ()

        target = query
        if status is VerificationStatus.DISPROVED:
            target = query.opposite()

        trace: list[ProofStep] = []
        visited: set[Literal] = set()

        def visit(literal: Literal) -> None:
            if literal in visited:
                return
            visited.add(literal)
            step = proof_by_literal.get(literal)
            if step is None:
                return
            for dependency in step.dependencies:
                visit(dependency)
            trace.append(step)

        visit(target)

        if status is VerificationStatus.INCONSISTENT:
            # 优先展示与查询直接相关的矛盾；否则展示首个被检测到的冲突对。
            if query in proof_by_literal and query.opposite() in proof_by_literal:
                visit(query.opposite())
            elif conflict is not None:
                visit(conflict[0])
                visit(conflict[1])
        return tuple(trace)
