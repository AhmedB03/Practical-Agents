"""LLM-as-judge for answer correctness and faithfulness.

- correctness: does the answer agree with the reference answer? (0/1)
- faithfulness: is every claim in the answer supported by the retrieved context,
  i.e. no hallucination? (0/1)

With an API provider these are graded by the judge model with a strict rubric.
With the local provider they are graded by a transparent lexical heuristic, so
the harness produces a full report offline. The heuristic is deliberately
conservative and documented as an approximation in the report.
"""
from __future__ import annotations

import json
import re

from src.practical_agents.llm import LLM, ABSTAIN

_CORRECTNESS_PROMPT = """You grade a documentation answer against a reference.
Return JSON: {{"correct": true|false, "reason": "<short>"}}.
Mark correct=true only if the answer conveys the same key facts as the reference,
even if worded differently. Missing or contradicting a key fact => false.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""

_FAITHFULNESS_PROMPT = """You check whether an answer is fully supported by the context.
Return JSON: {{"faithful": true|false, "reason": "<short>"}}.
faithful=true only if every factual claim in the answer appears in the context.
An explicit "I don't have enough information" answer is always faithful.

CONTEXT:
{context}

ANSWER: {answer}"""


class Judge:
    def __init__(self, settings):
        self.settings = settings
        self.llm = LLM(settings)
        self.is_local = settings.is_local_llm

    def score(self, question, reference, answer, context, answerable) -> dict:
        if self.is_local:
            return self._heuristic(question, reference, answer, context, answerable)
        return self._llm(question, reference, answer, context)  # pragma: no cover

    # -- API judge -------------------------------------------------------
    def _llm(self, question, reference, answer, context):  # pragma: no cover
        c = self.llm.chat(
            "You are a strict, fair grader. Reply only with JSON.",
            _CORRECTNESS_PROMPT.format(
                question=question, reference=reference, answer=answer
            ),
            model=self.settings.judge_model,
        )
        f = self.llm.chat(
            "You are a strict, fair grader. Reply only with JSON.",
            _FAITHFULNESS_PROMPT.format(context=context, answer=answer),
            model=self.settings.judge_model,
        )
        return {
            "correctness": 1.0 if _parse(c["text"], "correct") else 0.0,
            "faithfulness": 1.0 if _parse(f["text"], "faithful") else 0.0,
            "judge_tokens": c["prompt_tokens"] + c["completion_tokens"]
            + f["prompt_tokens"] + f["completion_tokens"],
        }

    # -- local heuristic judge ------------------------------------------
    def _heuristic(self, question, reference, answer, context, answerable):
        abstained = ABSTAIN[:24].lower() in answer.lower()
        if not answerable:
            # Correct iff it abstained (or otherwise conveys "not supported").
            correct = 1.0 if abstained else 0.0
            faithful = 1.0 if abstained else _supported(answer, context)
            return {"correctness": correct, "faithfulness": faithful, "judge_tokens": 0}
        # Answerable: correctness = enough reference key-terms present in answer.
        ref_terms = _key_terms(reference)
        ans_terms = _key_terms(answer)
        coverage = len(ref_terms & ans_terms) / (len(ref_terms) or 1)
        correct = 1.0 if coverage >= 0.5 and not abstained else 0.0
        faithful = _supported(answer, context)
        return {"correctness": correct, "faithfulness": faithful, "judge_tokens": 0}


def _supported(answer: str, context: str) -> float:
    """Fraction-of-claim-terms-in-context proxy, thresholded to 0/1."""
    if ABSTAIN[:24].lower() in answer.lower():
        return 1.0
    a = _key_terms(answer)
    c = _key_terms(context)
    if not a:
        return 1.0
    return 1.0 if len(a & c) / len(a) >= 0.7 else 0.0


_STOP = set(
    "the a an of to in and or is are for with on by as at be it this that from you "
    "your can do does what how when where which who why into per not no more than "
    "goes straight its it's after before an one".split()
)


def _key_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_.]+", text.lower()) if w not in _STOP and len(w) > 1}


def _parse(text: str, key: str) -> bool:  # pragma: no cover
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        return bool(json.loads(m.group(0)).get(key))
    except Exception:
        return key in text.lower() and "true" in text.lower()
