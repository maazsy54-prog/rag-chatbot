"""Input and output guardrails.

Input: reject empty, over-long and prompt-injection / jailbreak questions before
any paid call is made. Output: never let an answer leak API keys or the system
prompt. Pattern matching cannot catch every attack, so it is one layer among
several: the system prompt, source tagging and citation checking are the others.
"""

from __future__ import annotations

import re

from app.config import settings


class GuardrailViolation(Exception):
    """A request or answer was blocked by a guardrail.

    Attributes:
        code: Short machine-readable reason, e.g. ``"prompt_injection"``.
        message: Explanation that is safe to show the user.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_INJECTION_PATTERNS = [
    # "ignore all previous instructions", "forget your rules", "disregard the above prompt"
    r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|preceding|all|any|your)\b.{0,30}\b(instructions?|prompts?|rules?|guidelines?|training|messages?|restrictions?)\b",
    # asking the assistant to reveal its own prompt or rules
    r"\b(reveal|show|print|repeat|display|output|leak|tell me|give me|what (is|are|were))\b.{0,30}\b(system|hidden)\s+(prompt|instructions?|message|rules)\b",
    r"\b(reveal|show|print|repeat|display|output|leak|tell me|give me|what (is|are|were))\b.{0,30}\byour\s+(initial|original|secret|hidden|system)?\s*(prompt|instructions?|rules)\b",
    # role-change and jailbreak phrases
    r"\byou are (now )?(dan|in developer mode|no longer bound|free from|unrestricted)\b",
    r"\b(developer|god|admin|sudo|jailbreak|dan) mode\b",
    r"\bdo anything now\b",
    r"\bpretend (to be|that you (are|have no)|you (are|have no))\b",
    r"\b(act|behave|respond) (as|like) (if )?(you (are|were) )?(an? )?(unrestricted|unfiltered|uncensored|evil)\b",
    r"\b(you|assistant|ai|chatbot)\b.{0,40}\bwithout (any )?(restrictions|filters|rules|limitations|guidelines|safety)\b",
    r"\b(bypass|override|disable|turn off|remove)\b.{0,20}\byour (safety|guardrails?|filters?|restrictions)\b",
    # asking for secrets
    r"\byour (api[ _-]?keys?|secret keys?|access tokens?|credentials|passwords?)\b",
    # tag injection
    r"</?\s*(system|sources?|assistant)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)

_SECRET_PATTERNS = [re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), re.compile(r"\bpa-[A-Za-z0-9_\-]{20,}")]
_SHINGLE_WORDS = 10  # a copied run of this many prompt words counts as a leak


def contains_injection(text: str) -> bool:
    """Return ``True`` if the text looks like a prompt-injection or jailbreak attempt."""
    return _INJECTION_RE.search(text) is not None


def check_question(question: str) -> str:
    """Validate and normalise a user question.

    Returns:
        The question with whitespace collapsed.

    Raises:
        GuardrailViolation: With code ``empty_question``, ``question_too_long``
            or ``prompt_injection``.
    """
    cleaned = " ".join(question.split())
    if not cleaned:
        raise GuardrailViolation("empty_question", "Please enter a question.")
    if len(cleaned) > settings.max_question_chars:
        raise GuardrailViolation(
            "question_too_long",
            f"The question is too long (limit: {settings.max_question_chars} characters).",
        )
    if contains_injection(cleaned):
        raise GuardrailViolation(
            "prompt_injection",
            "This request looks like an attempt to change the assistant's instructions "
            "or reveal internal details, so it was blocked. Ask a question about your documents.",
        )
    return cleaned


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def check_output(answer: str, system_prompt: str) -> str | None:
    """Check a model answer before it reaches the user.

    Returns:
        ``None`` if the answer is safe, otherwise a reason code:
        ``"leaked_secret"`` (an API key pattern or a real configured key) or
        ``"leaked_system_prompt"`` (a long run of words copied from the prompt).
    """
    if any(pattern.search(answer) for pattern in _SECRET_PATTERNS):
        return "leaked_secret"
    for secret in (settings.anthropic_api_key, settings.voyage_api_key):
        if secret and len(secret) > 8 and secret in answer:
            return "leaked_secret"

    prompt_words = _words(system_prompt)
    shingles = {
        " ".join(prompt_words[i : i + _SHINGLE_WORDS])
        for i in range(len(prompt_words) - _SHINGLE_WORDS + 1)
    }
    answer_words = _words(answer)
    for i in range(len(answer_words) - _SHINGLE_WORDS + 1):
        if " ".join(answer_words[i : i + _SHINGLE_WORDS]) in shingles:
            return "leaked_system_prompt"
    return None
