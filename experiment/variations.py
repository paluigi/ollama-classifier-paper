"""Shared variation catalog for ``experiment.py`` (writer) and
``analyze_confidence.py`` (reader).

This module is the single source of truth for variation keys, run ordering,
and the adaptive-``generate`` call-budget sweep. Centralizing them here means a
key renamed in one place is automatically consistent in the other — the two
scripts can no longer drift apart silently.

Each :class:`VariationSpec` describes one benchmark variation without reference
to the actual COICOP label data: ``flavor`` tells the consumer *which* choice
set to build (``names_only`` list or ``names_with_desc`` dict), and
``has_opt_out`` tells it whether to append the opt-out label. The data-dependent
label lists/dicts are constructed in ``experiment.py``.
"""

from dataclasses import dataclass

# Call-budget sweep for the adaptive ``generate`` strategy.
MAX_CALLS = [1, 3, 5, 8]


@dataclass(frozen=True)
class VariationSpec:
    """One benchmark variation.

    Attributes:
        key: Stable column prefix written into ``results.xlsx`` and read back by
            the confidence analysis. Also the basis for ``{key}_gen_mc{N}``.
        classifier: ``"bart"``, ``"ollama"``, or ``"skllm"``.
        method: ``"classify"`` or ``"generate"`` for ollama; ``""`` for the
            label-only baselines (BART / scikit-llm).
        flavor: ``"names_only"`` (list of subclass names) or
            ``"names_with_desc"`` (``{name: description}`` dict).
        has_opt_out: Whether the opt-out label is part of this variation's
            candidate set. For BART/ollama it is folded into ``choices`` by the
            caller; for scikit-llm it is passed as ``use_opt_out``.
        max_calls: Call budget for ``generate``; ``None`` otherwise.
    """

    key: str
    classifier: str
    method: str
    flavor: str
    has_opt_out: bool
    max_calls: int | None = None

    def base_display(self) -> str:
        """Human-readable name without the generate call-budget suffix."""
        if self.classifier == "bart":
            return "BART (names + opt-out)" if self.has_opt_out else "BART (names only)"
        if self.classifier == "skllm":
            return (
                "scikit-llm (names + opt-out)"
                if self.has_opt_out
                else "scikit-llm (names only)"
            )
        # ollama
        if self.flavor == "names_with_desc":
            return (
                "Ollama (desc. + opt-out)"
                if self.has_opt_out
                else "Ollama (names + desc.)"
            )
        return "Ollama (names + opt-out)" if self.has_opt_out else "Ollama (names only)"


def build_variation_specs() -> list[VariationSpec]:
    """Build all 24 variations in run order.

    Order: BART (2), ollama classify (4), scikit-llm (2), then ollama generate
    (4 choice-configs x 4 call budgets = 16). This ordering is stable across the
    writer and the reader because both import :data:`VARIATION_SPECS`.
    """
    specs: list[VariationSpec] = [
        VariationSpec("bart_names_only", "bart", "", "names_only", False),
        VariationSpec("bart_names_optout", "bart", "", "names_only", True),
        VariationSpec("ollama_names_only", "ollama", "classify", "names_only", False),
        VariationSpec("ollama_names_optout", "ollama", "classify", "names_only", True),
        VariationSpec(
            "ollama_names_desc", "ollama", "classify", "names_with_desc", False
        ),
        VariationSpec(
            "ollama_desc_optout", "ollama", "classify", "names_with_desc", True
        ),
        VariationSpec("skllm_names_only", "skllm", "", "names_only", False),
        VariationSpec("skllm_names_optout", "skllm", "", "names_only", True),
    ]

    # Ollama generate: the same four ollama choice-configs as classify, each at
    # the four call budgets in MAX_CALLS.
    _gen_configs = [
        ("ollama_names_only", "names_only", False),
        ("ollama_names_optout", "names_only", True),
        ("ollama_names_desc", "names_with_desc", False),
        ("ollama_desc_optout", "names_with_desc", True),
    ]
    for base_key, flavor, has_opt_out in _gen_configs:
        for mc in MAX_CALLS:
            specs.append(
                VariationSpec(
                    f"{base_key}_gen_mc{mc}",
                    "ollama",
                    "generate",
                    flavor,
                    has_opt_out,
                    mc,
                )
            )

    return specs


VARIATION_SPECS = build_variation_specs()
