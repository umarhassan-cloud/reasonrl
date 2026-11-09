"""Train/eval contamination detection via lexical n-gram overlap.

For every eval item we compute the fraction of its word n-grams that appear
anywhere in the training prompts (``containment``), and the best single
matching train item. Items above ``threshold`` are flagged and excluded from
the *clean* eval subset that headline numbers are reported on.

Defaults follow common practice: 8-gram matching as in Tülu 3's
decontamination; GPT-3 used 13-grams on longer documents. Short math problems
need smaller n, so items shorter than n fall back to n = len(item).
``mask_numbers`` additionally catches templated near-duplicates whose
numbers were changed (a frequent leak in synthetic math data).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from reasonrl.utils.text import tokens, word_ngrams


@dataclass
class ContaminationItem:
    eval_id: str
    n_tokens: int
    overlap: float  # fraction of eval n-grams found anywhere in train
    best_train_id: str | None
    best_pair_overlap: float  # fraction found in the single best train item
    flagged: bool


@dataclass
class ContaminationReport:
    ngram: int
    threshold: float
    mask_numbers: bool
    n_eval: int
    n_flagged: int
    rate: float
    items: list = field(default_factory=list)

    @property
    def flagged_ids(self) -> list[str]:
        return [i.eval_id for i in self.items if i.flagged]

    @property
    def clean_ids(self) -> list[str]:
        return [i.eval_id for i in self.items if not i.flagged]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flagged_ids"] = self.flagged_ids
        return d

    def to_markdown(self, top: int = 10) -> str:
        lines = [
            f"**Contamination** (word {self.ngram}-grams, threshold {self.threshold}, "
            f"mask_numbers={self.mask_numbers}): {self.n_flagged}/{self.n_eval} flagged "
            f"({100 * self.rate:.1f}%)",
            "",
            "| eval id | overlap | best train match | pair overlap |",
            "|---|---|---|---|",
        ]
        for it in sorted(self.items, key=lambda i: -i.overlap)[:top]:
            mark = " (flagged)" if it.flagged else ""
            lines.append(f"| {it.eval_id}{mark} | {it.overlap:.2f} | {it.best_train_id} | {it.best_pair_overlap:.2f} |")
        return "\n".join(lines)


class NGramIndex:
    """Inverted index ``ngram -> train ids``, built lazily per n."""

    def __init__(self, train_texts: dict[str, str], mask_numbers: bool = False) -> None:
        self.toks = {tid: tokens(t, mask_numbers) for tid, t in train_texts.items()}
        self._index: dict[int, dict[tuple, set[str]]] = {}

    def index(self, n: int) -> dict[tuple, set[str]]:
        if n not in self._index:
            idx: dict[tuple, set[str]] = defaultdict(set)
            for tid, tk in self.toks.items():
                for g in word_ngrams(tk, n):
                    idx[g].add(tid)
            self._index[n] = idx
        return self._index[n]

    def query(self, toks: list[str], n: int) -> tuple[float, str | None, float]:
        n_eff = min(n, len(toks))
        if n_eff == 0:
            return 0.0, None, 0.0
        grams = word_ngrams(toks, n_eff)
        idx = self.index(n_eff)
        hits = 0
        per_train: dict[str, int] = defaultdict(int)
        for g in grams:
            ids = idx.get(g)
            if ids:
                hits += 1
                for tid in ids:
                    per_train[tid] += 1
        if not per_train:
            return 0.0, None, 0.0
        best = max(per_train, key=lambda t: per_train[t])
        return hits / len(grams), best, per_train[best] / len(grams)


def check_contamination(
    train_texts: dict[str, str],
    eval_texts: dict[str, str],
    ngram: int = 8,
    threshold: float = 0.5,
    mask_numbers: bool = False,
) -> ContaminationReport:
    index = NGramIndex(train_texts, mask_numbers)
    items = []
    for eid, text in eval_texts.items():
        tk = tokens(text, mask_numbers)
        overlap, best, pair = index.query(tk, ngram)
        items.append(ContaminationItem(eid, len(tk), overlap, best, pair, overlap >= threshold))
    n_flag = sum(i.flagged for i in items)
    return ContaminationReport(
        ngram=ngram,
        threshold=threshold,
        mask_numbers=mask_numbers,
        n_eval=len(items),
        n_flagged=n_flag,
        rate=n_flag / max(1, len(items)),
        items=items,
    )


def contamination_for_examples(train_examples, eval_examples, ngram=8, threshold=0.5, mask_numbers=False):
    """Convenience wrapper over :class:`reasonrl.data.Example` lists (question text)."""
    return check_contamination(
        {e.id: e.question for e in train_examples},
        {e.id: e.question for e in eval_examples},
        ngram,
        threshold,
        mask_numbers,
    )


def split_clean(eval_examples, report: ContaminationReport):
    flagged = set(report.flagged_ids)
    clean = [e for e in eval_examples if e.id not in flagged]
    dirty = [e for e in eval_examples if e.id in flagged]
    return clean, dirty
