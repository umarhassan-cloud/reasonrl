from reasonrl.data.dataset import load_examples
from reasonrl.eval.contamination import check_contamination, contamination_for_examples, split_clean
from reasonrl.utils.text import normalize_text, word_ngrams

TRAIN = {
    "t1": "Tom has 3 apples and buys 5 more apples at the market. How many apples does Tom have now?",
    "t2": "A train travels 60 km per hour for 3 hours. How far does it travel in km?",
}


def test_normalize_and_ngrams():
    assert normalize_text("Hello, World!  3.5") == "hello world 3 5"
    assert normalize_text("x has 3 apples", mask_numbers=True) == "x has <num> apples"
    assert word_ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}
    assert word_ngrams(["a"], 2) == set()


def test_exact_copy_flagged_and_clean_kept():
    ev = {
        "e_copy": TRAIN["t1"],
        "e_clean": "What is the remainder when 100 is divided by 7? Show your reasoning.",
    }
    rep = check_contamination(TRAIN, ev, ngram=8, threshold=0.5)
    assert rep.flagged_ids == ["e_copy"]
    assert rep.clean_ids == ["e_clean"]
    assert rep.rate == 0.5
    item = next(i for i in rep.items if i.eval_id == "e_copy")
    assert item.overlap == 1.0 and item.best_train_id == "t1"


def test_number_masking_catches_templated_leak():
    ev = {"e_tmpl": "A train travels 80 km per hour for 5 hours. How far does it travel in km?"}
    plain = check_contamination(TRAIN, ev, ngram=8, threshold=0.7)
    masked = check_contamination(TRAIN, ev, ngram=8, threshold=0.7, mask_numbers=True)
    assert plain.n_flagged == 0 and masked.n_flagged == 1


def test_short_items_fall_back_to_full_length_ngram():
    rep = check_contamination({"t": "what is 7 plus 3"}, {"e": "What is 7 plus 3?", "f": "what is 8 plus 3"}, ngram=8)
    assert rep.flagged_ids == ["e"]


def test_sample_data_planted_leaks_are_detected(repo_root):
    d = repo_root / "data" / "sample"
    train = load_examples([d / "math_train.jsonl", d / "code_train.jsonl"])
    ev = load_examples([d / "math_eval.jsonl", d / "code_eval.jsonl"])
    rep = contamination_for_examples(train, ev, ngram=8, threshold=0.5, mask_numbers=True)
    planted = {e.id for e in ev if e.meta.get("planted")}
    assert planted and set(rep.flagged_ids) == planted
    clean, dirty = split_clean(ev, rep)
    assert len(clean) + len(dirty) == len(ev) and not ({e.id for e in clean} & planted)
    assert "flagged" in rep.to_markdown()
