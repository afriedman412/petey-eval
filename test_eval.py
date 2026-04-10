"""Tests for evaluate_claude_data scoring functions."""

from evaluate_claude_data import (
    score_enum,
    score_text,
    get_scorer,
    match_pred_key,
    _strip_enum,
)


# ── Enum scoring ─────────────────────────────────────────────────────────────

class TestEnum:
    def test_exact(self):
        assert score_enum("Denied", "Denied")[0] == 1.0

    def test_case_insensitive(self):
        assert score_enum("Denied", "denied")[0] == 1.0

    def test_pydantic_enum(self):
        assert score_enum("Denied", "determination_enum.denied")[0] == 1.0

    def test_pydantic_enum_multiword(self):
        s = "determination_enum.granted_in_part"
        assert score_enum("Granted in Part", s)[0] == 1.0

    def test_petitioner_type_enum(self):
        assert score_enum("Tenant", "petitioner_type_enum.tenant")[0] == 1.0
        assert score_enum("Owner", "petitioner_type_enum.owner")[0] == 1.0

    def test_ra_determination_enum(self):
        s1 = "ra_determination_enum.terminated"
        s2 = "ra_determination_enum.determined"
        assert score_enum("Terminated", s1)[0] == 1.0
        assert score_enum("Determined", s2)[0] == 1.0

    def test_mismatch(self):
        assert score_enum("Denied", "Granted")[0] == 0.0

    def test_both_null(self):
        assert score_enum(None, None)[0] == 1.0

    def test_pred_null(self):
        assert score_enum("Denied", None)[0] == 0.0

    def test_gender_aliases(self):
        assert score_enum("M", "Male")[0] == 1.0
        assert score_enum("F", "Female")[0] == 1.0
        assert score_enum("NB", "Non-binary")[0] == 1.0

    def test_gender_pydantic_enum(self):
        assert score_enum("M", "gender_enum.male")[0] == 1.0
        assert score_enum("F", "gender_enum.female")[0] == 1.0
        assert score_enum("NB", "gender_enum.non-binary")[0] == 1.0

    def test_gender_mismatch(self):
        assert score_enum("M", "F")[0] == 0.0
        assert score_enum("Male", "Female")[0] == 0.0


# ── Text scoring ─────────────────────────────────────────────────────────────

class TestText:
    def test_exact(self):
        assert score_text("hello world", "hello world")[0] == 1.0

    def test_case_insensitive(self):
        assert score_text("Hello", "hello")[0] == 1.0

    def test_token_containment(self):
        assert score_text("back pain", "lower back pain")[0] == 1.0

    def test_both_null(self):
        assert score_text(None, None)[0] == 1.0

    def test_pred_null(self):
        assert score_text("something", None)[0] == 0.0

    def test_gt_null(self):
        assert score_text(None, "something")[0] == 0.0

    def test_edit_distance_fallback(self):
        sc = score_text("ZD410039OM", "2D4100390M")[0]
        assert sc >= 0.7

    def test_date_exact(self):
        assert score_text("2013-04-11", "2013-04-11")[0] == 1.0

    def test_date_wrong(self):
        assert score_text("2013-04-11", "2012-04-11")[0] < 1.0

    def test_age_exact(self):
        assert score_text("34", "34")[0] == 1.0


# ── Scorer dispatch ──────────────────────────────────────────────────────────

class TestGetScorer:
    def test_schema_enum(self):
        assert get_scorer("anything", "enum").__name__ == "score_enum"

    def test_schema_string(self):
        assert get_scorer("anything", "string").__name__ == "score_text"

    def test_schema_date(self):
        # date type falls through to score_text
        assert get_scorer("anything", "date").__name__ == "score_text"

    def test_no_schema(self):
        assert get_scorer("address").__name__ == "score_text"
        assert get_scorer("petitioner").__name__ == "score_text"


# ── Key matching ─────────────────────────────────────────────────────────────

class TestMatchPredKey:
    def test_exact_match(self):
        rec = {"name": "Alice"}
        assert match_pred_key(rec, "name") == "Alice"

    def test_normalized_match(self):
        rec = {"visit_outcome": "discharged"}
        assert match_pred_key(rec, "visit_outcome") == "discharged"

    def test_no_match(self):
        rec = {"foo": "bar"}
        assert match_pred_key(rec, "name") is None


# ── Enum stripping ───────────────────────────────────────────────────────────

class TestStripEnum:
    def test_simple(self):
        assert _strip_enum("Denied") == "denied"

    def test_pydantic_enum(self):
        assert _strip_enum("determination_enum.denied") == "denied"

    def test_multiword(self):
        s = "determination_enum.granted_in_part"
        assert _strip_enum(s) == "granted in part"

    def test_null(self):
        assert _strip_enum(None) == ""

    def test_gender_alias(self):
        assert _strip_enum("M") == "male"
        assert _strip_enum("F") == "female"
        assert _strip_enum("NB") == "non-binary"
