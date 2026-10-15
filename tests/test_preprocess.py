"""CSV quote artifact versus curly quotes that belong to the tweet."""

from preprocess import has_csv_quote_wrap, normalize_for_dedup, strip_csv_quotes


def test_csv_quote_wrap_detector():
    assert has_csv_quote_wrap('"نص"')
    assert not has_csv_quote_wrap("\u201cنص\u201d")
    assert not has_csv_quote_wrap("نص")
    assert not has_csv_quote_wrap('"نص')


def test_strip_and_dedup_keep_a_curly_quote():
    assert strip_csv_quotes('"نص"') == "نص"
    assert strip_csv_quotes("\u201cنص\u201d") == "\u201cنص\u201d"
    assert normalize_for_dedup('"نص"') == "نص"
