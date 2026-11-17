from mathref.data.mention_type import classify_mention_type


def test_classifies_english_implicit_mentions() -> None:
    assert classify_mention_type("it", "en") == "implicit"
    assert classify_mention_type("this system_00", "en") == "implicit"
    assert classify_mention_type("these coefficients", "en") == "implicit"
    assert classify_mention_type("here", "en") == "implicit"


def test_classifies_japanese_implicit_mentions() -> None:
    assert classify_mention_type("これ", "jp") == "implicit"
    assert classify_mention_type("この式", "jp") == "implicit"
    assert classify_mention_type("こういう形", "ja") == "implicit"
    assert classify_mention_type("ここからここまで", "jp") == "implicit"


def test_classifies_explicit_mentions() -> None:
    assert classify_mention_type("the first row", "en") == "explicit"
    assert classify_mention_type("x = 0", "en") == "explicit"
    assert classify_mention_type("行列Aの要素", "jp") == "explicit"
    assert classify_mention_type("左辺", "jp") == "explicit"


def test_unknown_is_reserved_for_unusable_or_unsupported_text() -> None:
    assert classify_mention_type("  ", "en") == "unknown"
    assert classify_mention_type("this", "unknown") == "unknown"
