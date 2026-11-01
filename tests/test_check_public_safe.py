"""The leak check's matching rules (tools/check_public_safe.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from check_public_safe import denylist_pattern, load_denylist, scan_text  # noqa: E402


def _reasons(line, deny=None, private=frozenset(), longest=0):
    return [reason for _, reason in scan_text([line], deny, set(private), longest)]


def test_denylist_is_whole_word():
    deny = denylist_pattern(["XY"])
    assert _reasons("the XY team", deny) == ["denied word 'XY'"]
    assert _reasons("(XY)", deny) == ["denied word 'XY'"]
    assert _reasons("key and they", deny) == []
    assert _reasons("XYZ or AXY or XY_1", deny) == []


def test_denylist_is_case_sensitive():
    assert _reasons("xy and Xy", denylist_pattern(["XY"])) == []


def test_denylist_entries_with_punctuation():
    deny = denylist_pattern(["two-part", "some_folder"])
    assert _reasons("a two-part task in some_folder/", deny) == ["denied word 'two-part'", "denied word 'some_folder'"]


def test_denylist_file_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / ".denylist"
    path.write_text("# comment\n\nalpha  # trailing comment\nbeta\n")
    assert load_denylist(path) == ["alpha", "beta"]
    assert load_denylist(tmp_path / "missing") == []


# Built at run time, so this file does not itself contain an old-scheme token.
OLD_TOKEN = "Wo" + "rd12"


def test_old_scheme_tokens_are_flagged_and_new_ones_are_not():
    assert _reasons(f"{OLD_TOKEN} next to Ven12 Tok481 Acc33") == [f"old-scheme token '{OLD_TOKEN}'"]
    assert _reasons("Words, AWord3, Word") == []


def test_private_strings_match_across_punctuation():
    assert _reasons('"Ven1 Ven2", 10.0', private={"Ven1 Ven2"}, longest=2) == ["string from the private file (2 tokens)"]
    assert _reasons("Ven1 and Ven2", private={"Ven1 Ven2"}, longest=2) == []
