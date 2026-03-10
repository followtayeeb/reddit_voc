# -*- coding: utf-8 -*-
"""Tests for voc/utils.py helper functions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import pandas as pd
import pytest

from utils import (
    extract_subreddit_name,
    format_datetime,
    sanitize_markdown,
    validate_subreddit,
)


# ---------------------------------------------------------------------------
# extract_subreddit_name
# ---------------------------------------------------------------------------

class TestExtractSubredditName:
    def test_full_url(self):
        assert extract_subreddit_name("https://www.reddit.com/r/python/") == "python"

    def test_url_without_trailing_slash(self):
        assert extract_subreddit_name("https://reddit.com/r/MachineLearning") == "MachineLearning"

    def test_url_with_post_path(self):
        assert extract_subreddit_name("https://www.reddit.com/r/learnpython/comments/abc123/title/") == "learnpython"

    def test_plain_name(self):
        assert extract_subreddit_name("python") == "python"

    def test_plain_name_with_underscores(self):
        assert extract_subreddit_name("ask_reddit") == "ask_reddit"

    def test_empty_string(self):
        assert extract_subreddit_name("") is None

    def test_none_returns_none(self):
        assert extract_subreddit_name(None) is None

    def test_url_with_domain_dots_not_treated_as_name(self):
        # Has dots — looks like a domain, should not extract as name
        result = extract_subreddit_name("some.domain.com")
        assert result is None

    def test_case_insensitive_url_match(self):
        result = extract_subreddit_name("HTTPS://WWW.REDDIT.COM/R/Python/")
        assert result == "Python"


# ---------------------------------------------------------------------------
# validate_subreddit
# ---------------------------------------------------------------------------

class TestValidateSubreddit:
    def test_valid_name(self):
        assert validate_subreddit("python") == "python"

    def test_valid_name_with_numbers(self):
        assert validate_subreddit("test123") == "test123"

    def test_valid_name_with_underscores(self):
        assert validate_subreddit("ask_reddit") == "ask_reddit"

    def test_strips_r_prefix(self):
        assert validate_subreddit("r/python") == "python"

    def test_strips_leading_slash(self):
        assert validate_subreddit("/python") == "python"

    def test_strips_whitespace(self):
        assert validate_subreddit("  python  ") == "python"

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            validate_subreddit("ab")

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            validate_subreddit("a" * 22)

    def test_spaces_in_name_raises(self):
        with pytest.raises(ValueError):
            validate_subreddit("has spaces")

    def test_special_chars_raises(self):
        with pytest.raises(ValueError):
            validate_subreddit("name-with-dashes")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_subreddit("")

    def test_21_chars_valid(self):
        name = "a" * 21
        assert validate_subreddit(name) == name

    def test_3_chars_valid(self):
        assert validate_subreddit("abc") == "abc"


# ---------------------------------------------------------------------------
# sanitize_markdown
# ---------------------------------------------------------------------------

class TestSanitizeMarkdown:
    def test_clean_text_unchanged(self):
        text = "This is plain **markdown** text."
        assert sanitize_markdown(text) == text

    def test_escapes_html_tags(self):
        text = "<script>alert('xss')</script>"
        result = sanitize_markdown(text)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_preserves_markdown_bold(self):
        text = "**bold** and _italic_"
        assert sanitize_markdown(text) == text

    def test_empty_string(self):
        assert sanitize_markdown("") == ""

    def test_none_like_falsy(self):
        # sanitize_markdown checks `if not text: return text`
        assert sanitize_markdown("") == ""

    def test_angle_brackets_both_escaped(self):
        result = sanitize_markdown("<div>Hello</div>")
        assert "&lt;div&gt;" in result
        assert "&lt;/div&gt;" in result


# ---------------------------------------------------------------------------
# format_datetime
# ---------------------------------------------------------------------------

class TestFormatDatetime:
    def test_datetime_object(self):
        dt = datetime(2024, 3, 15, 10, 30, 0)
        assert format_datetime(dt) == "2024-03-15 10:30:00"

    def test_timestamp_string(self):
        result = format_datetime("2024-03-15 10:30:00")
        assert result == "2024-03-15 10:30:00"

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-06-01 12:00:00")
        assert format_datetime(ts) == "2024-06-01 12:00:00"

    def test_nat_returns_empty(self):
        result = format_datetime(pd.NaT)
        assert result == ""

    def test_none_returns_empty(self):
        result = format_datetime(None)
        assert result == ""
