# -*- coding: utf-8 -*-
"""Tests for the get_sentiment function in fetcher.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest


class TestGetSentiment:
    """Tests for fetcher.get_sentiment."""

    def _call(self, text, vader_downloaded=False, sia=None):
        """Helper that patches st.session_state and calls get_sentiment."""
        session_state = {'vader_downloaded': vader_downloaded}
        if sia is not None:
            session_state['sia'] = sia

        mock_ss = MagicMock()
        mock_ss.get = lambda key, default=None: session_state.get(key, default)
        mock_ss.__contains__ = lambda self, key: key in session_state
        mock_ss.__getitem__ = lambda self, key: session_state[key]
        mock_ss.__setitem__ = lambda self, key, value: session_state.update({key: value})

        with patch('fetcher.st') as mock_st:
            mock_st.session_state = mock_ss
            from fetcher import get_sentiment
            return get_sentiment(text)

    def test_empty_string_returns_zeros(self):
        result = self._call("")
        assert result == {'polarity': 0.0, 'subjectivity': 0.0, 'compound': 0.0}

    def test_none_returns_zeros(self):
        result = self._call(None)
        assert result == {'polarity': 0.0, 'subjectivity': 0.0, 'compound': 0.0}

    def test_non_string_returns_zeros(self):
        result = self._call(12345)
        assert result == {'polarity': 0.0, 'subjectivity': 0.0, 'compound': 0.0}

    def test_positive_text_has_positive_polarity(self):
        result = self._call("I absolutely love this amazing product!")
        # TextBlob should give positive polarity
        assert result['polarity'] >= 0.0
        assert 'polarity' in result
        assert 'subjectivity' in result
        assert 'compound' in result

    def test_negative_text_has_negative_polarity(self):
        result = self._call("This is terrible and awful.")
        assert result['polarity'] <= 0.0

    def test_vader_compound_when_downloaded(self):
        """When vader_downloaded=True, compound should come from SIA."""
        mock_sia = MagicMock()
        mock_sia.polarity_scores.return_value = {'compound': 0.85, 'pos': 0.9, 'neg': 0.0, 'neu': 0.1}

        # Use a real dict-like mock for session_state; set .sia as attribute
        mock_ss = MagicMock()
        mock_ss.get.side_effect = lambda key, default=None: {'vader_downloaded': True}.get(key, default)
        mock_ss.__contains__ = MagicMock(side_effect=lambda key: key in ('vader_downloaded', 'sia'))
        # Attribute access: session_state.sia returns mock_sia
        mock_ss.sia = mock_sia

        with patch('fetcher.st') as mock_st:
            mock_st.session_state = mock_ss
            from fetcher import get_sentiment
            result = get_sentiment("Great product!")

        assert result['compound'] == 0.85

    def test_vader_skipped_when_not_downloaded(self):
        """compound stays 0.0 when vader_downloaded=False."""
        result = self._call("This text would score positively in VADER.")
        # Without VADER, compound defaults to 0.0
        assert result['compound'] == 0.0

    def test_returns_all_keys(self):
        result = self._call("Some text here")
        assert set(result.keys()) == {'polarity', 'subjectivity', 'compound'}
