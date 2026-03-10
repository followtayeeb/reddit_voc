# -*- coding: utf-8 -*-
"""Tests for the public (requests-based) fetch functions in fetcher.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests


def _make_posts_json(count=2):
    """Helper: produce valid Reddit posts JSON structure."""
    children = []
    for i in range(count):
        children.append({
            'kind': 't3',
            'data': {
                'id': f'post{i}',
                'title': f'Post title {i}',
                'selftext': f'Body text {i}',
                'score': 100 + i,
                'num_comments': 10 + i,
                'created_utc': 1700000000.0 + i * 3600,
                'author': f'author{i}',
                'permalink': f'/r/test/comments/post{i}/title_{i}/',
                'stickied': False,
            },
        })
    return {'data': {'children': children, 'after': None}, 'kind': 'Listing'}


def _make_comments_json(post_id='abc', count=2):
    """Helper: produce valid Reddit comments JSON structure."""
    children = []
    for i in range(count):
        children.append({
            'kind': 't1',
            'data': {
                'id': f'comm{i}',
                'body': f'Comment text {i}',
                'score': 20 + i,
                'created_utc': 1700010000.0 + i * 60,
                'author': f'commenter{i}',
                'is_submitter': False,
            },
        })
    # Reddit JSON for comments is a 2-element list
    return [
        {'kind': 'Listing', 'data': {'children': []}},  # post listing
        {'kind': 'Listing', 'data': {'children': children, 'after': None}},  # comments
    ]


class TestFetchPostsRequests:
    """Tests for fetcher.fetch_posts_requests (bypassing st.cache_data)."""

    def _call(self, subreddit='test', limit=10, sort='hot', mock_response=None):
        """Patch requests.get and st.cache_data, then call the inner function."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response or _make_posts_json()
        mock_resp.raise_for_status = MagicMock()

        with patch('fetcher.requests.get', return_value=mock_resp), \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)
            mock_st.cache_data = MagicMock(return_value=lambda fn: fn)

            # Import after patching to get the un-decorated function
            import importlib
            import fetcher as fetcher_mod
            # Call the underlying logic directly via the module-level function
            # We need to bypass the cache decorator
            result = fetcher_mod.fetch_posts_requests.__wrapped__(subreddit, limit, sort) \
                if hasattr(fetcher_mod.fetch_posts_requests, '__wrapped__') \
                else fetcher_mod.fetch_posts_requests(subreddit, limit, sort)
        return result

    def test_returns_dataframe_on_success(self):
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_posts_json(count=2)
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            # Call bypassing cache: use __wrapped__ if available, else call directly
            fn = getattr(fetcher_mod.fetch_posts_requests, '__wrapped__', fetcher_mod.fetch_posts_requests)
            result = fn('test', 10, 'hot')

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'Post ID' in result.columns
        assert 'Title' in result.columns
        assert len(result) == 2

    def test_returns_none_on_404_json(self):
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = {'error': 404}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_posts_requests, '__wrapped__', fetcher_mod.fetch_posts_requests)
            result = fn('doesnotexist', 10, 'hot')

        assert result is None

    def test_returns_none_on_timeout(self):
        with patch('fetcher.requests.get', side_effect=requests.exceptions.Timeout), \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_posts_requests, '__wrapped__', fetcher_mod.fetch_posts_requests)
            result = fn('test', 10, 'hot')

        assert result is None

    def test_returns_none_on_json_decode_error(self):
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_posts_requests, '__wrapped__', fetcher_mod.fetch_posts_requests)
            result = fn('test', 10, 'hot')

        assert result is None

    def test_expected_columns_present(self):
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_posts_json(count=1)
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_posts_requests, '__wrapped__', fetcher_mod.fetch_posts_requests)
            result = fn('test', 5, 'hot')

        expected_cols = {'Post ID', 'Title', 'Content', 'Score', 'Comments Count',
                         'Created Date', 'Author', 'URL',
                         'Sentiment Polarity', 'Sentiment Subjectivity', 'Sentiment Compound'}
        assert expected_cols.issubset(set(result.columns))


class TestFetchCommentsRequests:
    """Tests for fetcher.fetch_comments_requests (bypassing st.cache_data)."""

    def test_returns_dataframe_on_success(self):
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_comments_json(count=3)
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_comments_requests, '__wrapped__', fetcher_mod.fetch_comments_requests)
            result = fn('test', 'abc123', 10)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'Comment Body' in result.columns
        assert len(result) == 3

    def test_returns_none_on_bad_json_structure(self):
        """JSON that doesn't match the expected 2-element list structure."""
        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()

            mock_resp = MagicMock()
            mock_resp.json.return_value = {'error': 'not a list'}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_comments_requests, '__wrapped__', fetcher_mod.fetch_comments_requests)
            result = fn('test', 'abc123', 10)

        assert result is None

    def test_returns_none_on_timeout(self):
        with patch('fetcher.requests.get', side_effect=requests.exceptions.Timeout), \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_comments_requests, '__wrapped__', fetcher_mod.fetch_comments_requests)
            result = fn('test', 'abc123', 10)

        assert result is None

    def test_filters_deleted_comments(self):
        """Comments with [deleted] body should be excluded."""
        deleted_comment = {
            'kind': 't1',
            'data': {
                'id': 'del1',
                'body': '[deleted]',
                'score': 5,
                'created_utc': 1700010000.0,
                'author': 'deleteduser',
                'is_submitter': False,
            },
        }
        good_comment = {
            'kind': 't1',
            'data': {
                'id': 'good1',
                'body': 'A real comment',
                'score': 10,
                'created_utc': 1700010060.0,
                'author': 'realuser',
                'is_submitter': False,
            },
        }
        comments_json = [
            {'kind': 'Listing', 'data': {'children': []}},
            {'kind': 'Listing', 'data': {'children': [deleted_comment, good_comment]}},
        ]

        with patch('fetcher.requests.get') as mock_get, \
             patch('fetcher.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=False)
            mock_st.session_state.__contains__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.json.return_value = comments_json
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            import fetcher as fetcher_mod
            fn = getattr(fetcher_mod.fetch_comments_requests, '__wrapped__', fetcher_mod.fetch_comments_requests)
            result = fn('test', 'abc123', 10)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['Comment Body'] == 'A real comment'
