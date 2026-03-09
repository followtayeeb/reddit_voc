# -*- coding: utf-8 -*-
"""Tests for concurrent comment fetching in fetcher.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fetcher import fetch_all_comments_concurrent


def _make_comments_df(post_id: str, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'Comment ID': f'{post_id}_c{i}',
            'Post ID': post_id,
            'Comment Body': f'Comment {i} for {post_id}',
            'Score': i * 10,
            'Created Date': pd.Timestamp('2024-01-15'),
            'Author': f'user{i}',
            'Is Submitter': False,
            'Sentiment Polarity': 0.0,
            'Sentiment Subjectivity': 0.0,
            'Sentiment Compound': 0.0,
        }
        for i in range(n)
    ])


class TestFetchAllCommentsConcurrent:
    """Tests for fetch_all_comments_concurrent."""

    def test_returns_dict_keyed_by_post_id(self):
        post_ids = ['abc', 'def']

        def fake_fetch(subreddit, post_id, sort_by, limit, praw_details):
            return _make_comments_df(post_id)

        with patch('fetcher.fetch_comments_for_post', side_effect=fake_fetch):
            result = fetch_all_comments_concurrent(
                post_ids=post_ids,
                subreddit_name='test',
                sort_by='top',
                limit=50,
                praw_details=None,
            )

        assert set(result.keys()) == {'abc', 'def'}
        assert isinstance(result['abc'], pd.DataFrame)
        assert isinstance(result['def'], pd.DataFrame)

    def test_failed_post_returns_none_not_raises(self):
        """A failing fetch should store None, not propagate the exception."""
        def fail_fetch(subreddit, post_id, sort_by, limit, praw_details):
            raise RuntimeError("Simulated network error")

        with patch('fetcher.fetch_comments_for_post', side_effect=fail_fetch):
            result = fetch_all_comments_concurrent(
                post_ids=['xyz'],
                subreddit_name='test',
                sort_by='top',
                limit=50,
                praw_details=None,
            )

        assert 'xyz' in result
        assert result['xyz'] is None

    def test_mixed_success_and_failure(self):
        def mixed_fetch(subreddit, post_id, sort_by, limit, praw_details):
            if post_id == 'good':
                return _make_comments_df(post_id)
            raise ValueError("Bad post")

        with patch('fetcher.fetch_comments_for_post', side_effect=mixed_fetch):
            result = fetch_all_comments_concurrent(
                post_ids=['good', 'bad'],
                subreddit_name='test',
                sort_by='top',
                limit=50,
                praw_details=None,
            )

        assert result['good'] is not None
        assert result['bad'] is None

    def test_empty_post_ids_returns_empty_dict(self):
        result = fetch_all_comments_concurrent(
            post_ids=[],
            subreddit_name='test',
            sort_by='top',
            limit=50,
            praw_details=None,
        )
        assert result == {}

    def test_all_posts_processed(self):
        post_ids = [f'post{i}' for i in range(10)]

        def fake_fetch(subreddit, post_id, sort_by, limit, praw_details):
            return _make_comments_df(post_id, n=2)

        with patch('fetcher.fetch_comments_for_post', side_effect=fake_fetch):
            result = fetch_all_comments_concurrent(
                post_ids=post_ids,
                subreddit_name='test',
                sort_by='top',
                limit=50,
                praw_details=None,
                max_workers=4,
            )

        assert len(result) == 10
        for pid in post_ids:
            assert pid in result
            assert result[pid] is not None
