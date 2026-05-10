import pytest

from user_behavior_engine import (
    AdTargetingSystem,
    PrivacyGuard,
    RecommendationSystem,
    UserBehaviorTrackingSystem,
    UserInteraction,
)


def test_tracking_and_profile_generation_with_privacy_compliance():
    guard = PrivacyGuard(require_consent=True)
    system = UserBehaviorTrackingSystem(privacy_guard=guard)

    guard.grant_consent("user-1")
    system.track_interaction(
        UserInteraction(user_id="user-1", query="best ai models", clicked_topics=["AI", "Finance"], dwell_seconds=12)
    )
    system.track_interaction(
        UserInteraction(user_id="user-1", query="investment strategies", clicked_topics=["Finance"], dwell_seconds=25)
    )

    profile = system.build_user_profile("user-1")
    assert profile.anonymized_user_id != "user-1"
    assert "finance" in profile.interests
    assert len(profile.embedding) == 16


def test_tracking_blocked_without_consent():
    system = UserBehaviorTrackingSystem(privacy_guard=PrivacyGuard(require_consent=True))
    with pytest.raises(PermissionError):
        system.track_interaction(UserInteraction(user_id="user-2", query="hello"))


def test_recommendation_and_ad_targeting_use_profile():
    guard = PrivacyGuard(require_consent=False)
    system = UserBehaviorTrackingSystem(privacy_guard=guard)
    system.track_interaction(UserInteraction(user_id="u", query="ai security", clicked_topics=["AI", "Security"], dwell_seconds=30))
    profile = system.build_user_profile("u")

    recs = RecommendationSystem().recommend(
        profile,
        [
            {"id": "a", "tags": ["AI", "Cloud"]},
            {"id": "b", "tags": ["Sports"]},
            {"id": "c", "tags": ["Security", "AI"]},
        ],
        top_k=2,
    )
    assert len(recs) == 2
    assert recs[0]["id"] in {"a", "c"}

    segments = AdTargetingSystem().target_segments(profile)
    assert "ai_professionals" in segments
