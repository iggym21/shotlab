import os

import pytest

from analyzer.classifier import train_and_label


def _rep(rep_id, elbow, knee, wrist, arc):
    return {
        "rep_id": rep_id,
        "elbow_at_release": elbow,
        "knee_bend_at_setup": knee,
        "wrist_follow_through": wrist,
        "arc_proxy": arc,
        "label": "unlabeled",
    }


def test_train_and_label_below_min_reps_marks_unclassified(tmp_path):
    reps = [_rep(0, 170, 140, 80, 90), _rep(1, 172, 141, 81, 91)]
    result = train_and_label(reps, output_dir=str(tmp_path), min_reps=3)

    assert [r["label"] for r in result] == ["unclassified", "unclassified"]
    assert not os.path.exists(tmp_path / "classifier.pkl")


def test_train_and_label_separates_clear_outlier(tmp_path):
    reps = [_rep(i, 170 + i * 0.5, 140 + i * 0.5, 80 + i * 0.5, 90 + i * 0.5) for i in range(5)]
    reps.append(_rep(5, 90, 179, 10, 20))  # obvious outlier: collapsed elbow, straight knee, no follow-through

    result = train_and_label(reps, output_dir=str(tmp_path), min_reps=3)
    labels = {r["rep_id"]: r["label"] for r in result}

    assert labels[5] == "inconsistent"
    for rep_id in range(5):
        assert labels[rep_id] == "consistent"


def test_train_and_label_saves_model_when_enough_reps(tmp_path):
    reps = [_rep(i, 170 + i, 140 + i, 80 + i, 90 + i) for i in range(4)]
    train_and_label(reps, output_dir=str(tmp_path), min_reps=3)
    assert (tmp_path / "classifier.pkl").exists()


def test_train_and_label_only_uses_consistent_or_inconsistent_labels(tmp_path):
    reps = [_rep(i, 170 + i, 140 + i, 80 + i, 90 + i) for i in range(6)]
    result = train_and_label(reps, output_dir=str(tmp_path), min_reps=3)
    assert set(r["label"] for r in result) <= {"consistent", "inconsistent"}


def test_train_and_label_mutates_and_returns_same_list(tmp_path):
    reps = [_rep(i, 170 + i, 140 + i, 80 + i, 90 + i) for i in range(4)]
    result = train_and_label(reps, output_dir=str(tmp_path), min_reps=3)
    assert result is reps


def test_train_and_label_handles_identical_reps_without_crashing(tmp_path):
    # Identical feature vectors can make KMeans collapse to a single effective
    # cluster, which previously crashed LogisticRegression.fit on single-class data.
    reps = [_rep(i, 170, 140, 80, 90) for i in range(4)]
    result = train_and_label(reps, output_dir=str(tmp_path), min_reps=3)
    assert [r["label"] for r in result] == ["consistent"] * 4
