from gemma_grpo.groups import add_group_advantages, trainable_groups


def test_advantages_are_normalized_within_group():
    rows = add_group_advantages(
        [
            {"group_id": "a", "reward": 1.0, "scorable": True},
            {"group_id": "a", "reward": 0.0, "scorable": True},
            {"group_id": "b", "reward": 0.0, "scorable": True},
            {"group_id": "b", "reward": 0.0, "scorable": True},
        ]
    )
    group_a = [row for row in rows if row["group_id"] == "a"]
    assert sorted(row["advantage"] for row in group_a) == [-1.0, 1.0]
    assert all(row["trainable"] for row in group_a)
    assert "b" not in trainable_groups(rows)


def test_unscorable_row_has_no_advantage():
    row = add_group_advantages([{"group_id": "a", "reward": None, "scorable": False}])[0]
    assert row["advantage"] is None
    assert not row["trainable"]

