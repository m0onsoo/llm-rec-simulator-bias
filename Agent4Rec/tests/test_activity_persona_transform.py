from simulation.activity_persona import transformed_activity_group


def test_identity_transform_preserves_all_groups():
    assert [transformed_activity_group(group) for group in (1, 2, 3)] == [1, 2, 3]


def test_swap_extremes_only_exchanges_infrequent_and_frequent():
    assert [
        transformed_activity_group(group, "swap_extremes")
        for group in (1, 2, 3)
    ] == [3, 2, 1]


def test_transform_accepts_numpy_style_integer_values():
    class IntegerLike:
        def __int__(self):
            return 1

    assert transformed_activity_group(IntegerLike(), "swap_extremes") == 3
