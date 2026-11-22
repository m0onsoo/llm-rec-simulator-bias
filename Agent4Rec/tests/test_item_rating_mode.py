from simulation.item_presentation import format_recommended_item, history_rating_fragment


def test_vanilla_item_format_is_byte_identical():
    expected = (
        "<- Toy Story (1995) -> <- History ratings: 4.15 -> "
        "<- Summary: Toys go on an adventure. ->\n"
    )
    assert format_recommended_item(
        "Toy Story (1995)", 4.146846, "Toys go on an adventure."
    ) == expected


def test_rating_hidden_format_removes_only_rating_fragment():
    assert format_recommended_item(
        "Toy Story (1995)", 4.146846, "Toys go on an adventure.", "excluded"
    ) == "<- Toy Story (1995) -> <- Summary: Toys go on an adventure. ->\n"


def test_legacy_compact_rating_fragment_is_preserved():
    assert history_rating_fragment(
        3.201141, "included", space_after_colon=False
    ) == " <- History ratings:3.2 ->"
    assert history_rating_fragment(
        3.201141, "excluded", space_after_colon=False
    ) == ""
