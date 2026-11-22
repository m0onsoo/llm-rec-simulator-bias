"""Formatting helpers for item-side information shown to an avatar."""


ITEM_RATING_MODES = ("included", "excluded")


def history_rating_fragment(rating, mode="included", space_after_colon=True):
    """Return the legacy rating fragment, or nothing for the rating-hidden arm."""
    if mode not in ITEM_RATING_MODES:
        raise ValueError(f"Unsupported item rating mode: {mode}")
    if mode == "excluded":
        return ""
    separator = " " if space_after_colon else ""
    return f" <- History ratings:{separator}{round(rating, 2)} ->"


def format_recommended_item(title, rating, summary, mode="included"):
    """Format a normal recommendation item while preserving the vanilla bytes."""
    return (
        f"<- {title} ->"
        + history_rating_fragment(rating, mode, space_after_colon=True)
        + f" <- Summary: {summary} ->\n"
    )
