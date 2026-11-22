ACTIVITY_PERSONA_TRANSFORMS = ("identity", "swap_extremes")


def transformed_activity_group(activity_group, transform="identity"):
    """Return the activity group shown in the prompt without changing user labels."""
    group = int(activity_group)
    if transform == "identity":
        return group
    if transform == "swap_extremes":
        return {1: 3, 2: 2, 3: 1}[group]
    raise ValueError(f"Unknown activity persona transform: {transform}")
