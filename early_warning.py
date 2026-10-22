
def get_warning(predicted_capacity, reference_capacity):
    """Classify predicted capacity degradation for a prototype demo."""

    if reference_capacity <= 0:
        return "Invalid reference capacity"

    decline = 1 - (predicted_capacity / reference_capacity)

    if decline >= 0.30:
        return "Critical Degradation"
    elif decline >= 0.20:
        return "High Warning"
    elif decline >= 0.10:
        return "Early Warning"
    else:
        return "Normal"
