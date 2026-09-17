def probability_to_risk(probability):
    """
    Common waterlogging/disaster risk bands.
    You can change these thresholds to match your project definition.
    """
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return None

    if p <= 0.30:
        return "LOW"
    if p <= 0.60:
        return "MEDIUM"
    return "HIGH"


def add_risk_label(result):
    if not isinstance(result, dict):
        return result

    p = result.get("positive_probability")
    if p is not None:
        result["risk_level"] = probability_to_risk(p)
    return result
