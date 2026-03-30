def generate_explanation(materials):
    explanations = []
    for item in materials:
        best = item["recommendations"][0]
        explanations.append(
            f"For wall {item['wall']}, {best[0]} is best due to optimal strength-cost balance ({best[1]:.2f})"
        )
    return explanations