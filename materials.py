
import math

materials_db = [
    {"name": "AAC", "cost": 2, "strength": 5},
    {"name": "Brick", "cost": 5, "strength": 7},
    {"name": "RCC", "cost": 9, "strength": 10}
]

def score(m, wall_type):
    if wall_type == "load":
        wc, ws = 0.3, 0.7
    else:
        wc, ws = 0.6, 0.4

    return ws * m["strength"] - wc * m["cost"]

def classify_wall(w):
    x1, y1 = w[0]
    x2, y2 = w[1]

    length = math.hypot(x2 - x1, y2 - y1)
    return "load" if length > 100 else "partition"

def recommend_materials(walls):
    results = []

    for w in walls:
        wtype = classify_wall(w)

        ranked = sorted(materials_db,
                        key=lambda m: score(m, wtype),
                        reverse=True)

        recommendations = [
            (m["name"], score(m, wtype))
            for m in ranked
        ]

        results.append({
            "wall": w,
            "type": wtype,
            "best": ranked[0]["name"],
            "recommendations": recommendations
        })

    return results


def build_cost_report(materials, pixels_per_meter=100):
    items = []
    totals = {}
    total_cost = 0.0

    for index, item in enumerate(materials):
        wall = item["wall"]
        material_name = item["best"]
        cost_rate = next((m["cost"] for m in materials_db if m["name"] == material_name), 0)
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_m = length_px / pixels_per_meter
        estimated_cost = round(length_m * cost_rate, 2)
        total_cost += estimated_cost
        totals[material_name] = totals.get(material_name, 0.0) + estimated_cost

        items.append({
            "wallIndex": index,
            "wall": wall,
            "wallType": item["type"],
            "material": material_name,
            "lengthMeters": round(length_m, 2),
            "unitCost": float(cost_rate),
            "totalCost": estimated_cost
        })

    summary = {
        "totalCost": round(total_cost, 2),
        "materials": [
            {"name": name, "cost": round(cost, 2)}
            for name, cost in totals.items()
        ]
    }

    return {
        "items": items,
        "summary": summary
    }