def build_geometry(walls):
    points = set()
    for w in walls:
        points.add(tuple(w[0]))
        points.add(tuple(w[1]))

    return list(points), walls