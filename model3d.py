DEFAULT_THICKNESS = 20
DEFAULT_HEIGHT = 300


def generate_3d_model(walls, height=DEFAULT_HEIGHT, thickness=DEFAULT_THICKNESS):
    model = []
    for w in walls:
        wall = {
            "start": [w[0][0], w[0][1], 0],
            "end": [w[1][0], w[1][1], 0],
            "height": height,
            "thickness": thickness
        }
        model.append(wall)
    return model
def create_3d(data):
    walls = data["walls"]
    rooms = data["rooms"]

    objects = []

    # Walls (tall)
    for w in walls:
        objects.append({
            "type": "wall",
            "coords": w,
            "height": 5,
            "thickness": DEFAULT_THICKNESS
        })

    # Rooms (short + different color)
    for r in rooms:
        objects.append({
            "type": "room",
            "coords": r,
            "height": 2
        })

    return objects
