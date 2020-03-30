import numpy as np


def predict_proba(points, angles, distances):
    """ Predicts probabilities that a point belongs to a lane (for each lane).


    """
    n_points = points.shape[0]
    n_lines = len(angles)

    cost = np.tile(np.cos(angles), (n_points, 1))
    sint = np.tile(np.sin(angles), (n_points, 1))

    r = np.tile(distances, (n_points, 1))
    x = np.tile(points[:, 0], (n_lines, 1)).T
    y = np.tile(points[:, 1], (n_lines, 1)).T

    # Distance to lines represented with (angle,distance).
    d = r - (x * cost + y * sint)

    # Drop the sign.
    d = np.sqrt(d * d)

    # Calculate sigmoid based on distance from each line.
    e = np.exp(-d)

    return e / np.tile(np.sum(e, axis=1), (n_lines, 1)).T


def predict(points, angles, distances):
    """ Returns the most probable lane for points.

    """
    probs = predict_proba(points, angles, distances)

    return np.argmax(probs, axis=1)


def predict_vehicle_lane(json_data, angles, distances, vehicle_class=1, score_threshold=0.999, scale=(300, 300)):
    """ Returns the most probable lane for each vehicle on the image.

    """
    x = []
    y = []

    classes = json_data['classes']
    scores = json_data['scores']
    for i in range(len(classes)):
        if classes[i] == vehicle_class and scores[i] > score_threshold:
            box = json_data['boxes'][i]
            x.append((box[1] + box[3]) / 2)  # We take the midpoint of the lower edge.
            y.append(box[0])

    x = np.array(x) * scale[0]
    y = np.array(y) * scale[1]

    X = np.vstack([y, scale[0] - x]).T  # Convert to image coordinates.

    return predict(X, angles, distances)

