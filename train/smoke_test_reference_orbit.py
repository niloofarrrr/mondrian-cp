"""Check legacy reference compatibility and outward orbit recovery direction."""
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_sac_lag import _reference_warmup_action

rng = np.random.default_rng(610962)
for state in rng.uniform([-3, -3, -math.pi], [3, 3, math.pi], (1000, 3)):
    target = (-.9, .65) if state[1] < .45 else (1., 1.2)
    angle = math.atan2(target[1] - state[1], target[0] - state[0])
    error = (angle - state[2] + math.pi) % (2 * math.pi) - math.pi
    expected = np.array([1., np.clip(1.5 * error / .8, -1., 1.)], dtype=np.float32)
    np.testing.assert_array_equal(_reference_warmup_action(state, .8, (1., 1.2), (-.9, .65)), expected)

# On the left and bottom of the desired orbit, recovery turns from the
# clockwise tangent toward the outward radial direction, not into the obstacle.
for state in (np.array([-.8, 0., math.pi / 2]), np.array([0., -.8, math.pi])):
    action = _reference_warmup_action(state, .8, (0., 1.8), (-.9, .65), .9, 1., 2.)
    assert action[0] == 1. and 0. < action[1] <= 1.
print('Reference orbit: 1,000 legacy actions unchanged; outward recovery direction checks pass.')
