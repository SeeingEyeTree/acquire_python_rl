import unittest

import enums
from rl_env import AcquireGymEnv


class TestAcquireGymEnv(unittest.TestCase):
    def test_reset_returns_playable_state(self):
        env = AcquireGymEnv(num_players=2)
        state = env.reset()
        self.assertEqual(state["current_action"]["game_action_id"], enums.GameActions.PlayTile.value)
        self.assertTrue(state["legal_actions"])

    def test_step_returns_gym_style_tuple(self):
        env = AcquireGymEnv(num_players=2)
        env.reset()
        action = env.legal_actions()[0]
        next_state, reward, done, truncated = env.step(action)

        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(reward, int)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)

    def test_illegal_action_raises(self):
        env = AcquireGymEnv(num_players=2)
        env.reset()
        with self.assertRaises(ValueError):
            env.step((enums.GameActions.PlayTile.value, 999))


if __name__ == "__main__":
    unittest.main()
