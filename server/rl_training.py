import random
from collections import deque
from itertools import combinations_with_replacement
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import enums
from rl_env import AcquireGymEnv


MAX_PLAYERS = 6
RACK_SIZE = 6
BOARD_WIDTH = 12
BOARD_HEIGHT = 9
MAX_DISPOSE_SHARES = 25


Action = Tuple[Any, ...]


def build_action_space() -> List[Action]:
    """Builds a fixed action space that covers all legal action tuple shapes."""
    action_space: List[Action] = []

    action_space.append((enums.GameActions.StartGame.value,))

    # PlayTile: tile index in rack.
    for tile_index in range(RACK_SIZE):
        action_space.append((enums.GameActions.PlayTile.value, tile_index))

    # Select chain ids.
    for type_id in range(7):
        action_space.append((enums.GameActions.SelectNewChain.value, type_id))
        action_space.append((enums.GameActions.SelectMergerSurvivor.value, type_id))
        action_space.append((enums.GameActions.SelectChainToDisposeOfNext.value, type_id))

    # DisposeOfShares: (trade_even, sell)
    for trade in range(0, MAX_DISPOSE_SHARES + 1, 2):
        for sell in range(MAX_DISPOSE_SHARES - trade + 1):
            action_space.append((enums.GameActions.DisposeOfShares.value, trade, sell))

    # PurchaseShares: list of up to 3 shares and whether to end game.
    purchase_patterns = [()]
    for amount in range(1, 4):
        purchase_patterns.extend(combinations_with_replacement(range(7), amount))

    for purchase in purchase_patterns:
        purchase_list = list(purchase)
        action_space.append((enums.GameActions.PurchaseShares.value, purchase_list, 0))
        action_space.append((enums.GameActions.PurchaseShares.value, purchase_list, 1))

    # De-duplicate while preserving order.
    deduped: List[Action] = []
    seen = set()
    for action in action_space:
        key = action_to_key(action)
        if key not in seen:
            deduped.append(action)
            seen.add(key)
    return deduped


def action_to_key(action: Sequence[Any]) -> Tuple[Any, ...]:
    """Hashable canonical key for action tuples containing nested lists."""
    key = []
    for item in action:
        if isinstance(item, list):
            key.append(tuple(item))
        else:
            key.append(item)
    return tuple(key)


def state_to_numeric_vector(state: Dict[str, Any], max_players: int = MAX_PLAYERS) -> np.ndarray:
    """Converts the dict game state into a fixed-size numeric feature vector."""
    features: List[float] = []

    board = state["board"]
    for x in range(BOARD_WIDTH):
        col = board[x]
        for y in range(BOARD_HEIGHT):
            features.append(float(col[y]))

    features.extend(float(value) for value in state["chain_size"])
    features.extend(float(value) for value in state["price"])
    features.extend(float(value) for value in state["available"])

    players = state["player_data"]
    for idx in range(max_players):
        if idx < len(players):
            pdata = players[idx]
            features.extend(float(v) for v in pdata["shares"])
            features.append(float(pdata["cash"]))
            features.append(float(pdata["net"]))
        else:
            features.extend([0.0] * 9)

    turn_player_id = state["turn_player_id"]
    current_action = state["current_action"]
    features.append(float(turn_player_id if turn_player_id is not None else -1))
    features.append(float(current_action["game_action_id"]))
    features.append(float(current_action["player_id"]))

    rack = state["active_player_rack"]
    for idx in range(RACK_SIZE):
        if idx < len(rack) and rack[idx] is not None:
            tile = rack[idx]["tile"]
            features.extend([float(tile[0]), float(tile[1]), float(rack[idx]["game_board_type"])])
        else:
            features.extend([-1.0, -1.0, -1.0])

    features.append(1.0 if state.get("done") else 0.0)

    return np.asarray(features, dtype=np.float32)


def legal_action_mask(
    legal_actions: Iterable[Action],
    action_to_index: Dict[Tuple[Any, ...], int],
    action_space_size: int,
) -> np.ndarray:
    mask = np.zeros(action_space_size, dtype=np.float32)
    for action in legal_actions:
        key = action_to_key(action)
        if key in action_to_index:
            mask[action_to_index[key]] = 1.0
    return mask


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state_vec, action_idx, reward, next_state_vec, done, next_legal_mask):
        self.buffer.append((state_vec, action_idx, reward, next_state_vec, done, next_legal_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = map(np.asarray, zip(*batch))
        return states, actions, rewards.astype(np.float32), next_states, dones.astype(np.float32), next_masks

    def __len__(self):
        return len(self.buffer)


def build_q_network(input_dim: int, output_dim: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def epsilon_greedy_action(
    model: tf.keras.Model,
    state_vec: np.ndarray,
    legal_actions: Sequence[Action],
    action_to_index: Dict[Tuple[Any, ...], int],
    epsilon: float,
) -> Action:
    if random.random() < epsilon:
        return random.choice(list(legal_actions))

    q_values = model.predict(state_vec[np.newaxis, :], verbose=0)[0]
    legal_indices = [action_to_index[action_to_key(action)] for action in legal_actions]
    best_idx = max(legal_indices, key=lambda idx: q_values[idx])
    return ACTION_SPACE[best_idx]


def train_dqn(
    num_episodes: int = 500,
    max_steps_per_episode: int = 500,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    replay_capacity: int = 10000,
    train_start_size: int = 1000,
    target_update_every: int = 10,
    save_path: str = "saved_models/acquire_dqn.keras",
):
    env = AcquireGymEnv(num_players=2)

    initial_state = env.reset()
    input_dim = state_to_numeric_vector(initial_state).shape[0]
    output_dim = len(ACTION_SPACE)

    q_model = build_q_network(input_dim, output_dim)
    target_model = build_q_network(input_dim, output_dim)
    target_model.set_weights(q_model.get_weights())

    replay_buffer = ReplayBuffer(replay_capacity)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state_vec = state_to_numeric_vector(state)
        total_reward = 0.0

        for _step in range(max_steps_per_episode):
            legal_actions = state["legal_actions"]
            if not legal_actions:
                break

            action = epsilon_greedy_action(q_model, state_vec, legal_actions, ACTION_TO_INDEX, epsilon)
            next_state, reward, done, _truncated = env.step(action)

            next_state_vec = state_to_numeric_vector(next_state)
            next_legal_mask = legal_action_mask(next_state["legal_actions"], ACTION_TO_INDEX, output_dim)

            replay_buffer.add(
                state_vec,
                ACTION_TO_INDEX[action_to_key(action)],
                reward,
                next_state_vec,
                done,
                next_legal_mask,
            )

            state = next_state
            state_vec = next_state_vec
            total_reward += reward

            if len(replay_buffer) >= max(train_start_size, batch_size):
                states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)

                next_q = target_model.predict(next_states, verbose=0)
                masked_next_q = np.where(next_masks > 0.0, next_q, -1e9)
                next_best = np.max(masked_next_q, axis=1)
                targets = rewards + (1.0 - dones) * gamma * next_best

                current_q = q_model.predict(states, verbose=0)
                row_idx = np.arange(batch_size)
                current_q[row_idx, actions.astype(np.int32)] = targets

                # Keep fit verbosity at 1 to surface progress while training.
                q_model.fit(states, current_q, epochs=1, verbose=1)

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % target_update_every == 0:
            target_model.set_weights(q_model.get_weights())

        print(f"episode={episode + 1}/{num_episodes} reward={total_reward:.2f} epsilon={epsilon:.3f}")

    q_model.save(save_path)
    print(f"Saved trained model to: {save_path}")
    return q_model


ACTION_SPACE = build_action_space()
ACTION_TO_INDEX = {action_to_key(action): idx for idx, action in enumerate(ACTION_SPACE)}


if __name__ == "__main__":
    save_path = "saved_models/acquire_dqn.keras"
    train_dqn(save_path=save_path)
