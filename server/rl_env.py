import collections
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import enums
import server as acquire_server


@dataclass
class _DummyClient:
    username: str
    client_id: int
    game_id: int = None
    player_id: int = None


class AcquireGymEnv:
    """Minimal RL-friendly wrapper around the existing Acquire game engine.

    The action format is `(game_action_id, *params)` where `game_action_id`
    matches `enums.GameActions`.
    """

    def __init__(self, num_players: int = 2, tile_bag: Sequence[Tuple[int, int]] = None):
        if not 2 <= num_players <= 6:
            raise ValueError("num_players must be between 2 and 6")
        self.num_players = num_players
        self._tile_bag = list(tile_bag) if tile_bag is not None else None
        self._messages: List[Any] = []
        self._clients: List[_DummyClient] = []
        self.game = None

    def reset(self) -> Dict[str, Any]:
        self._messages = []
        self.game = acquire_server.Game(
            game_id=1,
            internal_game_id=1,
            mode=enums.GameModes.Singles.value,
            max_players=self.num_players,
            add_pending_messages=self._add_pending_messages,
            logging_enabled=False,
            tile_bag=list(self._tile_bag) if self._tile_bag is not None else None,
        )

        self._clients = []
        for index in range(self.num_players):
            client = _DummyClient(username=f"player_{index}", client_id=index + 1)
            self._clients.append(client)
            self.game.join_game(client)

        # StartGame is always required before the playable state.
        if self._current_action().game_action_id == enums.GameActions.StartGame.value:
            self._apply_action((enums.GameActions.StartGame.value,))

        return self.get_state()

    def step(self, action: Sequence[Any]):
        if self.game is None:
            raise RuntimeError("Call reset() before step().")
        if self.game.state == enums.GameStates.Completed.value:
            raise RuntimeError("Game is already completed; call reset() for a new episode.")

        action = self._normalize_action(action)
        legal_actions = self.legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}. Legal actions: {legal_actions}")

        actor_player_id = self._current_action().player_id
        previous_nets = self._net_worths()

        self._apply_action(action)

        next_state = self.get_state()
        new_nets = self._net_worths()
        reward = self._relative_reward(previous_nets, new_nets, actor_player_id)
        done = self.game.state == enums.GameStates.Completed.value
        truncated = False
        return next_state, reward, done, truncated

    def legal_actions(self) -> List[Tuple[Any, ...]]:
        action = self._current_action()
        action_id = action.game_action_id

        if action_id == enums.GameActions.StartGame.value:
            return [(action_id,)]

        if action_id == enums.GameActions.PlayTile.value:
            legal = []
            rack = self.game.tile_racks.racks[action.player_id]
            for tile_index, tile_data in enumerate(rack):
                if (
                    tile_data
                    and tile_data[1] != enums.GameBoardTypes.CantPlayNow.value
                    and tile_data[1] != enums.GameBoardTypes.CantPlayEver.value
                ):
                    legal.append((action_id, tile_index))
            return legal

        if action_id == enums.GameActions.SelectNewChain.value:
            return [(action_id, type_id) for type_id in sorted(action.game_board_type_ids)]

        if action_id == enums.GameActions.SelectMergerSurvivor.value:
            return [(action_id, type_id) for type_id in sorted(action.type_id_sets[0])]

        if action_id == enums.GameActions.SelectChainToDisposeOfNext.value:
            return [(action_id, type_id) for type_id in sorted(action.defunct_type_ids)]

        if action_id == enums.GameActions.DisposeOfShares.value:
            legal = []
            max_trade = min(
                action.defunct_type_count,
                action.controlling_type_available * 2,
            )
            for trade in range(0, max_trade + 1, 2):
                max_sell = action.defunct_type_count - trade
                for sell in range(max_sell + 1):
                    legal.append((action_id, trade, sell))
            return legal

        if action_id == enums.GameActions.PurchaseShares.value:
            return self._purchase_share_legal_actions(action)

        return []

    def get_state(self) -> Dict[str, Any]:
        self.game.score_sheet.update_net_worths()
        current_action = self._current_action()
        return {
            "board": self.game.game_board.x_to_y_to_board_type,
            "chain_size": list(self.game.score_sheet.chain_size),
            "price": list(self.game.score_sheet.price),
            "available": list(self.game.score_sheet.available),
            "player_data": [
                {
                    "shares": list(player_row[:7]),
                    "cash": player_row[enums.ScoreSheetIndexes.Cash.value],
                    "net": player_row[enums.ScoreSheetIndexes.Net.value],
                }
                for player_row in self.game.score_sheet.player_data
            ],
            "turn_player_id": self.game.turn_player_id,
            "current_action": {
                "game_action_id": current_action.game_action_id,
                "player_id": current_action.player_id,
            },
            "active_player_rack": self._serialize_rack(current_action.player_id),
            "legal_actions": self.legal_actions(),
            "done": self.game.state == enums.GameStates.Completed.value,
        }

    def _add_pending_messages(self, messages, client_ids=None):
        self._messages.append((messages, client_ids))

    def _current_action(self):
        return self.game.actions[-1]

    def _apply_action(self, action: Tuple[Any, ...]):
        action_id = action[0]
        params = action[1:]
        active_action = self._current_action()
        actor = self._client_for_player_id(active_action.player_id)
        self.game.do_game_action(actor, action_id, params)


    def _client_for_player_id(self, player_id: int):
        for client in self._clients:
            if client.player_id == player_id:
                return client
        raise RuntimeError(f"No client found for player_id={player_id}")

    def _purchase_share_legal_actions(self, action) -> List[Tuple[Any, ...]]:
        chain_to_max = {}
        cash = self.game.score_sheet.player_data[action.player_id][
            enums.ScoreSheetIndexes.Cash.value
        ]
        for type_id in range(7):
            if (
                self.game.score_sheet.chain_size[type_id]
                and self.game.score_sheet.available[type_id] > 0
                and self.game.score_sheet.price[type_id] > 0
            ):
                chain_to_max[type_id] = self.game.score_sheet.available[type_id]

        purchases = {()}
        chain_ids = sorted(chain_to_max)
        for amount in range(1, 4):
            for combo in itertools.combinations_with_replacement(chain_ids, amount):
                counts = collections.Counter(combo)
                cost = 0
                valid = True
                for type_id, count in counts.items():
                    if count > chain_to_max[type_id]:
                        valid = False
                        break
                    cost += self.game.score_sheet.price[type_id] * count
                if valid and cost <= cash:
                    purchases.add(combo)

        legal = []
        for purchase in sorted(purchases):
            as_list = list(purchase)
            legal.append((enums.GameActions.PurchaseShares.value, as_list, 0))
            if action.can_end_game:
                legal.append((enums.GameActions.PurchaseShares.value, as_list, 1))
        return legal

    def _net_worths(self) -> List[int]:
        self.game.score_sheet.update_net_worths()
        return [
            row[enums.ScoreSheetIndexes.Net.value]
            for row in self.game.score_sheet.player_data
        ]

    def _relative_reward(self, before_nets: List[int], after_nets: List[int], player_id: int) -> int:
        # If leading, compare to second place; otherwise compare to leader.
        ranked = sorted(((net, idx) for idx, net in enumerate(before_nets)), reverse=True)
        leader_id = ranked[0][1]
        if player_id == leader_id and len(ranked) > 1:
            reference_id = ranked[1][1]
        else:
            reference_id = leader_id

        before_gap = before_nets[player_id] - before_nets[reference_id]
        after_gap = after_nets[player_id] - after_nets[reference_id]
        return after_gap - before_gap

    def _serialize_rack(self, player_id: int):
        if self.game.tile_racks is None:
            return []
        rack = self.game.tile_racks.racks[player_id]
        output = []
        for tile_data in rack:
            if tile_data is None:
                output.append(None)
            else:
                tile, board_type, _ = tile_data
                output.append({"tile": tile, "game_board_type": board_type})
        return output

    @staticmethod
    def _normalize_action(action: Sequence[Any]) -> Tuple[Any, ...]:
        if isinstance(action, tuple):
            return action
        if isinstance(action, list):
            return tuple(action)
        if isinstance(action, dict):
            action_id = action["game_action_id"]
            data = action.get("data", [])
            if not isinstance(data, (list, tuple)):
                raise ValueError("dict action `data` must be a list or tuple")
            return (action_id, *data)
        raise ValueError("action must be tuple, list, or dict")


def step_game_state(env: AcquireGymEnv, action: Sequence[Any]):
    """Stateless-feeling helper: pass env and action, get next state tuple."""
    return env.step(action)
