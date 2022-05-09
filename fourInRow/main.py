import copy
import pickle
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Union, Tuple

import numpy as np


class State(object):
    ROWS = 6
    COLS = 7
    WIN = 4
    PLAYER_1 = 1
    PLAYER_2 = -1
    count = np.reshape([3 ** i for i in range(ROWS * COLS)], (ROWS, COLS))

    def __init__(self, board: np.ndarray = None):
        if board is None:
            self.board: np.ndarray = np.array([[0] * self.COLS for _ in range(self.ROWS)])
        else:
            self.board = board

        self.win = [None, None]
        self.hash = {1: int(((np.abs(self.board) * (1.5 - self.board / 2)) * self.count).sum()),
                     -1: int(((np.abs(self.board) * (1.5 + self.board / 2)) * self.count).sum())}

    def __hash__(self):
        return self.hash[1]
        # if self.hash[1] > 0:
        #     return self.hash[1]
        # h = 0
        # count = 1
        # for row in self.board:
        #     all_zero = True
        #     for c in row:
        #         h += abs(c) * (1.5 - c / 2) * count
        #         count *= 3
        #         all_zero &= c == 0
        #     if all_zero:
        #         break
        # self.hash[1] = int(h)
        # return int(h)

    def __str__(self):
        s = ""
        for row in reversed(self.board):
            s += f"{row}\n"
        return s

    @staticmethod
    def hash_to_board(h: int):
        if h < 0:
            return np.empty(0)
        ass_vel = h
        ret = np.zeros((State.ROWS, State.COLS), dtype=int)
        for i in reversed(range(State.ROWS)):
            for j in reversed(range(State.COLS)):
                ret[i][j] = int(h // (3**(i*j)))
                h -= ret[i][j] * (3**(i*j))
        assert ass_vel == int(((np.abs(ret) * (1.5 - ret / 2)) * State.count).sum())
        return ret

    def player_hash(self, player: int):
        return self.hash[player]

    def flip(self):
        return State(-self.board)

    def next(self, col: int, player: int) -> Union[None, 'State']:
        if col is None:
            return None
        if player != self.PLAYER_1 and player != self.PLAYER_2:
            return None
        if self.player_win(player):
            return None

        ind = 0
        while ind < self.ROWS and self.board[ind][col] != 0:
            ind += 1
        if ind == self.ROWS:
            return None
        board = self.board.copy()
        board[ind][col] = player
        return State(board)

    def player_win(self, player: int) -> bool:
        if player != self.PLAYER_1 and player != self.PLAYER_2:
            return False
        player_index = int((1 - player) // 2)
        if self.win[player_index] is not None:
            # noinspection PyTypeChecker
            return self.win[player_index]

        # check rows
        for row in self.board:
            count = 0
            for c in row:
                if c == player:
                    count += 1
                    if count == self.WIN:
                        self.win[player_index] = True
                        return True
                else:
                    count = 0
        # check cols
        for j in range(self.COLS):
            count = 0
            for i in range(self.ROWS):
                c = self.board[i][j]
                if c == 0:
                    break
                if c == player:
                    count += 1
                    if count == self.WIN:
                        self.win[player_index] = True
                        return True
                else:
                    count = 0

        # diagonal
        for s in range(max(self.COLS, self.ROWS)):
            count1, count2, count3, count4 = 0, 0, 0, 0
            for m in range(s + 1):
                i, j = m, s - m
                count1, is_win = self.is_win(i, j, count1, player)
                if is_win:
                    self.win[player_index] = True
                    return True
                i, j = (self.ROWS - 1) - (s - m), (self.COLS - 1) - m
                count2, is_win = self.is_win(i, j, count2, player)
                if is_win:
                    self.win[player_index] = True
                    return True
                i, j = m, (self.COLS - 1) - (s - m)
                count3, is_win = self.is_win(i, j, count3, player)
                if is_win:
                    self.win[player_index] = True
                    return True
                i, j = (self.ROWS - 1) - (s - m), m
                count4, is_win = self.is_win(i, j, count4, player)
                if is_win:
                    self.win[player_index] = True
                    return True
        self.win[player_index] = False
        return False

    def is_win(self, i, j, count, player):
        if not (0 <= i < self.ROWS and 0 <= j < self.COLS):
            return count, False
        c = self.board[i][j]
        if c == player:
            count += 1
            if count == self.WIN:
                return 0, True
        else:
            count = 0
        return count, False


class Player(object):
    def __init__(self, player, gamma=1.0, policy=0):
        self.q = {}
        self.player = player
        self.gamma = gamma
        if policy == 0:
            self.policy = self.policy_random
        elif policy == 1:
            self.policy = self.policy_max
        elif policy == 2:
            self.policy = self.policy_probability
        elif policy == 3:
            self.policy = self.policy_user
        self.last_state = None
        self.last_action = None
        self.last_r = None
        self.step = {}

    def save(self, base_dir: Path):
        with open(base_dir / f"player_{self.player}.pickle", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def init_q(self, state_hash, a: int):
        if (state_hash, a) not in self.q:
            self.q[(state_hash, a)] = np.random.rand()

    def update_q(self, state, a, r):
        if self.last_state == self.last_state and self.last_action == self.last_action and self.last_r == self.last_r:
            last_state_hash = self.last_state.player_hash(self.player)
            if state is None:
                next_reward = 0
            else:
                state_hash = state.player_hash(self.player)
                if (state_hash, a) not in self.q:
                    self.init_q(state_hash, a)
                next_reward = self.gamma * self.q[(state_hash, a)]
            updated_index = (last_state_hash, self.last_action)
            self.step[updated_index] = self.step.get(updated_index, 0) + 1
            learning_rate = 1 / np.sqrt(self.step[updated_index])
            self.q[updated_index] = (self.q.get(updated_index, 0)*(1-learning_rate)
                                     + learning_rate * (self.last_r + next_reward))
        self.last_state = state
        self.last_action = a
        self.last_r = r

    def get_next_state_reward(self, state: State):
        r = []
        state_hash = state.player_hash(self.player)
        for c in range(state.COLS):
            next_state = state.next(c, self.player)
            if next_state is None:
                self.q[state_hash, c] = -np.inf
                continue
            if (state_hash, c) not in self.q:
                self.init_q(state_hash, c)
            r.append([c, self.q[(state_hash, c)]])
        return r

    def policy_probability(self, state: State):
        r_list = self.get_next_state_reward(state)
        if not r_list:
            return None
        v_a = np.array([v[1] for v in r_list])
        probability = np.exp(v_a.copy())

        if np.isinf(probability).any():
            print(f"INF:{v_a}:\n{state}")
            probability[np.isinf(probability)] = 1

        p_sum = probability.sum()
        if p_sum == 0:
            probability = np.random.rand(len(r_list))
            p_sum = probability.sum()

        probability = probability / p_sum
        try:
           ind = np.random.choice(len(r_list), p=probability)
        except ValueError:
            print(f"Nan:{v_a}:\n{state}")
            ind = np.argmax(probability[~np.isnan(probability)])
        return r_list[ind][0]

    def policy_max(self, state: State, alpha=0.1):
        r_list = self.get_next_state_reward(state)
        if not r_list:
            return None
        v_a = np.array([v[1] for v in r_list])
        if np.random.rand() < alpha:
            ind = np.random.choice(len(r_list))
        else:
            ind = v_a.argmax()  # max([(i, v[1]) for i, v in enumerate(r_list)], key=itemgetter(1))
        return r_list[ind][0]

    def policy_random(self, state: State):
        r_list = self.get_next_state_reward(state)
        if not r_list:
            return None
        return r_list[np.random.randint(len(r_list))][0]

    @staticmethod
    def policy_user(state: State):
        print(f"{state}")
        col = -1
        while not (0 <= col < state.COLS):
            try:
                col = int(input(f"enter col number [0-{state.COLS}]:"))
            except ValueError:
                print(f"values must be int between 0-{state.COLS}")
        return col


def reward(state: State, player):
    if state is None:
        return -1, False
    if state.player_win(player):
        return state.ROWS * state.COLS, True
    return 0, False


def play(player1: Player, player2: Player) -> Tuple[int, Union[State, None]]:
    previous_state = State()
    player1.player = 1
    if np.random.rand() < 0.5:
        player, previous_player = player1, player2
    else:
        player, previous_player = player2, player1
    i = 0
    last_reward = 0
    last_action = -1
    a = player.policy(previous_state)
    while True:
        state = previous_state.next(a, player.player)
        current_reward, is_wining = reward(state, player.player)
        player.update_q(previous_state, a, current_reward-last_reward)
        if is_wining:
            previous_player.update_q(state, last_action, -current_reward+last_reward)
            break
        if state is None:
            previous_player.update_q(state, last_action, current_reward-last_reward)
            return 0, state
        player, previous_player = previous_player, player
        last_action = a
        a = player.policy(state)
        previous_state = state
        last_reward = current_reward
        i += 1

    return player.player, state


def play_interactive(player_path: str):
    player1 = Player.load(Path(player_path))
    player2 = Player(player=-1, policy=3)
    flag = True
    while flag:
        winner, state = play(player1, player2)
        print(f"Player {winner} Wins: {state}")
        flag = None
        while flag is None:
            x = input(f"Do you want to play another play[Y|N]:")
            if x == 'N':
                flag = False
            elif x == 'Y':
                flag = True
    player1.save(Path(player_path).parent)


def main2():
    player1: Player = Player.load(Path(r"/home/urihein/data/four_in_row/player_1.pickle"))
    player2: Player = Player.load(Path(r"/home/urihein/data/four_in_row/player_1.pickle"))
    # player2: Player = Player(player=-1)
    player2.player = -1
    player1_wins, player2_wins, draws = 0, 0, 0
    for i in range(50000000):
        winner = play(player1, player2)[0]
        if winner == 1:
            player1_wins += 1
        elif winner == -1:
            player2_wins += 1
        else:
            draws += 1
        if (i % 1000) == 0:
            s = player2_wins + player1_wins + draws
            print(
                f"{datetime.now().strftime('%H:%M:%S')}-{i}, draw:{draws / s} 1:{player1_wins / s} 2:{player2_wins / s}"
            )
            player1_wins, player2_wins, draws = 0, 0, 0
        if i >= 1000000 and (i % 1000000) == 0:
            player1.save(Path(r"/home/urihein/data/four_in_row"))

    player1.save(Path(r"/home/urihein/data/four_in_row"))
    # player2.save(Path(r"/home/urihein/data/four_in_row"))


def main():
    player1 = Player.load(Path(r"/home/urihein/data/four_in_row/player_1.pickle"))#Player(1, policy=2)
    player2 = Player(-1, policy=1)
    player1_wins, player2_wins, draws = 0, 0, 0
    for i in range(10000000):
        winner = play(player1, player2)[0]
        if winner == 1:
            player1_wins += 1
        elif winner == -1:
            player2_wins += 1
        else:
            draws += 1
        if (i % 1000) == 0:
            s = player2_wins + player1_wins + draws
            print(
                f"{datetime.now().strftime('%H:%M:%S')}-{i}, draw:{draws / s} 1:{player1_wins / s} 2:{player2_wins / s}")
            player1_wins, player2_wins, draws = 0, 0, 0
        if i >= 100000 and (i % 100000) == 0:
            player1.save(Path(r"/home/urihein/data/four_in_row"))
            player2.save(Path(r"/home/urihein/data/four_in_row"))

    player1.save(Path(r"/home/urihein/data/four_in_row"))
    player2.save(Path(r"/home/urihein/data/four_in_row"))
    print("End")


if __name__ == "__main__":
    # main()
    play_interactive(r"/home/urihein/data/four_in_row/player_-1.pickle")
