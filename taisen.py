#!/usr/bin/env python3
"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
from datetime import datetime
import itertools
from time import sleep, time
import gym
import numpy as np
import cv2
from appium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction
import threading
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import random as rd


SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
TEMPLATE_MATCHING_THRESHOLD = 0.99
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.984
TRAINNING_IMAGE_SIZE = 256, 75  # small
SCREENSHOT_SIZE = 1920, 1080, 3
# TRAINNING_IMAGE_SIZE = 256, 144  # big
NUM_OF_DELIMITERS = 36
COORDINATES_RETRY = 200, 1755
COORDINATES_ROTATE30 = 500, 1800
COORDINATES_CENTER = 540, 800
# 背景色 (bgr)
BACKGROUND_COLOR = np.array([251, 208, 49], dtype=np.uint8)
BACKGROUND_COLOR_DARK = BACKGROUND_COLOR - 4
BLACK = np.zeros(3, dtype=np.uint8)
WHITE = BLACK + 255
WHITE_DARK = WHITE - 15


def is_result_screen(img_gray: np.ndarray) -> bool:
    """
    Check the back button to determine game end.
    """
    template = cv2.imread("src/back.png", 0)
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    # print(res.max())
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD


def get_height(img_gray: np.ndarray) -> float | None:
    """
    Get height
    """
    img_gray_height = img_gray[65:129, :]
    dict_digits = {}
    for i in list(range(10))+["dot"]:
        template = cv2.imread(f"src/height{i}.png", 0)
        res = cv2.matchTemplate(
            img_gray_height, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= TEMPLATE_MATCHING_THRESHOLD)
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    height = ""
    for key in sorted(dict_digits.items()):
        if key[1] == "dot":
            height += "."
        else:
            height += str(key[1])
    if height:
        height = float(height)
    else:
        height = None
    return height


def get_animal_count(img_bgr: np.ndarray) -> int | None:
    """
    動物の数を取得
    引数にはカラー画像を与える!!
    """
    img_shadow = cv2.inRange(
        img_bgr[264:328], BACKGROUND_COLOR_DARK, WHITE)
    dict_digits = {}
    for i in range(10):
        template = cv2.imread(f"src/count{i}_shadow.png", 0)
        res = cv2.matchTemplate(
            img_shadow, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD)
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    if dict_digits:
        animal_num = int("".join([str(i)
                         for _, i in sorted(dict_digits.items())]))
    else:
        animal_num = None
    return animal_num


def to_training_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    入力BGR画像を訓練用画像にする
    """
    # 小さい盤面
    img_bin = cv2.bitwise_not(cv2.inRange(
        img_bgr, BACKGROUND_COLOR_DARK, WHITE))
    cropped_img_bin = img_bin[:1665, 295:785]
    resized_and_cropped_img_bin = cv2.resize(
        cropped_img_bin, TRAINNING_IMAGE_SIZE[::-1])
    return resized_and_cropped_img_bin

    # 大きい盤面
    # return cv2.bitwise_not(cv2.inRange(
    #     cv2.resize(img_bgr, dsize=TRAINNING_IMAGE_SIZE[::-1]), BACKGROUND_COLOR_DARK, WHITE))


def is_off_x8(img_bgr: np.ndarray) -> bool:
    """
    x8の停止を検知
    """
    template = cv2.imread("src/x8_start.png", 1)
    res = cv2.matchTemplate(
        img_bgr, template, cv2.TM_CCOEFF_NORMED)
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD


class AnimalTowerBattleServer(threading.Thread):
    """
    端末を動かすためのスレッド
    """

    def __init__(self):
        super(AnimalTowerBattleServer, self).__init__()
        print("Appium設定中")
        caps = {
            "platformName": "android",
            "appium:ensureWebviewHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True
        }
        self.driver = webdriver.Remote("http://localhost:4723/wd/hub", caps)
        # 操作
        self.operations = ActionChains(self.driver)
        self.operations.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
        self.running = False
        self.img_bgr = np.random.randint(0, 256, SCREENSHOT_SIZE)
        print(self.img_bgr.shape)
        # タスクを溜めるバッファ
        self.task_queue = []

        # どちらが先手後手か決める
        # [Player1のターン, Player2のターン]
        self.turns = [0, 1]
        self.info = {"img_gray": None}

    def run(self):
        """
        実行部分
        """
        self.running = True
        # 停止まで無限ループ
        while self.running:
            self._execute_tasks()
            # ひたすらスクショ
            self.driver.save_screenshot(SCREENSHOT_PATH)
            # bgr画像をメモリにロード
            # 画像処理はクライアント側に任せる
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.img_bgr = img_bgr
            self.info["img_gray"] = img_gray
            if is_off_x8(self.img_bgr):
                print("Server   : x8 speederを適用")
                self.add_task(("server", "apply_x8"))

    def get_image(self):
        """
        bgrのスクショを返す
        """
        return self.img_bgr

    def add_task(self, task: tuple):
        """
        タスクを追加
        (送信元, タスク名, その他)
        """
        # print(task)
        self.task_queue.append(task)

    def get_turn(self, p: int):
        """
        ターンを返す
        """
        return self.turns[p]

    def stop(self):
        """
        停止
        """
        self.running = False

    def _execute_tasks(self):
        # リトライ重複回避用
        first_retry = True
        # タスク消化
        while self.task_queue:
            # print(f"Server   : タスク一覧 {self.task_queue}")
            # 先頭のタスク取り出し
            task = self.task_queue.pop(0)
            # 回転と移動操作
            if task[1] == "rotate_move":
                self._rotate_and_move(task[2])
            elif task[1] == "apply_x8":
                self._apply_x8()
            elif task[1] == "retry":
                if first_retry:
                    self._retry()
                    first_retry = False

    def _tap(self, coordinates):
        """
        タップ
        """
        self.operations.w3c_actions.pointer_action.move_to_location(
            *coordinates)
        self.operations.w3c_actions.pointer_action.click()
        self.operations.w3c_actions.pointer_action.pause(0.05)
        self.operations.perform()

    def _rotate_and_move(self, a):
        """
        回転と移動
        """
        if a[0] > 0:
            # 回転タップ
            self.operations.w3c_actions.pointer_action.move_to_location(
                *COORDINATES_ROTATE30)
            for _ in range(a[0]):
                self.operations.w3c_actions.pointer_action.click()
                self.operations.w3c_actions.pointer_action.pause(0.05)
            # まとめて適用
            self.operations.w3c_actions.perform()
        # 座標タップ
        self.operations.w3c_actions.pointer_action.move_to_location(
            a[1], 800)
        self.operations.w3c_actions.pointer_action.click()
        # 適用
        self.operations.w3c_actions.perform()

    def _retry(self):
        """
        リセットするまでずっとここにいる
        """
        self.prev_height = None
        self.prev_animal_count = None
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while self.prev_height is None or self.prev_animal_count is None:
            # リトライボタンをタップして少し待つ
            self._tap(COORDINATES_RETRY)
            sleep(0.5)
            # リセットが確認できるまでスクショ
            self.driver.save_screenshot(SCREENSHOT_PATH)
            # サーバから画像取得
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.prev_height = get_height(img_gray)
            self.prev_animal_count = get_animal_count(img_bgr)
        # 外から参照可能とする
        self.img_bgr = img_bgr
        # rd.shuffle(self.turns)
        print("-"*NUM_OF_DELIMITERS)
        print(f"Server   : ターンリセット{self.turns}")
        # print("Server   : リセット完了")

    def _apply_x8(self):
        """
        x8 speeder の適用
        """
        self._tap((1032, 1857))
        sleep(0.5)
        self._tap((726, 1171))
        sleep(5)


class AnimalTowerBattleClient(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self, dtb_server: AnimalTowerBattleServer, player: int, log_path="train.csv", log_episode_max=0x7fffffff, verbose=2):
        self.player = player
        self.dtb_server = dtb_server
        self.verbose = verbose
        self._print_with_name("Initializing...", 1)

        r = np.linspace(0, 11, 12, dtype=np.uint8)
        # b = [150, 540, 929]
        m = np.linspace(440, 640, 3, dtype=np.uint32)
        self.ACTION_MAP = np.array([v for v in itertools.product(r, m)])
        self.action_space = gym.spaces.Discrete(self.ACTION_MAP.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAINNING_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [0.0, 1.0]

        self.log_path = log_path
        self.episode_count = 0
        self.log_episode_max = log_episode_max

        self.obs_path = f"./observation_p{self.player}.png"

        # ヘッダのみ書き込み
        with open(self.log_path, "w") as f:
            print(f"animals,height", file=f)

        # print("-"*NUM_OF_DELIMITERS)

        # 時間計測用
        self.t0 = time()

    def reset(self) -> np.ndarray:
        """
        リセット
        2つのプレイヤーが同じコマンド送ってもいいや
        """
        self._print_with_name("Resetting...", 2)
        # リトライ申請
        self.dtb_server.add_task((self.player, "retry"))
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while True:
            # サーバから画像取得
            img_bgr = self.dtb_server.get_image()
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            obs = to_training_image(img_bgr)
            self.prev_height = get_height(img_gray)
            self.prev_animal_count = get_animal_count(img_bgr)
            # デバッグ
            self._print_with_name(
                f"初期動物数: {self.prev_animal_count}, 初期高さ: {self.prev_height}", 2)
            sleep(1)
            if self.prev_height is None:
                continue
            if self.prev_animal_count is None:
                continue
            # 自分のターンまでゆっくり待つ
            # 相手が初手で落としたら何もなかったことにする
            if self.prev_animal_count & 1 == self.dtb_server.get_turn(self.player):
                break

        cv2.imwrite(self.obs_path, obs)

        t1 = time()
        self._print_with_name(f"リセット所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))

    def step(self, action_index) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        # 自分のターン待ち
        # 相手の行動後の状態を観測値とする
        if self.prev_animal_count & 1 != self.dtb_server.get_turn(self.player):
            obs, reward, done, _ = self._wait_for_my_turn()
        action = self.ACTION_MAP[action_index]
        self._print_with_name(
            f"ActionID: ({action_index}/{self.ACTION_MAP.shape[0]-1}), Action: {action[0], action[1]}", 1)
        # 回転と移動
        self.dtb_server.add_task((self.player, "rotate_move", action))
        sleep(0.7)
        # 変数の初期化
        done = False
        reward = 0.0
        while True:
            # サーバから取得
            img_bgr = self.dtb_server.get_image()
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            obs = to_training_image(img_bgr)
            # ループで必ず高さと動物数を取得
            height = get_height(img_gray)
            animal_count = get_animal_count(img_bgr)
            self._print_with_name(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}", 2)
            # 終端
            if is_result_screen(img_gray):
                self._print_with_name("負け", 1)
                done = True
                reward = -1.0
                with open(self.log_path, "a") as f:
                    print(f"{self.prev_animal_count},{self.prev_height}", file=f)
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                self._print_with_name("結果画面遷移中")
                pass
            # 高さ更新を検知
            elif height > self.prev_height:
                self._print_with_name(f"Height update: {height}m")
                break
            # 高さ更新はないが動物数更新を検知
            elif animal_count > self.prev_animal_count:
                self._print_with_name("No height update")
                break
            sleep(1)

        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count

        # 自分のターン待ち
        # 相手の行動後の状態を観測値とする
        if done:
            cv2.imwrite(self.obs_path, obs)
            obs_3d = np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))
        elif self.prev_animal_count & 1 != self.dtb_server.get_turn(self.player):
            obs_3d, reward, done, _ = self._wait_for_my_turn()

        t1 = time()
        self._print_with_name(f"ステップ所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1

        self._print_with_name(f"return obs, {reward}, {done}, {{}}", 1)
        return obs_3d, reward, done, {}

    def render(self):
        pass

    def _wait_for_my_turn(self) -> tuple[np.ndarray, float, bool, dict]:
        """
        自分のターンを待つ
        """
        self._print_with_name(f"Player({self.player}) 待機中")
        # 変数の初期化
        reward = 0.0
        done = False
        while True:
            # サーバから取得
            img_bgr = self.dtb_server.get_image()
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            obs = to_training_image(img_bgr)
            # ループで必ず高さと動物数を取得
            height = get_height(img_gray)
            animal_count = get_animal_count(img_bgr)
            self._print_with_name(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}", 2)
            # 終端
            if is_result_screen(img_gray):
                self._print_with_name("勝ち", 1)
                done = True
                reward = 1.0
                with open(self.log_path, "a") as f:
                    print(f"{self.prev_animal_count},{self.prev_height}", file=f)
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                self._print_with_name("結果画面遷移中")
                pass
            # 高さ更新を検知
            elif height > self.prev_height:
                self._print_with_name(f"Height update: {height}m")
                break
            # 高さ更新はないが動物数更新を検知
            elif animal_count > self.prev_animal_count:
                self._print_with_name("No height update")
                break
            sleep(1)

        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count
        cv2.imwrite(self.obs_path, obs)
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE)), reward, done, {}

    def _print_with_name(self, moji: str, verbose=2):
        """
        文字列をプレイヤー名と一緒に出力
        引数はひとつだけ
        """
        if verbose <= self.verbose:
            print(f"Player({self.player}): {moji}")


class AnimalTowerBattleLearning(threading.Thread):
    """
    学習のためのスレッド
    """

    def __init__(self, dtb_server, player, verbose=2):
        super(AnimalTowerBattleLearning, self).__init__()

        name_prefix = f"_a2c_cnn_r12m3s_p{player}"
        env = AnimalTowerBattleClient(
            dtb_server, player, f"{name_prefix}.csv", verbose=1)

        self.model = A2C(policy="CnnPolicy", env=env,
                         verbose=1, tensorboard_log="tensorboard")
        self.checkpoint_callback = CheckpointCallback(
            save_freq=100, save_path="models",
            name_prefix=name_prefix
        )

    def run(self):
        """
        学習実行
        停止はどうしよう
        とりあえずキーボード割り込みを何回かしてもらう
        """
        try:
            self.model.learn(
                total_timesteps=10000,
                callback=[self.checkpoint_callback]
            )
        finally:
            print("学習停止")


if __name__ == "__main__":
    # print(threading.enumerate())
    dtb_server = AnimalTowerBattleServer()
    try:
        # サーバ開始
        dtb_server.start()
        sleep(5)
        print(threading.enumerate())
        verbose = 2

        learning_list = []
        for i in range(2):
            dtb_learning = AnimalTowerBattleLearning(
                dtb_server, i, verbose=verbose)
            # デーモンをセットすることで止まるらしい
            dtb_learning.daemon = True
            dtb_learning.start()
            learning_list.append(dtb_learning)

        # メインは謎のループ
        for i in range(1000000):
            sleep(300)
            print(datetime.now())

    except Exception as e:
        raise e
    # 別スレッドを停止させるため
    finally:
        print("最後")
        dtb_server.stop()

    print(threading.enumerate())
