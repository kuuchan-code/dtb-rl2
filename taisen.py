#!/usr/bin/env python3
"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
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


SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
TEMPLATE_MATCHING_THRESHOLD = 0.99
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.984
TRAINNING_IMAGE_SIZE = 256, 75  # small
TRAINNING_IMAGE_SIZE = 256, 144  # big
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
        cropped_img_bin, TRAINNING_IMAGE_SIZE)
    return resized_and_cropped_img_bin

    # 大きい盤面
    # return cv2.bitwise_not(cv2.inRange(
    #     cv2.resize(img_bgr, dsize=TRAINNING_IMAGE_SIZE[::-1]), BACKGROUND_COLOR_DARK, WHITE))


def is_off_x8(img_gray):
    """
    x8の停止を検知
    """
    template = cv2.imread("src/x8_start.png", 0)
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    # print(res.max())
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD


class AnimalTowerServer(threading.Thread):
    """
    端末を動かすためのスレッド
    """

    def __init__(self):
        super(AnimalTowerServer, self).__init__()
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

    def run(self):
        """
        実行部分
        """
        self.running = True
        while self.running:
            self.driver.save_screenshot(SCREENSHOT_PATH)

    def stop(self):
        """
        停止
        """
        self.running = False


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self, player, log_path="train.csv", log_episode_max=0x7fffffff):
        print("Initializing...", end=" ", flush=True)
        r = np.linspace(0, 11, 12, dtype=np.uint8)
        # b = [150, 540, 929]
        m = np.linspace(440, 640, 3, dtype=np.uint32)
        self.player = player
        self.ACTION_MAP = np.array([v for v in itertools.product(r, m)])
        np.random.seed(0)
        np.random.shuffle(self.ACTION_MAP)
        self.action_space = gym.spaces.Discrete(self.ACTION_MAP.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAINNING_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [0.0, 1.0]
        caps = {
            "platformName": "android",
            "appium:ensureWebviewHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True
        }
        self.driver = webdriver.Remote(
            "http://localhost:4723/wd/hub", caps)
        self.operations = ActionChains(self.driver)
        self.operations.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))

        self.log_path = log_path
        self.episode_count = 0
        self.log_episode_max = log_episode_max
        # ヘッダのみ書き込み
        with open(self.log_path, "w") as f:
            print(f"animals,height", file=f)

        print("Done")
        print("-"*NUM_OF_DELIMITERS)

        # 時間計測用
        self.t0 = time()

    def reset(self) -> np.ndarray:
        """
        リセット
        """
        print("Resetting...", end=" ", flush=True)
        self.prev_height = None
        self.prev_animal_count = None
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while self.prev_height is None or self.prev_animal_count is None:
            # リトライボタンをタップして3秒待つ
            self._tap(COORDINATES_RETRY)
            sleep(0.5)
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            obs = to_training_image(img_bgr)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.prev_height = get_height(img_gray)
            self.prev_animal_count = get_animal_count(img_bgr)
            cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
            # デバッグ
            print(f"初期動物数: {self.prev_animal_count}, 初期高さ: {self.prev_height}")
        print("Done")
        t1 = time()
        print(f"リセット所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))

    def step(self, action_index) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        self.driver.save_screenshot(SCREENSHOT_PATH)
        img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
        animal_count = get_animal_count(img_bgr)
        while animal_count is None or animal_count % 2 != int(self.player[-1])-1:
            print(f"{self.player}待機中...")
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            animal_count = get_animal_count(img_bgr)
            sleep(10)
        action = self.ACTION_MAP[action_index]
        print(f"Action({action[0], action[1]})")
        # 回転と移動
        self._rotate_and_move(action)
        sleep(0.7)
        # 変数の初期化
        done = False
        reward = 0.0
        while True:
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            obs = to_training_image(img_bgr)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if is_off_x8(img_gray):
                print("x8 speederを適用")
                self._tap((1032, 1857))
                sleep(0.5)
                self._tap((726, 1171))
                sleep(5)
                continue
            # ループで必ず高さと動物数を取得
            height = get_height(img_gray)
            animal_count = get_animal_count(img_bgr)
            print(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}")
            # 終端
            if is_result_screen(img_gray):
                print("Game over")
                done = True
                with open(self.log_path, "a") as f:
                    print(f"{self.prev_animal_count},{self.prev_height}", file=f)
                self.episode_count += 1
                assert self.episode_count < self.log_episode_max, f"エピソード{self.log_episode_max}到達"
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                print("結果画面遷移中")
                pass
            # 高さ更新を検知
            elif height > self.prev_height:
                print(f"Height update: {height}m")
                reward = 1.0
                break
            # 高さ更新はないが動物数更新を検知
            elif animal_count > self.prev_animal_count:
                print("No height update")
                reward = 1.0
                break
            sleep(0.1)
        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count
        # 共通処理
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        t1 = time()
        print(f"ステップ所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1
        print(f"return obs, {reward}, {done}, {{}}")
        print("-"*NUM_OF_DELIMITERS)
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE)), reward, done, {}

    def render(self):
        pass

    def _tap(self, coordinates: tuple) -> None:
        """
        Tap
        """
        self.operations.w3c_actions.pointer_action.move_to_location(
            *coordinates)
        self.operations.w3c_actions.pointer_action.click()
        self.operations.w3c_actions.pointer_action.pause(0.05)
        self.operations.perform()

    def _rotate_and_move(self, a: np.ndarray) -> None:
        """
        高速化のために回転と移動を同時に操作
        """
        # 回転タップ
        self.operations.w3c_actions.pointer_action.move_to_location(
            *COORDINATES_ROTATE30)
        for _ in range(a[0]):
            self.operations.w3c_actions.pointer_action.click()
            self.operations.w3c_actions.pointer_action.pause(0.05)
        self.operations.w3c_actions.perform()
        # 座標タップ
        self.operations.w3c_actions.pointer_action.move_to_location(
            a[1], 800)
        self.operations.w3c_actions.pointer_action.click()
        # 適用
        self.operations.w3c_actions.perform()


if __name__ == "__main__":
    print(threading.enumerate())
    dtb_server = AnimalTowerServer()
    try:
        dtb_server.start()
        print(threading.enumerate())
        for i in range(10):
            print(i)
            print(dtb_server.is_alive())
            sleep(1)
    except Exception as e:
        raise e
    finally:
        print("最後")
        dtb_server.stop()
    print(threading.enumerate())
