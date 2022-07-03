"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
from time import sleep, time
import gym
import numpy as np
import cv2
from appium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction


SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
TEMPLATE_MATCHING_THRESHOLD = 0.99
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.984
TRAINNING_IMAGE_SIZE = 256, 75  # 適当（縦、横）
NUM_OF_DELIMITERS = 36
COORDINATES_RETRY = 200, 1755
COORDINATES_ROTATE30 = 500, 1800
COORDINATES_DROP = 540, 800
WAITTIME_AFTER_RETRY = 0.5
POLLING_INTERVAL = 0.1
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
    img_bin = cv2.bitwise_not(cv2.inRange(
        img_bgr, BACKGROUND_COLOR_DARK, WHITE))
    resized_and_cropped_img_bin = cv2.resize(
        img_bin[:1665, 295:785], dsize=TRAINNING_IMAGE_SIZE[::-1])
    return resized_and_cropped_img_bin


def is_off_x8(img_gray):
    template = cv2.imread("src/x8_start.png", 0)
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    # print(res.max())
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD

class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self):
        print("Initializing...", end=" ", flush=True)
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAINNING_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [-1, 1]
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
        print("Done")
        print("-"*NUM_OF_DELIMITERS)

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
            sleep(WAITTIME_AFTER_RETRY)
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
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        print(f"Action({action:.0f})")
        for _ in range(int(action)):
            self._tap(COORDINATES_ROTATE30)
        # タップして4秒待機
        self._tap(COORDINATES_DROP)
        # 変数の初期化
        done = False
        reward = 0
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
                sleep(7)
            # ループで必ず高さと動物数を取得
            height = get_height(img_gray)
            animal_count = get_animal_count(img_bgr)
            print(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}")
            # 終端
            if is_result_screen(img_gray):
                if height is None and animal_count is None:
                    reward = -1
                print("Game over")
                done = True
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                print("結果画面遷移中")
                pass
            # 高さ更新を検知
            elif height > self.prev_height:
                print(f"Height update: {height}m")
                reward = 1
                break
            # 高さ更新はないが動物数更新を検知
            elif animal_count > self.prev_animal_count:
                print("No height update")
                # 高さ更新がない場合の報酬は1らしい
                reward = 1
                break
            sleep(POLLING_INTERVAL)
        # ステップの終わりに必ず高さと動物数を更新!!
        # これをしないと, 動物数が変化したにもかかわらずprev_animal_countが更新されない場合がある
        self.prev_height = height
        self.prev_animal_count = animal_count
        # 共通処理
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        print(f"return obs, {reward}, {done}, {{}}")
        obs_3d = np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))
        print("-"*NUM_OF_DELIMITERS)
        return obs_3d, reward, done, {}

    def render(self):
        pass

    def _tap(self, coordinates: tuple) -> None:
        """
        Tap
        """
        # アンパックに変更
        self.operations.w3c_actions.pointer_action.move_to_location(
            *coordinates)
        self.operations.w3c_actions.pointer_action.pointer_down()
        self.operations.w3c_actions.pointer_action.release()
        self.operations.perform()
