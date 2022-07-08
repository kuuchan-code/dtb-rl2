"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
import pickle
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
import random as rd
import os


SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
TEMPLATE_MATCHING_THRESHOLD = 0.95
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.85
TRAINNING_IMAGE_SIZE = 256, 75  # small
TRAINNING_IMAGE_SIZE = 256, 144  # big
NUM_OF_DELIMITERS = 36
COORDINATES_RETRY = 200, 1755
COORDINATES_CENTER = 540, 800
# 背景色 (bgr)
BACKGROUND_COLOR = np.array([251, 208, 49], dtype=np.uint8)
BACKGROUND_COLOR_DARK = BACKGROUND_COLOR - 4
BLACK = np.zeros(3, dtype=np.uint8)
WHITE = BLACK + 255
WHITE_DARK = WHITE - 15

global_idx = 0

# あさひくんの、私の、園田さん("CB512C5QDQ")の、
udid_list = ["P3PDU18321001333", "353477091491152", "353010080451240"]

# 園田, Android5
# udid_list = ["CB512C5QDQ", "482707805697"]
# udid_list = ["353010080451240", "CB512C5QDQ"]


def is_result_screen(img_gray: np.ndarray, mag=1.0) -> bool:
    """
    Check the back button to determine game end.
    """
    template = cv2.imread("src/back.png", 0)
    h, w = template.shape
    template = cv2.resize(template, (int(w*mag), int(h*mag)))
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    # print("バックボタン一致率", res.max())
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD


def get_height(img_gray: np.ndarray, mag=1.0) -> float | None:
    """
    Get height
    """
    img_gray_height = img_gray[int(60*mag):int(130*mag), :]
    dict_digits = {}
    for i in list(range(10))+["dot"]:
        template = cv2.imread(f"src/height{i}.png", 0)
        h, w = template.shape
        template = cv2.resize(template, (int(w*mag), int(h*mag)))
        res = cv2.matchTemplate(
            img_gray_height, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= TEMPLATE_MATCHING_THRESHOLD)
        # print(loc)
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    height = ""
    prev_x = -float("inf")
    for x, key in sorted(dict_digits.items()):
        if x - prev_x >= 5:
            if key == "dot":
                height += "."
            else:
                height += str(key)
        prev_x = x
    if height:
        height = float(height)
    else:
        height = None
    return height


def get_animal_count(img_bgr: np.ndarray, mag=1.0) -> int | None:
    """
    動物の数を取得
    引数にはカラー画像を与える!!
    """
    img_shadow = cv2.inRange(
        img_bgr[int(264*mag):int(328*mag)], BACKGROUND_COLOR_DARK, WHITE)
    dict_digits = {}
    for i in range(10):
        template = cv2.imread(f"src/count{i}_shadow.png", 0)
        h, w = template.shape
        template = cv2.resize(template, (int(w*mag), int(h*mag)))
        res = cv2.matchTemplate(
            img_shadow, template, cv2.TM_CCOEFF_NORMED)
        # print(i, res.max())
        loc = np.where(res >= ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD)
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    animal_num = ""
    prev_x = -float("inf")
    for x, key in sorted(dict_digits.items()):
        if x - prev_x >= 5:
            animal_num += str(key)
        prev_x = x
    if animal_num:
        animal_num = int(animal_num)
    else:
        animal_num = None
    return animal_num


def to_training_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    入力BGR画像を訓練用画像にする
    """
    # 小さい盤面
    # img_bin = cv2.bitwise_not(cv2.inRange(
    #     img_bgr, BACKGROUND_COLOR_DARK, WHITE))
    # cropped_img_bin = img_bin[:1665, 295:785]
    # resized_and_cropped_img_bin = cv2.resize(cropped_img_bin, TRAINNING_IMAGE_SIZE)
    # return resized_and_cropped_img_bin

    # 大きい盤面

    return cv2.bitwise_not(cv2.inRange(
        cv2.resize(img_bgr, dsize=TRAINNING_IMAGE_SIZE[::-1]), BACKGROUND_COLOR_DARK, WHITE))


def is_off_x8(img_gray, mag=1.0):
    """
    x8の停止を検知
    """
    template = cv2.imread("src/x8_start.png", 0)
    h, w = template.shape
    template = cv2.resize(template, (int(w*mag), int(h*mag)))
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    # print(res.max())
    return res.max() >= TEMPLATE_MATCHING_THRESHOLD


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self, log_path="train.csv", log_episode_max=0x7fffffff, x8_enabled=True):
        sleep(rd.random() * 10)
        if os.path.exists("idx.pickle"):
            with open("idx.pickle", "rb") as f:
                i = pickle.load(f)
        else:
            i = 0
        my_udid = udid_list[i]
        with open("idx.pickle", "wb") as f:
            pickle.dump((i + 1) % len(udid_list), f)
        self.SCREENSHOT_PATH = f"./screenshot_{my_udid}.png"
        print("Initializing...", end=" ", flush=True)
        print(my_udid)
        r = [0, 6]
        m = np.linspace(150.5, 929.5, 11, dtype=np.uint32)
        self.ACTION_MAP = np.array([v for v in itertools.product(r, m)])
        # 出力サイズを変更し忘れていた!!
        self.action_space = gym.spaces.Discrete(self.ACTION_MAP.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=TRAINNING_IMAGE_SIZE, dtype=np.uint8)
        self.reward_range = [0.0, 1.0]
        caps = {
            "platformName": "android",
            "appium:udid": my_udid,
            "appium:ensureWebviewHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True
        }
        print(f"http://localhost:4723/wd/hub")
        self.driver = webdriver.Remote(
            "http://localhost:4723/wd/hub", caps)

        # 解像度チェッカー
        self.driver.save_screenshot(SCREENSHOT_PATH)
        img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
        print(img_bgr.shape)
        self.height_mag = img_bgr.shape[0] / 1920
        self.width_mag = img_bgr.shape[1] / 1080

        self.move_tap_height = 800 * self.height_mag

        print(self.height_mag, self.width_mag, self.move_tap_height)

        self.operations = ActionChains(self.driver)
        self.operations.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
        self.log_path = log_path
        self.total_step_count = 0
        self.episode_count = 0
        self.log_episode_max = log_episode_max

        # そもそもx8が使えない
        if my_udid == "482707805697":
            x8_enabled = False

        # x8が有効かどうかでタップ間隔を変える
        if x8_enabled:
            self.tap_intarval = 0.05
            self.retry_intarval = 0.5
            self.pooling_intarval = 0.1
        else:
            self.tap_intarval = 0.2
            self.retry_intarval = 2
            self.pooling_intarval = 0.4

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
        print(f"episode({self.episode_count + 1})")
        print("Resetting...", end=" ", flush=True)
        self.prev_height = None
        self.prev_animal_count = None
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while self.prev_height is None or self.prev_animal_count is None:
            # リトライボタンをタップして3秒待つ
            self._tap(COORDINATES_RETRY)
            sleep(self.retry_intarval)
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            obs = to_training_image(img_bgr)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.prev_height = get_height(img_gray, mag=self.height_mag)
            self.prev_animal_count = get_animal_count(
                img_bgr, mag=self.height_mag)
            cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
            # デバッグ
            print(f"初期動物数: {self.prev_animal_count}, 初期高さ: {self.prev_height}")
        print("Done")
        t1 = time()
        print(f"リセット所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1
        return obs

    def step(self, action_index) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        print(f"step({self.total_step_count + 1})")
        action = self.ACTION_MAP[action_index]
        # 何番目のactionか出力
        print(
            f"Action({action_index}/{self.ACTION_MAP.shape[0]-1}), {action[0], action[1]}")
        # 回転と移動
        self._rotate_and_move(action)
        sleep(0.7)
        # 変数の初期化
        done = False
        reward = 0.0
        while True:
            self.driver.save_screenshot(self.SCREENSHOT_PATH)
            img_bgr = cv2.imread(self.SCREENSHOT_PATH, 1)
            if img_bgr is None:
                continue
            obs = to_training_image(img_bgr)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # x8 speederが無効化された場合
            if is_off_x8(img_gray, mag=self.height_mag):
                print("x8 speederを適用")
                self._tap((1032, 1857))
                sleep(0.5)
                self._tap((726, 1171))
                sleep(5)
                # 画像は再読込
                continue
            # ループで必ず高さと動物数を取得
            height = get_height(img_gray, mag=self.height_mag)
            animal_count = get_animal_count(img_bgr, mag=self.width_mag)
            print(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}")
            # 終端
            if is_result_screen(img_gray, mag=self.height_mag):
                print("Game over")
                done = True
                # ログファイルに書き出し
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
            sleep(self.pooling_intarval)
        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count
        self.total_step_count += 1
        # 共通処理
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        t1 = time()
        print(f"ステップ所要時間: {t1 - self.t0:4.2f}秒")
        self.t0 = t1
        print(f"return obs, {reward}, {done}, {{}}")
        print("-"*NUM_OF_DELIMITERS)
        return obs, reward, done, {}

    def _tap(self, coordinates: tuple) -> None:
        """
        Tap
        """
        x = coordinates[0] * self.width_mag
        y = coordinates[1] * self.height_mag
        self.operations.w3c_actions.pointer_action.move_to_location(
            x, y)
        self.operations.w3c_actions.pointer_action.click()
        self.operations.perform()

    def _rotate_and_move(self, a: np.ndarray) -> None:
        """
        高速化のために回転と移動を同時に操作
        移動の前にperformをしないとバグる
        """
        # 回転タップ
        # 0回転は処理を短縮
        if a[0] > 0:
            self.operations.w3c_actions.pointer_action.move_to_location(
                500 * self.width_mag, 1800 * self.height_mag)
            for _ in range(a[0]):
                self.operations.w3c_actions.pointer_action.click()
                # 試した感じ0.05がバグらない最低値
                # x8なしはわからない
                self.operations.w3c_actions.pointer_action.pause(
                    self.tap_intarval)
            # 重要
            self.operations.w3c_actions.perform()
        # print(a[1] * self.width_mag, self.move_tap_height)
        # 座標タップ
        self.operations.w3c_actions.pointer_action.move_to_location(
            a[1] * self.width_mag, self.move_tap_height)
        self.operations.w3c_actions.pointer_action.click()
        # 適用
        self.operations.w3c_actions.perform()
