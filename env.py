"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
import pickle
import itertools
from time import sleep, time
import random as rd
import os
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

GLOBAL_IDX = 0
LOG_PATH = "train.csv"
LOG_EPISODE_MAX = 0x7fffffff

# あさひくんの、私の、園田さん("CB512C5QDQ")の、
udid_list = ["P3PDU18321001333", "353477091491152", "353010080451240"]

# 園田, Android5
# udid_list = ["CB512C5QDQ", "482707805697"]
# udid_list = ["353010080451240", "CB512C5QDQ"]


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


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self):
        rotate = [0, 6]
        move = np.linspace(150.5, 929.5, 11, dtype=np.uint32)
        self.actions = np.array(list(itertools.product(rotate, move)))

        self.action_space = gym.spaces.Discrete(self.actions.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=TRAINNING_IMAGE_SIZE, dtype=np.uint8)
        self.reward_range = [0.0, 1.0]

        self.prev_height = None
        self.prev_animal_count = None

        self.device = AnimalTowerDevice()

        self.total_step_count = 0
        self.episode_count = 0

        # ヘッダのみ書き込み
        with open(LOG_PATH, "w", encoding="utf-8") as log_f:
            print("animals,height", file=log_f)
        print(f"{LOG_PATH} created or overwritten.")

        # 時間計測用
        self.t_0 = time()

    def reset(self) -> np.ndarray:
        """
        リセット
        """
        print(f"Episode({self.episode_count + 1})")
        print("Resetting...", end=" ", flush=True)
        self.prev_height = None
        self.prev_animal_count = None
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while self.prev_height is None or self.prev_animal_count is None:
            # リトライボタンをタップして3秒待つ
            self.device.tap(COORDINATES_RETRY)
            sleep(self.device.retry_intarval)
            self.device.driver.save_screenshot(self.device.screenshot_path)
            img_bgr = cv2.imread(self.device.screenshot_path, 1)
            obs = to_training_image(img_bgr)
            self.device.img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.prev_height = self.device.get_height(
                mag=self.device.height_mag)
            self.prev_animal_count = self.device.get_animal_count(
                mag=self.device.height_mag)
            cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
            # デバッグ
            # print(f"初期動物数: {self.prev_animal_count}, 初期高さ: {self.prev_height}")
        print("Done")
        t_1 = time()
        print(f"リセット所要時間: {t_1 - self.t_0:4.2f}秒")
        self.t_0 = t_1
        return obs

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        print(f"step({self.total_step_count + 1})")
        action = self.actions[action]
        print(
            f"Action({action[0]}/{self.actions.shape[0]-1}), {action[0], action[1]}")
        # 回転と移動
        self.device.rotate_and_move(action)
        sleep(0.7)
        # 変数の初期化
        done = False
        reward = 0.0
        while True:
            self.device.driver.save_screenshot(self.device.screenshot_path)
            self.device.img_bgr = cv2.imread(self.device.screenshot_path, 1)
            try:
                obs = to_training_image(self.device.img_bgr)
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                continue
            self.device.img_gray = cv2.cvtColor(
                self.device.img_bgr, cv2.COLOR_BGR2GRAY)
            # x8 speederが無効化された場合
            if self.device.is_off_x8(mag=self.device.height_mag):
                print("x8 speederを適用")
                self.device.tap((1032, 1857))
                sleep(0.5)
                self.device.tap((726, 1171))
                sleep(5)
                # 画像は再読込
                continue
            # ループで必ず高さと動物数を取得
            height = self.device.get_height(mag=self.device.height_mag)
            animal_count = self.device.get_animal_count(
                mag=self.device.width_mag)
            print(
                f"動物数: {self.prev_animal_count} -> {animal_count}, \
                高さ: {self.prev_height} -> {height}")
            # 終端
            if self.device.is_result_screen(mag=self.device.height_mag):
                print("Game over")
                done = True
                # ログファイルに書き出し
                with open(LOG_PATH, "a", encoding="utf-8") as log_f:
                    print(
                        f"{self.prev_animal_count},{self.prev_height}", file=log_f)
                self.episode_count += 1
                assert self.episode_count < LOG_EPISODE_MAX, f"エピソード{LOG_EPISODE_MAX}到達"
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                print("結果画面遷移中")
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
            sleep(self.device.pooling_intarval)
        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count
        self.total_step_count += 1
        # 共通処理
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        t_1 = time()
        print(f"ステップ所要時間: {t_1 - self.t_0:4.2f}秒")
        self.t_0 = t_1
        print(f"return obs, {reward}, {done}, {{}}")
        print("-"*NUM_OF_DELIMITERS)
        return obs, reward, done, {}

    def render(self, mode=None):
        pass


class AnimalTowerDevice():
    """
    どうぶつタワーが起動してるデバイスに関するクラス
    """

    def __init__(self, x8_enabled=True):
        # udidを選択
        if os.path.exists("idx.pickle"):
            with open("idx.pickle", "rb") as pickle_f:
                i = pickle.load(pickle_f)
        else:
            i = 0
        udid = udid_list[i]
        with open("idx.pickle", "wb") as pickle_f:
            pickle.dump((i + 1) % len(udid_list), pickle_f)
        sleep(rd.random() * 10)
        print(f"Connecting to {udid}...", end=" ", flush=True)
        caps = {
            "platformName": "android",
            "appium:udid": udid,
            "appium:ensureWebviewHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True
        }
        self.driver = webdriver.Remote(
            "http://localhost:4723/wd/hub", caps)
        print(f"Done [localhost:4723/wd/hub: {udid}]")

        print("Checking device...", end=" ", flush=True)
        self.screenshot_path = f"./screenshot_{udid}.png"
        self.driver.save_screenshot(self.screenshot_path)
        img_bgr = cv2.imread(self.screenshot_path, 1)
        self.mag = img_bgr.shape[0] / 1920, img_bgr.shape[1] / 1080
        self.actions = ActionChains(self.driver)
        self.actions.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
        self.height_mag = img_bgr.shape[0] / 1920
        self.width_mag = img_bgr.shape[1] / 1080
        self.move_tap_height = 800 * self.height_mag
        # そもそもx8が使えない
        if udid == "482707805697":
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
        print(f"Done [{img_bgr.shape}, {x8_enabled}(x8)]")

        self.img_gray = None
        self.img_bgr = None

    def is_off_x8(self, mag=1.0):
        """
        x8の停止を検知
        """
        template = cv2.imread("src/x8_start.png", 0)
        height, witdh = template.shape
        template = cv2.resize(template, (int(witdh*mag), int(height*mag)))
        res = cv2.matchTemplate(
            self.img_gray, template, cv2.TM_CCOEFF_NORMED)
        # print(res.max())
        return res.max() >= TEMPLATE_MATCHING_THRESHOLD

    def is_result_screen(self: np.ndarray, mag=1.0) -> bool:
        """
        Check the back button to determine game end.
        """
        template = cv2.imread("src/back.png", 0)
        height, width = template.shape
        template = cv2.resize(template, (int(width*mag), int(height*mag)))
        res = cv2.matchTemplate(
            self.img_gray, template, cv2.TM_CCOEFF_NORMED)
        # print("バックボタン一致率", res.max())
        return res.max() >= TEMPLATE_MATCHING_THRESHOLD

    def get_height(self, mag=1.0) -> float | None:
        """
        Get height
        """
        img_gray_height = self.img_gray[int(60*mag):int(130*mag), :]
        dict_digits = {}
        for i in list(range(10))+["dot"]:
            template = cv2.imread(f"src/height{i}.png", 0)
            height, width = template.shape
            template = cv2.resize(template, (int(width*mag), int(height*mag)))
            res = cv2.matchTemplate(
                img_gray_height, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= TEMPLATE_MATCHING_THRESHOLD)
            # print(loc)
            for loc_y in loc[1]:
                dict_digits[loc_y] = i
        height = ""
        prev_loc_x = -float("inf")
        for loc_x, key in sorted(dict_digits.items()):
            if loc_x - prev_loc_x >= 5:
                if key == "dot":
                    height += "."
                else:
                    height += str(key)
            prev_loc_x = loc_x
        if height:
            height = float(height)
        else:
            height = None
        return height

    def get_animal_count(self, mag=1.0) -> int | None:
        """
        動物の数を取得
        引数にはカラー画像を与える!!
        """
        img_shadow = cv2.inRange(
            self.img_bgr[int(264*mag):int(328*mag)], BACKGROUND_COLOR_DARK, WHITE)
        dict_digits = {}
        for i in range(10):
            template = cv2.imread(f"src/count{i}_shadow.png", 0)
            height, width = template.shape
            template = cv2.resize(template, (int(width*mag), int(height*mag)))
            res = cv2.matchTemplate(
                img_shadow, template, cv2.TM_CCOEFF_NORMED)
            # print(i, res.max())
            loc = np.where(res >= ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD)
            for loc_y in loc[1]:
                dict_digits[loc_y] = i
        animal_num = ""
        prev_loc_x = -float("inf")
        for loc_x, key in sorted(dict_digits.items()):
            if loc_x - prev_loc_x >= 5:
                animal_num += str(key)
            prev_loc_x = loc_x
        if animal_num:
            animal_num = int(animal_num)
        else:
            animal_num = None
        return animal_num

    def tap(self, coordinates: tuple) -> None:
        """
        Tap
        """
        loc_x = coordinates[0] * self.width_mag
        loc_y = coordinates[1] * self.height_mag
        # print(x, y)
        self.actions.w3c_actions.pointer_action.move_to_location(
            loc_x, loc_y)
        self.actions.w3c_actions.pointer_action.click()
        self.actions.perform()

    def rotate_and_move(self, action: np.ndarray) -> None:
        """
        高速化のために回転と移動を同時に操作
        移動の前にperformをしないとバグる
        """
        # 回転タップ
        # 0回転は処理を短縮
        if action[0] > 0:
            self.actions.w3c_actions.pointer_action.move_to_location(
                500 * self.width_mag, 1800 * self.height_mag)
            for _ in range(action[0]):
                self.actions.w3c_actions.pointer_action.click()
                # 試した感じ0.05がバグらない最低値
                # x8なしはわからない
                self.actions.w3c_actions.pointer_action.pause(
                    self.tap_intarval)
            # 重要
            self.actions.w3c_actions.perform()
        # print(a[1] * self.width_mag, self.move_tap_height)
        # 座標タップ
        self.actions.w3c_actions.pointer_action.move_to_location(
            action[1] * self.width_mag, self.move_tap_height)
        self.actions.w3c_actions.pointer_action.click()
        # 適用
        self.actions.w3c_actions.perform()
