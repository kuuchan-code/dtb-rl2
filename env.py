"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from __future__ import annotations
import pickle
import itertools
import re
from time import sleep, time
import random as rd
import os
import gym
from gym.spaces import Discrete, Box
import numpy as np
import cv2
from appium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction
from datetime import datetime

SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
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


# あさひくんの、私の、園田さん("CB512C5QDQ")の、
udid_list = ["P3PDU18321001333", "353477091491152", "353010080451240"]

# 園田, Android5
# udid_list = ["CB512C5QDQ", "482707805697"]
# udid_list = ["353010080451240", "CB512C5QDQ"]


class Spec:
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps
        self.id = rd.randint(0, 0x7fffffff)


class AnimalTowerDummy(gym.Env):
    """
    ダミー環境でテストしたい
    """
    BLOCKS_HEIGHT_MAX = 10

    def __init__(self):
        self.act_num = 22
        self.action_space = Discrete(self.act_num)
        self.observation_space = Box(
            low=0, high=255, shape=TRAINNING_IMAGE_SIZE, dtype=np.uint8)
        self.reward_range = [0.0, 1.0]
        self.each_height = np.zeros((self.act_num,), dtype=np.uint8)
        self.blocks = np.zeros(
            (self.BLOCKS_HEIGHT_MAX, self.act_num), dtype=np.uint8)

        self.total_step_count = 0

    def reset(self) -> np.ndarray:
        self.each_height = np.zeros((self.act_num,), dtype=np.uint8)
        self.blocks = np.zeros(
            (self.BLOCKS_HEIGHT_MAX, self.act_num), dtype=np.uint8)
        # self.blocks = np.random.randint(
        #     0, 2, (10, self.act_num), dtype=np.uint8)
        # print(self.blocks)
        self.blocks[9] = np.ones(self.act_num)
        obs = self.get_training_image()
        # cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return obs

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        self.total_step_count += 1
        self.each_height[action] += 1
        self.blocks[self.BLOCKS_HEIGHT_MAX -
                    self.each_height[action] - 1, action] = 1
        # print(self.blocks)
        obs = self.get_training_image()
        done = False
        reward = 1.0
        if self.each_height[action] >= 3:
            done = True
            reward = 0.0
        print(self.total_step_count)
        # cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        # sleep(0.1)
        return obs, reward, done, {}

    def get_training_image(self):
        return cv2.resize(self.blocks * 255, dsize=TRAINNING_IMAGE_SIZE[::-1], interpolation=cv2.INTER_LANCZOS4)


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self, udid=None, log_prefix="train", x8_enabled=True):
        rotate = [0, 4, 6, 8]
        move = np.linspace(150.5, 929.5, 11, dtype=np.uint32)
        self.actions = np.array(list(itertools.product(rotate, move)))
        # 行動パターン数
        self.action_patterns = self.actions.shape[0]
        self.action_space = gym.spaces.Discrete(self.action_patterns)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAINNING_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [0.0, 1.0]

        self.prev_height = None
        self.prev_animal_count = None

        # udidを選択
        # if os.path.exists("idx.pickle"):
        #     with open("idx.pickle", "rb") as pickle_f:
        #         i = pickle.load(pickle_f)
        # else:
        #     i = 0
        # udid = udid_list[i]
        # with open("idx.pickle", "wb") as pickle_f:
        #     pickle.dump((i + 1) % len(udid_list), pickle_f)
        # udid = "790908812299"
        # udid = "CB512C5QDQ"
        # udid = "482707805697"
        sleep(rd.random() * 10)
        print(f"Connecting to {udid}(Server localhost:4723/wd/hub)...")
        self.device = AnimalTowerDevice(udid, x8_enabled)

        self.episode_step_count = 0
        self.total_step_count = 0
        self.episode_count = 0

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.result_log_path = f"log/{log_prefix}_result_{now}.csv"
        self.action_log_path = f"log/{log_prefix}_action_{now}.csv"

        # ヘッダのみ書き込み
        with open(self.result_log_path, "w", encoding="utf-8") as log_f:
            print("animals,height", file=log_f)
        with open(self.action_log_path, "w", encoding="utf-8") as log_f:
            print("step,action", file=log_f)

        print(f"{self.result_log_path} created or overwritten.")

        # 時間計測用
        self.t_0 = time()

    def reset(self) -> np.ndarray:
        """
        リセット
        """
        self.episode_step_count = 0
        self.episode_count += 1
        print(f"episode({self.episode_count})")
        self.prev_height = None
        self.prev_animal_count = None
        # 初期状態がリザルト画面とは限らないため, 初期の高さと動物数を取得できるまでループ
        while self.prev_height is None or self.prev_animal_count is None:
            # リトライボタンをタップして3秒待つ
            self.device.tap(COORDINATES_RETRY)
            sleep(self.device.retry_intarval)
            self.device.update_image()
            obs = self.device.to_training_image()
            self.prev_height = self.device.get_height()
            self.prev_animal_count = self.device.get_animal_count()
            cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
            # デバッグ
            # print(f"初期動物数: {self.prev_animal_count}, 初期高さ: {self.prev_height}")
        t_1 = time()
        print(f"リセット所要時間: {t_1 - self.t_0:4.2f}秒")
        self.t_0 = t_1
        return obs

    def step(self, action_index) -> tuple[np.ndarray, float, bool, dict]:
        """
        1アクション
        """
        print(f"Step({self.total_step_count + 1})")
        action = self.actions[action_index]
        print(
            f"Action({action_index}/{self.actions.shape[0]-1}), {action[0], action[1]}")
        # 行動記録 (ステップと行動の添字)
        with open(self.action_log_path, "a", encoding="utf-8") as log_f:
            print(f"{self.episode_step_count},{action_index}", file=log_f)

        # 回転と移動
        self.device.rotate_and_move(action)
        # 変数の初期化
        done = False
        reward = 0.0
        # 動物登場時の煙
        maybe_smoke = True
        while True:
            self.device.update_image()
            try:
                obs = self.device.to_training_image()
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                sleep(self.device.pooling_intarval)
                continue
            # ループで必ず高さと動物数を取得
            height = self.device.get_height()
            animal_count = self.device.get_animal_count()
            print(
                f"動物数: {self.prev_animal_count} -> {animal_count}, 高さ: {self.prev_height} -> {height}"
            )
            # 終端
            if self.device.is_result_screen(mag=self.device.mag[0]):
                print("Game over")
                done = True
                # ログファイルに書き出し
                with open(self.result_log_path, "a", encoding="utf-8") as log_f:
                    print(
                        f"{self.prev_animal_count},{self.prev_height}", file=log_f)
                break
            # 結果画面ではないが, 高さもしくは動物数が取得できない場合
            elif height is None or animal_count is None:
                print("数値取得失敗")
            # 高さ更新はないが動物数更新を検知
            elif animal_count > self.prev_animal_count:
                # 更新があっても, 煙があるかもしれないので撮り直し
                if maybe_smoke:
                    maybe_smoke = False
                    continue

                # 高さの差を計算
                height_diff = height - self.prev_height
                if height_diff > 0:
                    print(f"Height update: {height}m")
                else:
                    print("No height update")

                # 高さの差が+0.5未満のみ報酬を与える
                # 同じ座標だけ選ぶのを回避したい
                if height_diff < 0.5:
                    reward = 1.0
                break
            sleep(self.device.pooling_intarval)
        # ステップの終わりに高さと動物数を更新
        self.prev_height = height
        self.prev_animal_count = animal_count
        self.episode_step_count += 1
        self.total_step_count += 1
        # 共通処理
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        t_1 = time()
        print(f"ステップ所要時間: {t_1 - self.t_0:4.2f}秒")
        self.t_0 = t_1
        print(f"return obs, {reward}, {done}, {{}}")
        print("-"*NUM_OF_DELIMITERS)
        # baseline3のCnnPolicyの場合必要
        obs_3d = np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))
        return obs_3d, reward, done, {}

    def render(self, mode=None):
        pass

    def close(self):
        print("Close the appium connection")
        self.device.driver.quit()


class AnimalTowerDevice():
    """
    どうぶつタワーが起動してるデバイスに関するクラス
    """

    def __init__(self, udid=None, x8_enabled=True):
        caps = {
            "platformName": "android",
            "appium:ensureWebviewHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 3600,
            "appium:connectHardwareKeyboard": True
        }
        if udid is None:
            udid = "any"
        else:
            caps["appium:udid"] = udid
        self.driver = webdriver.Remote(
            "http://localhost:4723/wd/hub", caps)
        self.screenshot_path = f"./screenshot_{udid}.png"
        self.driver.save_screenshot(self.screenshot_path)
        self.img_bgr = cv2.imread(self.screenshot_path, 1)
        # 高さと幅の倍率
        self.mag = self.img_bgr.shape[0] / 1920, self.img_bgr.shape[1] / 1080
        self.actions = ActionChains(self.driver)
        self.actions.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
        # HDかどうか
        self.is_hd = self.img_bgr.shape[0] == 1280
        self.move_tap_height = int(800 * self.mag[0])
        # print(self.mag, self.is_hd, self.move_tap_height)

        # x8無効の場合の時間を定義
        self.tap_intarval = 0.2
        self.retry_intarval = 3.0
        self.pooling_intarval = 1.0
        # そもそもx8が使えない端末
        if udid == "482707805697":
            print("x8が使えない端末")
            x8_enabled = False
        # x8が有効かどうかでタップ間隔を変える
        if x8_enabled:
            self.tap_intarval = 0.05
            self.retry_intarval = 0.5
            self.pooling_intarval = 0.1
        print(
            f"Connected to {udid},  res{self.img_bgr.shape[:2]}, x8({x8_enabled})")

        self.img_gray = None
        self.img_bgr = None

    def update_image(self):
        """
        スクショを撮って変数更新
        """
        self.driver.save_screenshot(self.screenshot_path)
        self.img_bgr = cv2.imread(self.screenshot_path, 1)
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        # x8 speederが無効化された場合
        # deviceが勝手に判断
        if self.is_off_x8(mag=self.mag[0]):
            print("x8 speederを適用")
            self.tap((1032, 1857))
            sleep(0.5)
            self.tap((726, 1171))
            sleep(5)
            # 再帰呼出し
            self.update_image()

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
        return res.max() >= 0.95

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
        return res.max() >= 0.95

    def get_height(self) -> float | None:
        """
        Get height
        """
        if self.is_hd:
            cropped_img_bgr = self.img_bgr[39:85]
            fnamer_format = "src/height{}_HD.png".format
        else:
            cropped_img_bgr = self.img_bgr[60:130]
            fnamer_format = "src/height{}.png".format
        dict_digits = {}
        for i in list(range(10))+["dot"]:
            template = cv2.imread(fnamer_format(i), 1)
            res = cv2.matchTemplate(
                cropped_img_bgr, template, cv2.TM_CCOEFF_NORMED)
            # まだ調整が必要かもしれない
            loc = np.where(res >= 0.98)
            for loc_y in loc[1]:
                dict_digits[loc_y] = i
        height = ""
        for _, key in sorted(dict_digits.items()):
            if key == "dot":
                height += "."
            else:
                height += str(key)
        # 例外処理で対応
        try:
            height = float(height)
        except ValueError:
            height = None
        return height

    def get_animal_count(self) -> int | None:
        """
        動物の数を取得
        引数にはカラー画像を与える!!
        """
        if self.is_hd:
            cropped_img_bgr = self.img_bgr[175:225]
            fnamer_format = "src/count{:d}_HD_shadow.png".format
        else:
            cropped_img_bgr = self.img_bgr[264:328]
            fnamer_format = "src/count{:d}_shadow.png".format
        # 青や白を排除
        img_shadow = cv2.inRange(
            cropped_img_bgr, BACKGROUND_COLOR_DARK, WHITE)
        dict_digits = {}
        for i in range(10):
            template = cv2.imread(fnamer_format(i), 0)
            res = cv2.matchTemplate(
                img_shadow, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.97)
            for y in loc[1]:
                dict_digits[y] = i
        if dict_digits:
            animal_num = int("".join([str(i)
                                      for _, i in sorted(dict_digits.items())]))
        else:
            animal_num = None
        return animal_num

    def tap(self, coordinates: tuple) -> None:
        """
        Tap
        """
        loc_x = coordinates[0] * self.mag[0]
        loc_y = coordinates[1] * self.mag[1]
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
                500 * self.mag[1], 1800 * self.mag[0])
            for _ in range(action[0]):
                self.actions.w3c_actions.pointer_action.click()
                # 試した感じ0.05がバグらない最低値
                # x8なしはわからない
                self.actions.w3c_actions.pointer_action.pause(
                    self.tap_intarval)
            # 重要
            self.actions.w3c_actions.perform()
        # print(a[1] * self.mag[1], self.move_tap_height)
        # 座標タップ
        self.actions.w3c_actions.pointer_action.move_to_location(
            action[1] * self.mag[1], self.move_tap_height)
        self.actions.w3c_actions.pointer_action.click()
        # 適用
        self.actions.w3c_actions.perform()

    def to_training_image(self) -> np.ndarray:
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
            cv2.resize(self.img_bgr, dsize=TRAINNING_IMAGE_SIZE[::-1]),
            BACKGROUND_COLOR_DARK, WHITE))
