"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from time import sleep
import gym
import numpy as np
from cv2 import cv2
from appium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction


SCREENSHOT_PATH = "./screenshot.png"
OBSERVATION_IMAGE_PATH = "./observation.png"
HEIGHT_TEMPLATE_MATCHING_THRESHOLD = 0.99
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.95
TRAIN_IMAGE_SIZE = 32, 64  # 適当（横、縦）
NUM_OF_DELIMITERS = 36
RESET = {"coordinates": (200, 1755), "waittime_after": 5}
ROTATE = {"coordinates": (500, 1800), "waittime_after": 0.005}
WAITTIME_AFTER_ROTATE30 = 0.005
WAITTIME_AFTER_DROP = 4
WAITLOOP_TO_NEW_STATUS = 6
POLLONG_INTERVAL = 1
# 背景色 (bgr)
BACKGROUND_COLOR = np.array([251, 208, 49], dtype=np.uint8)
BACKGROUND_COLOR_LIGHT = BACKGROUND_COLOR + 4
BACKGROUND_COLOR_DARK = BACKGROUND_COLOR - 4
BLACK = np.zeros(3, dtype=np.uint8)
WHITE = BLACK + 255
WHITE_DARK = WHITE - 15


def is_result_screen(img_gray):
    """
    Check the back button to determine game end.
    """
    template = cv2.imread("src/back.png", 0)
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= HEIGHT_TEMPLATE_MATCHING_THRESHOLD)
    return len(loc[0]) > 0


def get_height(img_gray):
    """
    Get height
    """
    img_gray_height = img_gray[65:129, :]
    dict_digits = {}
    for i in list(range(10))+["dot"]:
        template = cv2.imread(f"src/height{i}.png", 0)
        res = cv2.matchTemplate(
            img_gray_height, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= HEIGHT_TEMPLATE_MATCHING_THRESHOLD)
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    height = ""
    for key in sorted(dict_digits.items()):
        if key[1] == "dot":
            height += "."
        else:
            height += str(key[1])
    if not height:
        height = 0
    return float(height)


def get_animal_num(img_bgr: np.ndarray) -> int:
    """
    動物の数を取得
    引数にはカラー画像を与える!!
    """
    img_shadow = cv2.inRange(
        img_bgr[264:328], BACKGROUND_COLOR_DARK, WHITE)
    dict_digits = {}
    for i in list(range(10)):
        template = cv2.imread(f"src/count{i}_shadow.png", 0)
        res = cv2.matchTemplate(
            img_shadow, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD)
        # print(loc, i)
        for y in loc[1]:
            dict_digits[y] = i
    animal_num = ""
    for key in sorted(dict_digits.items()):
        animal_num += str(key[1])
    if not animal_num:
        animal_num = 0
    return int(animal_num)


def cropping_to_train_image_size(img_bin):
    return img_bin[:1600, 295:785]


def image_binarization(img_bgr):
    """
    Binarize background and non-background
    """
    img_bin = cv2.bitwise_not(cv2.inRange(
        img_bgr, BACKGROUND_COLOR_DARK, WHITE))
    return img_bin


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self):
        print("Initializing...", end=" ", flush=True)
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAIN_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [0, 1]
        self.prev_height = 0
        caps = {}
        caps["platformName"] = "android"
        caps["appium:ensureWebviewsHavePages"] = True
        caps["appium:nativeWebScreenshot"] = True
        caps["appium:newCommandTimeout"] = 3600
        caps["appium:connectHardwareKeyboard"] = True
        self.driver = webdriver.Remote(
            "http://localhost:4723/wd/hub", caps)
        self.operations = ActionChains(self.driver)
        self.operations.w3c_actions = ActionBuilder(
            self.driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
        print("Done")
        print("-"*NUM_OF_DELIMITERS)

    def reset(self):
        print("Resetting...", end=" ", flush=True)
        self.prev_height = 0
        # Tap the Reset button
        self._tap(RESET["coordinates"], RESET["waittime_after"])
        self.driver.save_screenshot(SCREENSHOT_PATH)
        img_gray = cv2.imread(SCREENSHOT_PATH, 0)
        img_gray_resized = cv2.resize(img_gray, dsize=TRAIN_IMAGE_SIZE)
        obs = img_gray_resized
        # Returns obs after start
        print("Done")
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return np.reshape(obs, (1, *TRAIN_IMAGE_SIZE))

    def step(self, action):
        # Perform Action
        print(f"Action({action:.0f})")
        for _ in range(int(action)):
            self._tap(ROTATE["coordinates"], ROTATE["waittime_after"])
        sleep(WAITTIME_AFTER_ROTATE30)
        self._tap((action[1], 800), WAITTIME_AFTER_DROP)
        # Generate obs and reward, done flag, and return
        for _ in range(WAITLOOP_TO_NEW_STATUS):
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_gray = cv2.imread(SCREENSHOT_PATH, 0)
            height = get_height(img_gray)
            img_gray_resized = cv2.resize(img_gray, dsize=TRAIN_IMAGE_SIZE)
            obs = img_gray_resized
            if is_result_screen(img_gray):
                print("Game over")
                print("return observation, 0, True, {}")
                print("-"*NUM_OF_DELIMITERS)
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAIN_IMAGE_SIZE)), 0, True, {}
            if height and height > self.prev_height:
                print(f"Height update: {height}m")
                print("return obs, 1, False, {}")
                print("-"*NUM_OF_DELIMITERS)
                self.prev_height = height
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAIN_IMAGE_SIZE)), 1, False, {}
            sleep(POLLONG_INTERVAL)
        print("No height update")
        print("return obs, 1, False, {}")
        print("-"*NUM_OF_DELIMITERS)
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return np.reshape(obs, (1, *TRAIN_IMAGE_SIZE)), 1, False, {}

    def render(self):
        pass

    def _tap(self, coordinates, waittime):
        """
        Tap
        """
        while True:
            self.operations.w3c_actions.pointer_action.move_to_location(
                coordinates[0], coordinates[1])
            self.operations.w3c_actions.pointer_action.pointer_down()
            self.operations.w3c_actions.pointer_action.pause(0.1)
            self.operations.w3c_actions.pointer_action.release()
            self.operations.perform()
            sleep(waittime)
            break
