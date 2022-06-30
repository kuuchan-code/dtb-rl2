"""
Deep reinforcement learning on the small base of the Animal Tower.
"""
from time import sleep
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
ANIMAL_COUNT_TEMPLATE_MATCHING_THRESHOLD = 0.95
TRAINNING_IMAGE_SIZE = 256, 75  # 適当（縦、横）
NUM_OF_DELIMITERS = 36
RESET = {"coordinates": (200, 1755), "waittime_after": 3}
ROTATE30 = {"coordinates": (500, 1800), "waittime_after": 0.0001}
WAITTIME_AFTER_ROTATE = 0.1
WAITTIME_AFTER_DROP = 4
POLLING_INTERVAL = 0.5
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
    loc = np.where(res >= TEMPLATE_MATCHING_THRESHOLD)
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
        loc = np.where(res >= TEMPLATE_MATCHING_THRESHOLD)
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


def get_animal_count(img_bgr: np.ndarray) -> int:
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
        for loc_y in loc[1]:
            dict_digits[loc_y] = i
    animal_num = ""
    for key in sorted(dict_digits.items()):
        animal_num += str(key[1])
    if not animal_num:
        animal_num = 0
    return int(animal_num)


def to_training_image(img_bgr):
    """
    入力BGR画像を訓練用画像にする
    """
    img_bin = cv2.bitwise_not(cv2.inRange(
        img_bgr, BACKGROUND_COLOR_DARK, WHITE))
    cropped_img_bin = img_bin[:1665, 295:785]
    resized_and_cropped_img_bin = cv2.resize(
        cropped_img_bin, dsize=TRAINNING_IMAGE_SIZE[::-1])
    return resized_and_cropped_img_bin


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self):
        print("Initializing...", end=" ", flush=True)
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAINNING_IMAGE_SIZE), dtype=np.uint8)
        self.reward_range = [0, 27.79]
        self.prev_height = 0
        self.prev_animal_count = 0
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
        self.prev_animal_count = 0
        self._tap(RESET["coordinates"], RESET["waittime_after"])
        self.driver.save_screenshot(SCREENSHOT_PATH)
        img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
        obs = to_training_image(img_bgr)
        print("Done")
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE))

    def step(self, action):
        print(f"Action({action:.0f})")
        for _ in range(int(action)):
            self._tap(ROTATE30["coordinates"], ROTATE30["waittime_after"])
        sleep(WAITTIME_AFTER_ROTATE)
        self._tap((540, 800), WAITTIME_AFTER_DROP)
        self.driver.save_screenshot(SCREENSHOT_PATH)
        img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
        obs = to_training_image(img_bgr)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        height = get_height(img_gray)
        while True:
            self.driver.save_screenshot(SCREENSHOT_PATH)
            img_bgr = cv2.imread(SCREENSHOT_PATH, 1)
            obs = to_training_image(img_bgr)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            height = get_height(img_gray)
            if is_result_screen(img_gray):
                print("Game over")
                print("return obs, 0, True, {}")
                print("-"*NUM_OF_DELIMITERS)
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE)), 0, True, {}
            if height and height > self.prev_height:
                print(f"Height update: {height}m")
                print(f"return obs, {height}, False, {{}}")
                print("-"*NUM_OF_DELIMITERS)
                self.prev_height = height
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE)), height, False, {}
            animal_count = get_animal_count(img_bgr)
            if animal_count and animal_count > self.prev_animal_count:
                self.prev_animal_count = animal_count
                print("No height update")
                print(f"return obs, {height}, False, {{}}")
                print("-"*NUM_OF_DELIMITERS)
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAINNING_IMAGE_SIZE)), 1, False, {}
            sleep(POLLING_INTERVAL)

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
            self.operations.w3c_actions.pointer_action.pause(0.0001)
            self.operations.w3c_actions.pointer_action.release()
            self.operations.perform()
            sleep(waittime)
            break
