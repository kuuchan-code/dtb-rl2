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
TMPLATE_MATCHING_THRESHOLD = 0.99
TRAIN_SIZE = 64, 32  # 適当
NUM_OF_DELIMITERS = 36
RESET = {"coordinates": (200, 1755), "waittime_after": 5}
ROTATE = {"coordinates": (500, 1800), "waittime_after": 0.005}
WAITTIME_AFTER_ROTATE30 = 0.005
WAITTIME_AFTER_DROP = 4
WAITLOOP_TO_NEW_STATUS = 6
POLLONG_INTERVAL = 1
BACKGOUND_BGR = np.array([251, 208,  49])


def is_result_screen(img_gray):
    """
    Check the back button to determine game end.
    """
    template = cv2.imread("src/back.png", 0)
    res = cv2.matchTemplate(
        img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= TMPLATE_MATCHING_THRESHOLD)
    return len(loc[0]) > 0


def get_height(img_gray):
    """
    Get height
    """
    img_gray_height = img_gray[65:129, :]
    dict_digits = {}
    for i in list(range(10))+["dot"]:
        template = cv2.imread(f"images/{i}.png", 0)
        res = cv2.matchTemplate(
            img_gray_height, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= TMPLATE_MATCHING_THRESHOLD)
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


def image_binarization(img_bgr):
    img_mask = cv2.inRange(img_bgr, BACKGOUND_BGR, BACKGOUND_BGR)
    img_masked_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=img_mask)
    img_gray = cv2.cvtColor(img_masked_bgr, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return img_bin


class AnimalTower(gym.Env):
    """
    Small base for the Animal Tower, action is 12 turns gym environment
    """

    def __init__(self):
        print("Initializing...", end=" ", flush=True)
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *TRAIN_SIZE), dtype=np.uint8)
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
        img_gray_resized = cv2.resize(img_gray, dsize=TRAIN_SIZE)
        obs = img_gray_resized
        # Returns obs after start
        print("Done")
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return np.reshape(obs, (1, *TRAIN_SIZE))

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
            img_gray_resized = cv2.resize(img_gray, dsize=TRAIN_SIZE)
            obs = img_gray_resized
            if is_result_screen(img_gray):
                print("Game over")
                print("return observation, 0, True, {}")
                print("-"*NUM_OF_DELIMITERS)
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAIN_SIZE)), 0, True, {}
            if height and height > self.prev_height:
                print(f"Height update: {height}m")
                print("return obs, 1, False, {}")
                print("-"*NUM_OF_DELIMITERS)
                self.prev_height = height
                cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
                return np.reshape(obs, (1, *TRAIN_SIZE)), 1, False, {}
            sleep(POLLONG_INTERVAL)
        print("No height update")
        print("return obs, 1, False, {}")
        print("-"*NUM_OF_DELIMITERS)
        cv2.imwrite(OBSERVATION_IMAGE_PATH, obs)
        return np.reshape(obs, (1, *TRAIN_SIZE)), 1, False, {}

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
