import cv2
import numpy as np
import pyautogui
import time

class PopupScanner:
    def __init__(self, cooldown=1.5):
        self.cooldown = cooldown
        self.last_scan = 0
        
    def capture_screen(self):
        if time.time() - self.last_scan < self.cooldown:
            return None
            
        screenshot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.last_scan = time.time()
        return img
    
    def detect_popups(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        popups = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 200 < w < 800 and 100 < h < 600:
                popups.append((x, y, w, h))
        return popups