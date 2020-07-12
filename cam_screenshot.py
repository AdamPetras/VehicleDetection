from selenium import webdriver
from PIL import Image
import glob
import os
import time
from selenium.webdriver.support import expected_conditions as EC
IMAGE_PATH ="Images/"

# https://www.insecam.org/en/view/431570/
# https://www.insecam.org/en/view/831786/
# https://www.insecam.org/en/view/807253/
# https://www.insecam.org/en/view/422846/
# https://www.insecam.org/en/view/271799/
# https://www.insecam.org/en/view/513555/
# https://www.insecam.org/en/view/833391/
# https://www.insecam.org/en/bytag/Traffic/?page=33
#

def open_driver():
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_argument('window-size=1920x1080')
    chromeOptions.add_argument('disable-extensions')
    chromeOptions.add_argument("--headless")
    return webdriver.Chrome(options=chromeOptions)

def get_and_save_image(driver):
    driver.get('https://www.insecam.org/en/view/833391/')
    element = driver.find_element_by_id('image0')
    driver.execute_script("arguments[0].scrollIntoView();", element)
    location = element.location_once_scrolled_into_view
    size = element.size
    driver.save_screenshot("pageImage.png")
    crop_and_save_image(location,size)

def crop_and_save_image(location, size):
    x = location['x']
    y = location['y']
    width = location['x']+size['width']
    height = location['y']+size['height']
    im = Image.open("pageImage.png")
    im = im.convert('RGB')
    im = im.crop((int(x), int(y), int(width), int(height)))
    im.save(create_image_path_and_name())
def create_image_path_and_name():
    list_of_files = glob.glob(IMAGE_PATH+"*")
    if len(list_of_files) == 0:
        return IMAGE_PATH+"00001.jpg"
    latest_file = max(list_of_files, key=os.path.getctime)
    number = int(latest_file.split("/")[1].split(".")[0])
    number += 1
    number = str(number).zfill(5)
    return IMAGE_PATH+number+".jpg"

driver = open_driver()
while 1:
    get_and_save_image(driver)
    time.sleep(60)
driver.close()
