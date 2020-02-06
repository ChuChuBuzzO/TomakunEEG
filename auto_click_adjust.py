import pyautogui as pg
from time import sleep

target_file = "find_ok.png"

i = 1
while True:
    try:
        print("start")
        x, y = pg.locateCenterOnScreen('./{}'.format(target_file))
        pg.click(x, y)
        print("{} click".format(str(i)))
        i += 1

    except:
        sleep(2)