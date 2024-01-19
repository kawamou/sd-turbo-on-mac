import time

import cv2
import numpy as np
from PIL import Image
from sd_turbo import SdTurbo

prev_time = time.time()

prompt = "a photo of Elon Musk"

negative_prompt = ""

cap = cv2.VideoCapture(0)

img_dst = Image.new("RGB", (1024, 512))

sd_turbo = SdTurbo("stabilityai/sd-turbo")

while True:
    ret, frame = cap.read()

    width, height = frame.shape[1], frame.shape[0]
    left = (width - 1024) // 2
    top = (height - 1024) // 2
    right = (width + 1024) // 2
    bottom = (height + 1024) // 2

    img_init = Image.fromarray(frame).crop((left, top, right, bottom)).resize((512, 512), Image.NEAREST)

    result = sd_turbo.run(prompt, negative_prompt, img_init)

    if isinstance(result, Image.Image):
        img_dst.paste(img_init, (0, 0))
        img_dst.paste(result, (512, 0))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.imshow("{} fps".format(str(fps)), np.array(img_dst))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
