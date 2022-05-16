import cv2 as cv
from textblockdetector import dispatch as dispatch_ctd_detection
import asyncio

async def get_mask():
    img = cv.imread("c_8_muto.jpg")

    mask, final_mask, textlines = await dispatch_ctd_detection(img, False)

    cv.imshow("mask",mask)
    cv.imshow("final_mask", final_mask)
    cv.waitKey(0)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_mask())
