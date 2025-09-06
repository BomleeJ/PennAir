import numpy as np
import cv2 as cv
from src.CountourDrawer import add_contours_to_frame
import logging

logging.basicConfig(level=logging.INFO)

def analyze_image(result_name: str, image_path: str = "files/Grassy.png"):
    result_path = f"results/{result_name}.png"
    img = cv.imread(cv.samples.findFile(f"{image_path}"))
    processedImage = add_contours_to_frame(img, False)
    cv.imwrite(result_path, processedImage)

def analyze_video(result_name: str, video_path: str, with_depth=False):
    result_path = f"results/{result_name}.avi"
    cap = cv.VideoCapture(video_path)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    FPS = 20.0
    WIDTH = 1920
    HEIGHT = 1080

    out = cv.VideoWriter(result_path, fourcc, FPS, (WIDTH, HEIGHT), isColor=True)
    val = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        logging.info(f"Processing Frame {val} from video {video_path}")
        val += 1
        processed_frame = add_contours_to_frame(frame, with_depth=with_depth)
        out.write(processed_frame)

    cap.release()
    out.release()

def main():
    Phase2_video_path = "files/Phase2.mp4"
    Phase3_video_path = "files/Phase3.mp4"
    Phase4_video_path = "files/Phase3.mp4"
    
    analyze_image("Grassy_with_contours")
    logging.info("Image Complete!")
    
    analyze_video_2d("Phase2_with_contours2.avi", Phase2_video_path)
    logging.info("Video1 Complete!")
    
    analyze_video_2d("Phase3_with_contours2.avi", Phase3_video_path)
    logging.info("Video2 Complete!")
    
    
    analyze_video("Phase4_with_contours2.avi", Phase4_video_path, with_depth=True)
    logging.info("Video3 Complete!")
    

if __name__ == "__main__":
    main()

