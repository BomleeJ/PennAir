from main import analyze_video, analyze_image
from concurrent.futures import ProcessPoolExecutor


def main():
    with ProcessPoolExecutor() as executor:
        executor.submit(analyze_image, "Grassy_with_contours")
        executor.submit(analyze_video, "Phase2_with_contours2.avi", "files/Phase2.mp4")
        executor.submit(analyze_video, "Phase3_with_contours2.avi", "files/Phase3.mp4")
        executor.submit(analyze_video, "Phase4_with_contours2.avi", "files/Phase3.mp4", with_depth=True)

if __name__ == "__main__":
    main()