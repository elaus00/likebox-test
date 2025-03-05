import cv2

video_path = 'KakaoTalk_20250219_175712675.mp4'  # 확인하려는 영상 파일 경로
cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

cap.release()

print(f"해상도: {width} x {height}")
print(f"FPS: {fps}")
print(f"총 프레임 수: {frame_count}")
