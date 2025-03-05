import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Dict
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
import logging

class PlaylistOCR:
    def __init__(self, video_path: str = None):
        """
        OCR 파이프라인 초기화

        Args:
            video_path (str, optional): 비디오 파일 경로
        """
        self.video_path = video_path
        self.frame_buffer = []
        self.results = []
        self.frames_dir = None

        # 사용자 사전 파일 경로
        self.user_words_path = "musicbrainz_words.txt"

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def select_video(self) -> bool:
        """
        비디오 파일 선택 다이얼로그 열기

        Returns:
            bool: 파일이 선택되었으면 True, 취소되었으면 False
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="비디오 파일을 선택하세요",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video_path = file_path
            return True
        return False

    def create_frames_directory(self) -> Path:
        """
        프레임 이미지 저장을 위한 디렉토리 생성

        Returns:
            Path: 생성된 디렉토리 경로
        """
        video_path = Path(self.video_path)
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        self.frames_dir = frames_dir
        return frames_dir

    def extract_playlist_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        전체 프레임에서 플레이리스트 영역(화면 하단의 일정 영역)을 고정된 ROI로 추출.
        (전체 높이의 48% ~ 80% 영역을 사용)

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 추출된 관심 영역(ROI)
        """
        h, w, _ = frame.shape
        roi = frame[int(h*0.48):int(h*0.8),:]
        return roi

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        OCR을 위한 프레임 전처리: ROI 추출, 그레이스케일 변환, 히스토그램 평활화

        해상도 스케일링, 노이즈 제거, 어댑티브 이진화는 생략합니다.

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 전처리된 이미지 (ROI, 그레이스케일, 대비 개선)
        """
        # 1. ROI 추출 (플레이리스트 영역)
        roi = self.extract_playlist_roi(frame)
        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 3. 히스토그램 평활화로 대비 개선
        equalized = cv2.equalizeHist(gray)
        return equalized

    def split_image_into_rows(self, image: np.ndarray, row_height: int = 170) -> List[str]:
        """
        전처리된 ROI 이미지를 행 단위로 분할하여 각 행별 OCR을 수행합니다.
        (각 행의 높이는 고정하여 170 픽셀로 설정)

        Args:
            image (np.ndarray): 전처리된 ROI 이미지 (그레이스케일)
            row_height (int): 한 행의 높이 (픽셀 단위)

        Returns:
            List[str]: 각 행에서 추출된 텍스트 리스트
        """
        rows_text = []
        h, w = image.shape
        num_full_rows = h // row_height

        # 사용자 사전 옵션 문자열 생성
        custom_config = f'--user-words "{self.user_words_path}"'

        # 전체 행을 순서대로 처리 (위→아래)
        for i in range(num_full_rows):
            y_start = i * row_height
            y_end = y_start + row_height
            row_img = image[y_start:y_end, :]
            # 각 행에 대해 OCR 수행
            pil_row = Image.fromarray(row_img)
            # 사용자 사전 옵션을 config 파라미터에 추가가
            row_text = pytesseract.image_to_string(pil_row, lang='kor+eng', config=custom_config).strip()
            if row_text:  # 빈 행은 제외
                rows_text.append(row_text)
                self.logger.debug(f"행 {i} OCR 결과: {row_text}")

        # 만약 남은 부분이 충분 크다면(예: 전체 행 높이의 절반 이상) 처리
        remainder = h % row_height
        if remainder >= row_height // 2:
            y_start = num_full_rows * row_height
            row_img = image[y_start:, :]
            pil_row = Image.fromarray(row_img)
            row_text = pytesseract.image_to_string(pil_row, lang='kor+eng', config=custom_config).strip()
            if row_text:
                rows_text.append(row_text)
                self.logger.debug(f"남은 행 OCR 결과: {row_text}")

        return rows_text

    def extract_frames(self, interval: int = 10) -> None:
        """
        비디오에서 프레임 추출

        Args:
            interval (int): 프레임 추출 간격 (더 자주 추출)
        """
        if not self.video_path:
            raise ValueError("비디오 파일이 선택되지 않았습니다.")

        self.create_frames_directory()

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"총 {total_frames} 프레임 중 {interval} 프레임 간격으로 추출 중...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                self.frame_buffer.append(frame)
                frame_path = self.frames_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                self.logger.info(f"프레임 추출 및 저장: {frame_count}/{total_frames}")
            frame_count += 1

        cap.release()
        self.logger.info(f"총 {len(self.frame_buffer)}개의 프레임이 추출되었습니다.")

    def find_start_position(self, text: str) -> str:
        """
        '전체 선택' 이후의 텍스트만 추출하려 시도 (있을 경우).
        없으면 원본 텍스트 그대로 반환.

        Args:
            text (str): OCR로 추출된 전체 텍스트

        Returns:
            str: '전체 선택' 이후의 텍스트 또는 원본 텍스트
        """
        start_marker = "전체 선택"
        if start_marker in text:
            try:
                _, text = text.split(start_marker, 1)
                return text.strip()
            except ValueError:
                return text.strip()
        return text.strip()

    def parse_songs(self, text: str) -> List[Dict[str, str]]:
        """
        텍스트에서 곡 정보를 파싱합니다.
        (잘못 인식된 단일 문자나 특수 기호는 필터링)

        Args:
            text (str): OCR로 추출된 텍스트

        Returns:
            List[Dict[str, str]]: 파싱된 곡 정보 리스트
        """
        songs = []
        text = self.find_start_position(text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # 잘못 인식될 가능성이 있는 짧은 문자열 필터링
        filtered_lines = [line for line in lines if len(line) > 1 and line not in {"D", ">", "<", "▶", "❯", ""}]

        # 두 줄씩 묶어서 제목과 아티스트로 가정
        i = 0
        while i < len(filtered_lines) - 1:
            title = filtered_lines[i]
            artist = filtered_lines[i + 1]
            songs.append({
                'title': title,
                'artist': artist
            })
            i += 2

        return songs

    def perform_ocr(self) -> List[Dict]:
        """
        각 프레임에 대해 전처리된 ROI를 행 단위(행 높이 170)로 분할하여 각 행별 OCR을 수행한 후,
        읽기 순서(위→아래)로 결합하여 곡 정보를 파싱합니다.

        Returns:
            List[Dict]: 프레임별 OCR 결과 및 곡 정보
        """
        self.logger.info("OCR 처리를 시작합니다...")
        row_height = 170

        for idx, frame in enumerate(self.frame_buffer):
            # 전처리: ROI 추출, 그레이스케일 변환, 대비 개선
            preprocessed_roi = self.preprocess_frame(frame)
            # ROI를 행 단위로 분할하여 각 행별 OCR 수행
            row_texts = self.split_image_into_rows(preprocessed_roi, row_height=row_height)
            combined_text = "\n".join(row_texts)
            self.logger.debug(f"프레임 {idx} 결합된 텍스트:\n{combined_text}")

            # 곡 정보 파싱
            songs = self.parse_songs(combined_text)

            # 결과 저장 (추출 시 프레임 간격은 interval=10으로 가정)
            frame_number = idx * 10
            self.results.append({
                'frame_number': frame_number,
                'frame_path': str(self.frames_dir / f"frame_{frame_number:06d}.png"),
                'songs': songs
            })
            self.logger.info(f"프레임 {idx + 1}/{len(self.frame_buffer)} 처리 완료")
        return self.results

    def export_to_csv(self) -> str:
        """
        파싱된 곡 정보를 CSV 파일로 저장합니다.

        Returns:
            str: 저장된 CSV 파일 경로
        """
        if not self.results:
            raise ValueError("추출된 결과가 없습니다.")

        all_songs = []
        for result in self.results:
            for song in result['songs']:
                all_songs.append({
                    'frame': result['frame_number'],
                    'frame_path': result['frame_path'],
                    'title': song['title'],
                    'artist': song['artist']
                })

        df = pd.DataFrame(all_songs)
        df = df.drop_duplicates(subset=['title', 'artist'])
        video_path = Path(self.video_path)
        output_path = video_path.parent / f"{video_path.stem}_playlist.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"결과가 저장되었습니다: {output_path}")
        return str(output_path)


def main():
    ocr = PlaylistOCR()
    try:
        if not ocr.select_video():
            print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
            return

        print("프레임 추출을 시작합니다...")
        ocr.extract_frames(interval=10)  # 프레임 간격을 짧게 설정하여 더 많은 프레임 추출

        results = ocr.perform_ocr()
        output_path = ocr.export_to_csv()

        print("\nOCR 처리가 완료되었습니다!")
        print(f"결과가 다음 위치에 저장되었습니다: {output_path}")
        print(f"프레임 이미지가 다음 위치에 저장되었습니다: {ocr.frames_dir}")

    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        logging.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()