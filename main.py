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
        전체 프레임에서 플레이리스트 영역을 고정된 ROI로 추출.
        (화면 녹화된 플레이리스트는 보통 화면 하단 중간~하단 부분에 위치한다고 가정)

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 추출된 관심 영역(ROI)
        """
        h, w, _ = frame.shape

        # 예시: 전체 높이의 55%~90% 영역을 플레이리스트 영역으로 가정
        roi = frame[int(h * 0.55): int(h * 0.9), :]

        # (필요시 추가적인 수동 크롭이나 사용자 입력을 고려할 수 있습니다.)
        return roi

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        OCR을 위한 프레임 전처리: ROI 추출, 해상도 업스케일, 그레이스케일, 어댑티브 이진화, 노이즈 제거

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 전처리된 프레임(ROI)
        """
        # 1. 관심 영역(플레이리스트 영역) 추출
        roi = self.extract_playlist_roi(frame)

        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. 해상도 업스케일 (OCR 인식률 향상을 위해)
        scale_percent = 200  # 200% 확대
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # 4. 어댑티브 이진화 적용
        thresh = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 5. 노이즈 제거 (필요에 따라 파라미터 조정)
        denoised = cv2.fastNlMeansDenoising(thresh, h=30, templateWindowSize=7, searchWindowSize=21)

        return denoised

    def extract_frames(self, interval: int = 30) -> None:
        """
        비디오에서 프레임 추출

        Args:
            interval (int): 프레임 추출 간격
        """
        if not self.video_path:
            raise ValueError("비디오 파일이 선택되지 않았습니다.")

        # 프레임 저장 디렉토리 생성
        frames_dir = self.create_frames_directory()

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"총 {total_frames}개의 프레임 중 {interval} 프레임 간격으로 추출 중...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # 프레임 버퍼에 추가
                self.frame_buffer.append(frame)

                # 프레임 이미지 저장
                frame_path = frames_dir / f"frame_{frame_count:06d}.png"
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
        텍스트에서 곡 정보를 파싱. 재생 아이콘 등 잘못 인식된 짧은 문자열(D, > 등)은 필터링.

        Args:
            text (str): OCR로 추출된 텍스트

        Returns:
            List[Dict[str, str]]: 파싱된 곡 정보 리스트
        """
        songs = []
        # (옵션) '전체 선택' 이후 텍스트만 사용: 인식 결과에 포함되어 있다면 제거
        text = self.find_start_position(text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # 재생 로고 등 잘못 인식된 단일 문자나 짧은 특수문자 필터링
        filtered_lines = []
        for line in lines:
            # 일반적으로 곡 제목이나 아티스트는 2자 이상이고, 숫자 및 알파벳 혹은 한글을 포함
            if len(line) <= 1:
                continue
            # 대표적으로 잘못 인식되는 문자들: D, >, <, ▶ 등
            if line in {"D", ">", "<", "▶", "❯", ""}:
                continue
            filtered_lines.append(line)

        # 두 줄씩 묶어서 제목과 아티스트로 가정
        i = 0
        while i < len(filtered_lines) - 1:
            title = filtered_lines[i]
            artist = filtered_lines[i + 1]
            # 양쪽 모두 유효하면 결과에 추가
            if title and artist:
                songs.append({
                    'title': title,
                    'artist': artist
                })
            i += 2

        return songs

    def perform_ocr(self) -> List[Dict]:
        """
        모든 프레임에 대해 OCR 수행

        Returns:
            List[Dict]: OCR 결과 및 곡 정보
        """
        self.logger.info("OCR 처리를 시작합니다...")

        for idx, frame in enumerate(self.frame_buffer):
            # 프레임 전처리 (ROI 추출 및 이미지 개선)
            processed_frame = self.preprocess_frame(frame)
            pil_image = Image.fromarray(processed_frame)

            # OCR 수행 (필요에 따라 config 파라미터 추가 가능)
            text = pytesseract.image_to_string(pil_image, lang='kor+eng')
            self.logger.debug(f"프레임 {idx} OCR 결과: {text}")

            # 곡 정보 파싱
            songs = self.parse_songs(text)

            # 결과 저장 (프레임 번호 계산 시 interval 고려)
            frame_number = idx * 30  # interval을 30으로 가정
            self.results.append({
                'frame_number': frame_number,
                'frame_path': str(self.frames_dir / f"frame_{frame_number:06d}.png"),
                'songs': songs
            })

            self.logger.info(f"프레임 {idx + 1}/{len(self.frame_buffer)} 처리 완료")

        return self.results

    def export_to_csv(self) -> str:
        """
        파싱된 곡 정보를 CSV 파일로 저장

        Returns:
            str: 저장된 CSV 파일 경로
        """
        if not self.results:
            raise ValueError("추출된 결과가 없습니다.")

        # 모든 곡 정보 수집
        all_songs = []
        for result in self.results:
            for song in result['songs']:
                all_songs.append({
                    'frame': result['frame_number'],
                    'frame_path': result['frame_path'],
                    'title': song['title'],
                    'artist': song['artist']
                })

        # DataFrame 생성 및 중복 제거
        df = pd.DataFrame(all_songs)
        df = df.drop_duplicates(subset=['title', 'artist'])

        # 파일 저장
        video_path = Path(self.video_path)
        output_path = video_path.parent / f"{video_path.stem}_playlist.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        self.logger.info(f"결과가 저장되었습니다: {output_path}")
        return str(output_path)


def main():
    # OCR 인스턴스 생성
    ocr = PlaylistOCR()

    try:
        # 비디오 파일 선택
        if not ocr.select_video():
            print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
            return

        # 프레임 추출
        print("프레임 추출을 시작합니다...")
        ocr.extract_frames(interval=30)

        # OCR 수행 및 결과 저장
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