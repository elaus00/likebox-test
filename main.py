import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
import logging
from collections import Counter

# LayoutLMv3 관련 라이브러리 (transformers 설치 필요)
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch


class PlaylistOCR:
    def __init__(self, video_path: str = None, tesseract_config: str = None):
        """
        OCR 파이프라인 초기화 (LayoutLM v3를 통한 문서 내 엔티티 추출 포함)

        Args:
            video_path (str, optional): 비디오 파일 경로
            tesseract_config (str, optional): Tesseract OCR 설정
        """
        self.video_path = video_path
        self.frame_buffer = []
        self.results = []
        self.frames_dir = None
        # Tesseract OCR 설정 (인식률 향상을 위한 파라미터)
        self.tesseract_config = tesseract_config or '--oem 1 --psm 6 -l kor+eng'

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # LayoutLM v3 프로세서와 토큰 분류 모델 초기화
        # ※ 아래 모델 이름은 예시이며, 실제로는 해당 도메인에 맞게 fine-tuning된 모델을 사용해야 합니다.
        self.layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        # 예시 레이블 매핑 (0: O, 1: TITLE, 2: ARTIST)
        self.label_map = {0: "O", 1: "TITLE", 2: "ARTIST"}

        # 인식된 결과를 중복 확인하고 병합하기 위한 메모리
        self.song_memory = {}

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
        (녹화된 플레이리스트는 보통 화면 하단에 위치하므로, 전체 높이의 55% ~ 90% 영역을 사용)

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 추출된 관심 영역(ROI)
        """
        h, w, _ = frame.shape
        roi = frame[int(h * 0.55): int(h * 0.9), :]
        return roi

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        OCR을 위한 프레임 전처리: ROI 추출, 다양한 이미지 향상 기법 적용

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 1. ROI 추출 (플레이리스트 영역)
        roi = self.extract_playlist_roi(frame)

        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. 노이즈 제거 (Bilateral 필터 - 경계선은 보존하면서 노이즈 제거)
        denoised = cv2.bilateralFilter(gray, 5, 75, 75)

        # 4. 히스토그램 평활화로 대비 개선
        equalized = cv2.equalizeHist(denoised)

        # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        # 지역적 대비를 향상시켜 텍스트 인식률 개선
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(equalized)

        # 6. 샤프닝 (텍스트 경계 강화)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(clahe_applied, -1, kernel)

        return sharpened

    def apply_multiple_preprocessing(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        다양한 전처리 방법을 적용하여 OCR 인식률을 높입니다.
        각 전처리 방법으로 처리된 이미지를 리스트로 반환합니다.

        Args:
            frame (np.ndarray): 입력 프레임

        Returns:
            List[np.ndarray]: 다양하게 전처리된 이미지 리스트
        """
        roi = self.extract_playlist_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        preprocessed_images = []

        # 1. 기본 전처리 (그레이스케일 + 히스토그램 평활화)
        equalized = cv2.equalizeHist(gray)
        preprocessed_images.append(equalized)

        # 2. 노이즈 제거 + CLAHE
        denoised = cv2.bilateralFilter(gray, 5, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(denoised)
        preprocessed_images.append(clahe_applied)

        # 3. 샤프닝 적용
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(clahe_applied, -1, kernel)
        preprocessed_images.append(sharpened)

        # 4. 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)

        # 5. 적응형 이진화
        adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive_binary)

        return preprocessed_images

    def layoutlmv3_ocr_on_row(self, row_img: np.ndarray) -> Dict[str, str]:
        """
        한 행의 이미지에 대해 pytesseract를 이용해 단어와 바운딩 박스를 추출한 뒤,
        LayoutLM v3를 사용하여 각 단어의 엔티티(예, TITLE, ARTIST)를 예측합니다.

        Args:
            row_img (np.ndarray): 한 행의 이미지 (그레이스케일)

        Returns:
            Dict[str, str]: 추출된 텍스트 정보 (예, {"title": "...", "artist": "..."})
                            단어가 없거나 유의미한 정보가 없으면 빈 dict 반환.
        """
        # 다양한 전처리 이미지에 대해 OCR 수행 결과를 종합
        preprocessed_images = self.apply_multiple_preprocessing_for_row(row_img)
        all_ocr_results = []

        for img in preprocessed_images:
            # 그레이스케일 이미지를 RGB로 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb)

            # pytesseract로 단어 및 바운딩 박스 추출 (개선된 설정)
            data = pytesseract.image_to_data(pil_img, lang='kor+eng', config=self.tesseract_config,
                                             output_type=pytesseract.Output.DICT)

            words = []
            boxes = []
            confidences = []
            num_boxes = len(data["text"])

            for i in range(num_boxes):
                word = data["text"][i].strip()
                conf = int(data["conf"][i])

                # 신뢰도가 너무 낮은 결과 필터링 (30% 미만)
                if word != "" and conf > 30:
                    words.append(word)
                    left = data["left"][i]
                    top = data["top"][i]
                    width = data["width"][i]
                    height = data["height"][i]
                    boxes.append([left, top, left + width, top + height])
                    confidences.append(conf)

            if words:
                # LayoutLM v3 processor에 이미지, 단어, bounding boxes 전달
                encoding = self.layoutlm_processor(pil_img, words, boxes=boxes, return_tensors="pt", truncation=True)
                outputs = self.layoutlm_model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()

                if isinstance(predictions, int):
                    predictions = [predictions]

                predicted_labels = [self.label_map.get(pred, "O") for pred in predictions]

                # 각 단어에 대한 결과와 신뢰도 저장
                title_tokens = [(w, c) for w, label, c in zip(words, predicted_labels, confidences) if label == "TITLE"]
                artist_tokens = [(w, c) for w, label, c in zip(words, predicted_labels, confidences) if
                                 label == "ARTIST"]

                if title_tokens or artist_tokens:
                    # 신뢰도 가중치 적용하여 결과 추가
                    all_ocr_results.append({
                        "title_tokens": title_tokens,
                        "artist_tokens": artist_tokens
                    })

        # 결과 병합 및 최적의 결과 선택
        return self.merge_ocr_results(all_ocr_results)

    def apply_multiple_preprocessing_for_row(self, row_img: np.ndarray) -> List[np.ndarray]:
        """
        행 이미지에 대해 다양한 전처리 방법을 적용합니다.

        Args:
            row_img (np.ndarray): 행 이미지

        Returns:
            List[np.ndarray]: 전처리된 이미지 리스트
        """
        preprocessed_images = []

        # 1. 기본 그레이스케일 (입력이 이미 그레이스케일인 경우)
        if len(row_img.shape) == 3:
            gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = row_img.copy()
        preprocessed_images.append(gray)

        # 2. 히스토그램 평활화
        equalized = cv2.equalizeHist(gray)
        preprocessed_images.append(equalized)

        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(gray)
        preprocessed_images.append(clahe_applied)

        # 4. 노이즈 제거 + 샤프닝
        denoised = cv2.bilateralFilter(gray, 5, 75, 75)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        preprocessed_images.append(sharpened)

        # 5. 이진화 (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)

        return preprocessed_images

    def merge_ocr_results(self, results: List[Dict]) -> Dict[str, str]:
        """
        여러 이미지 처리 방법에서 얻은 OCR 결과를 병합하여 최적의 결과를 반환합니다.

        Args:
            results (List[Dict]): 다양한 전처리 방법으로 얻은 OCR 결과 목록

        Returns:
            Dict[str, str]: 병합된 최종 결과
        """
        if not results:
            return {}

        # 모든 타이틀 토큰과 아티스트 토큰 수집
        all_title_tokens = []
        all_artist_tokens = []

        for result in results:
            all_title_tokens.extend(result.get("title_tokens", []))
            all_artist_tokens.extend(result.get("artist_tokens", []))

        # 가장 많이 등장한 토큰과 신뢰도 높은 토큰을 기반으로 결과 선택
        title = self.select_best_tokens(all_title_tokens)
        artist = self.select_best_tokens(all_artist_tokens)

        return {"title": title, "artist": artist}

    def select_best_tokens(self, tokens: List[Tuple[str, int]]) -> str:
        """
        주어진 토큰 리스트에서 가장 신뢰도 높은 토큰들을 선택합니다.

        Args:
            tokens (List[Tuple[str, int]]): (토큰, 신뢰도) 형태의 리스트

        Returns:
            str: 선택된 토큰들을 공백으로 합친 문자열
        """
        if not tokens:
            return ""

        # 토큰별 등장 횟수 계산
        token_counter = Counter([t[0].lower() for t in tokens])

        # 신뢰도와 등장 횟수를 모두 고려하여 점수 계산
        token_scores = {}
        for token, conf in tokens:
            token_lower = token.lower()
            # 점수 = 신뢰도 * (1 + 등장 횟수/10)
            score = conf * (1 + token_counter[token_lower] / 10)
            if token_lower in token_scores:
                token_scores[token_lower] = max(token_scores[token_lower], score)
            else:
                token_scores[token_lower] = score

        # 점수가 높은 순으로 정렬된 고유 토큰 목록
        sorted_tokens = sorted(
            [(t, s) for t, s in token_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # 원래 형태(대소문자)로 토큰 복원 - 가장 많이 등장한 형태 선택
        best_case_tokens = {}
        for token, _ in tokens:
            token_lower = token.lower()
            if token_lower in best_case_tokens:
                best_case_tokens[token_lower].append(token)
            else:
                best_case_tokens[token_lower] = [token]

        # 가장 많이 등장한 형태로 변환
        result_tokens = []
        for token_lower, _ in sorted_tokens:
            if token_lower in best_case_tokens:
                counter = Counter(best_case_tokens[token_lower])
                best_form = counter.most_common(1)[0][0]
                result_tokens.append(best_form)

        return " ".join(result_tokens)

    def split_image_into_rows_with_layoutlm(self, image: np.ndarray, row_height: int = 190) -> List[Dict[str, str]]:
        """
        전처리된 ROI 이미지를 행 단위로 분할한 후, 각 행별로 LayoutLM v3 기반 OCR을 수행합니다.

        Args:
            image (np.ndarray): 전처리된 ROI 이미지 (그레이스케일)
            row_height (int): 한 행의 높이 (픽셀 단위)

        Returns:
            List[Dict[str, str]]: 각 행에서 추출된 텍스트 정보를 담은 dict 리스트
                                  (예: {"title": "...", "artist": "..."})
        """
        rows_info = []
        h, w = image.shape
        num_full_rows = h // row_height

        # 다양한 행 높이를 시도 (중첩된 영역으로 인식률 향상)
        for i in range(num_full_rows):
            y_start = i * row_height
            y_end = y_start + row_height

            # 기본 행 높이로 추출
            row_img = image[y_start:y_end, :]
            row_data = self.layoutlmv3_ocr_on_row(row_img)

            # 약간 확장된 영역으로 추가 추출 (10% 더 큰 영역)
            expanded_y_start = max(0, y_start - int(row_height * 0.05))
            expanded_y_end = min(h, y_end + int(row_height * 0.05))
            expanded_row_img = image[expanded_y_start:expanded_y_end, :]
            expanded_row_data = self.layoutlmv3_ocr_on_row(expanded_row_img)

            # 두 결과 중 더 좋은 결과 선택
            final_row_data = self.select_better_result(row_data, expanded_row_data)

            # 유의미한 정보가 있는 경우에만 저장
            if final_row_data and (final_row_data.get("title") or final_row_data.get("artist")):
                rows_info.append(final_row_data)
                self.logger.debug(f"행 {i} LayoutLM OCR 결과: {final_row_data}")

        # 남은 부분 처리 (전체 행 높이의 절반 이상일 경우)
        remainder = h % row_height
        if remainder >= row_height // 2:
            y_start = num_full_rows * row_height
            row_img = image[y_start:, :]
            row_data = self.layoutlmv3_ocr_on_row(row_img)
            if row_data and (row_data.get("title") or row_data.get("artist")):
                rows_info.append(row_data)
                self.logger.debug(f"남은 행 LayoutLM OCR 결과: {row_data}")

        return rows_info

    def select_better_result(self, result1: Dict[str, str], result2: Dict[str, str]) -> Dict[str, str]:
        """
        두 OCR 결과 중 더 좋은 결과를 선택합니다.

        Args:
            result1 (Dict[str, str]): 첫 번째 OCR 결과
            result2 (Dict[str, str]): 두 번째 OCR 결과

        Returns:
            Dict[str, str]: 선택된 더 좋은 결과
        """
        # 양쪽 다 비어있는 경우
        if not result1 and not result2:
            return {}
        # 한쪽만 비어있는 경우
        if not result1:
            return result2
        if not result2:
            return result1

        # 문자열 길이와 특수문자 비율로 품질 평가
        def score_text(text: str) -> float:
            if not text:
                return 0
            # 특수 문자 비율 계산 (너무 많은 특수문자는 좋지 않음)
            special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_char_ratio = special_char_count / len(text) if text else 0
            # 길이와 특수문자 비율을 고려한 점수
            return len(text) * (1 - special_char_ratio * 2)

        # 제목과 아티스트 점수 계산
        score1 = score_text(result1.get("title", "")) + score_text(result1.get("artist", ""))
        score2 = score_text(result2.get("title", "")) + score_text(result2.get("artist", ""))

        # 점수가 높은 결과 반환
        return result1 if score1 >= score2 else result2

    def extract_frames(self, interval: int = 5) -> None:
        """
        비디오에서 프레임 추출

        Args:
            interval (int): 프레임 추출 간격 (더 작은 간격으로 설정하여 더 많은 프레임 추출)
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

    def get_fuzzy_similarity(self, str1: str, str2: str) -> float:
        """
        두 문자열 간의 유사도를 계산합니다 (레벤슈타인 거리 기반).

        Args:
            str1 (str): 첫 번째 문자열
            str2 (str): 두 번째 문자열

        Returns:
            float: 0~1 사이의 유사도 (1이 완전 일치)
        """
        if not str1 or not str2:
            return 0.0

        # 레벤슈타인 거리 계산을 위한 함수
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        # 거리를 유사도로 변환 (최대 거리는 더 긴 문자열의 길이)
        max_len = max(len(str1), len(str2))
        distance = levenshtein(str1.lower(), str2.lower())
        similarity = 1 - (distance / max_len) if max_len > 0 else 0

        return similarity

    def is_similar_song(self, song1: Dict[str, str], song2: Dict[str, str], threshold: float = 0.8) -> bool:
        """
        두 곡 정보가 유사한지 확인합니다.

        Args:
            song1 (Dict[str, str]): 첫 번째 곡 정보
            song2 (Dict[str, str]): 두 번째 곡 정보
            threshold (float): 유사도 임계값 (이 값 이상이면 유사하다고 판단)

        Returns:
            bool: 유사하면 True, 아니면 False
        """
        # 제목 유사도
        title_similarity = self.get_fuzzy_similarity(
            song1.get("title", ""),
            song2.get("title", "")
        )

        # 아티스트 유사도
        artist_similarity = self.get_fuzzy_similarity(
            song1.get("artist", ""),
            song2.get("artist", "")
        )

        # 제목과 아티스트 모두 임계값 이상의 유사도를 가지면 유사한 곡으로 판단
        if title_similarity >= threshold and artist_similarity >= threshold:
            return True

        # 제목이 매우 유사하고(90% 이상) 아티스트 정보가 없거나 부분적인 경우
        if title_similarity >= 0.9 and (not song1.get("artist") or not song2.get("artist")):
            return True

        return False

    def merge_similar_songs(self, songs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        유사한 곡 정보를 병합합니다.

        Args:
            songs (List[Dict[str, str]]): 추출된 곡 정보 리스트

        Returns:
            List[Dict[str, str]]: 병합된 곡 정보 리스트
        """
        if not songs:
            return []

        merged_songs = []
        used_indices = set()

        for i, song1 in enumerate(songs):
            if i in used_indices:
                continue

            current_song = song1.copy()
            similar_songs = [song1]

            for j, song2 in enumerate(songs):
                if i == j or j in used_indices:
                    continue

                if self.is_similar_song(song1, song2):
                    similar_songs.append(song2)
                    used_indices.add(j)

            # 가장 긴/완전한 정보를 선택
            if len(similar_songs) > 1:
                title_lengths = [(s.get("title", ""), len(s.get("title", ""))) for s in similar_songs]
                artist_lengths = [(s.get("artist", ""), len(s.get("artist", ""))) for s in similar_songs]

                # 가장 긴 제목 선택
                best_title = max(title_lengths, key=lambda x: x[1])[0] if title_lengths else ""
                # 가장 긴 아티스트 선택
                best_artist = max(artist_lengths, key=lambda x: x[1])[0] if artist_lengths else ""

                current_song["title"] = best_title
                current_song["artist"] = best_artist

            merged_songs.append(current_song)
            used_indices.add(i)

        return merged_songs

    def perform_ocr(self) -> List[Dict]:
        """
        각 프레임에 대해 전처리된 ROI를 행 단위로 분할한 후,
        LayoutLM v3 기반 OCR을 수행하여 곡 정보를 추출합니다.

        Returns:
            List[Dict]: 프레임별 OCR 결과 및 곡 정보
        """
        self.logger.info("LayoutLM v3 OCR 처리를 시작합니다...")
        row_height = 190
        all_songs = []

        for idx, frame in enumerate(self.frame_buffer):
            # 전처리: ROI 추출, 그레이스케일 변환, 대비 개선
            preprocessed_roi = self.preprocess_frame(frame)
            # ROI를 행 단위로 분할하여 각 행별 LayoutLM v3 OCR 수행
            rows_data = self.split_image_into_rows_with_layoutlm(preprocessed_roi, row_height=row_height)
            # rows_data는 각 행의 {"title": ..., "artist": ...} dict 리스트
            songs = rows_data

            # 결과 저장 (추출 시 프레임 간격은 interval 파라미터에 따라 달라짐)
            frame_number = idx * 5  # interval=5로 가정
            self.results.append({
                'frame_number': frame_number,
                'frame_path': str(self.frames_dir / f"frame_{frame_number:06d}.png"),
                'songs': songs
            })
            all_songs.extend(songs)
            self.logger.info(f"프레임 {idx + 1}/{len(self.frame_buffer)} 처리 완료")

        # 유사한 곡 정보 병합
        merged_songs = self.merge_similar_songs(all_songs)

        # 기존 결과 업데이트
        for result in self.results:
            # 각 프레임의 곡 정보를 병합된 정보로 업데이트
            updated_songs = []
            for song in result['songs']:
                # 병합된 곡 목록에서 가장 유사한 곡 찾기
                for merged_song in merged_songs:
                    if self.is_similar_song(song, merged_song):
                        updated_songs.append(merged_song)
                        break
                else:
                    # 유사한 곡을 찾지 못한 경우 원래 곡 정보 유지
                    updated_songs.append(song)
            result['songs'] = updated_songs

        return self.results

    def confidence_voting(self, song_occurrences: List[Dict[str, str]]) -> Dict[str, str]:
        """
        동일한 곡으로 판단된 여러 인식 결과에서 투표를 통해 가장 좋은 결과를 선택합니다.

        Args:
            song_occurrences (List[Dict[str, str]]): 동일 곡으로 판단된 여러 인식 결과

        Returns:
            Dict[str, str]: 투표를 통해 선택된 최종 결과
        """
        if not song_occurrences:
            return {}

        # 제목 투표
        title_votes = Counter()
        for song in song_occurrences:
            title = song.get("title", "").strip()
            if title:
                title_votes[title] += 1

        # 아티스트 투표
        artist_votes = Counter()
        for song in song_occurrences:
            artist = song.get("artist", "").strip()
            if artist:
                artist_votes[artist] += 1

        # 가장 많은 투표를 받은 결과 선택
        best_title = title_votes.most_common(1)[0][0] if title_votes else ""
        best_artist = artist_votes.most_common(1)[0][0] if artist_votes else ""

        return {
            "title": best_title,
            "artist": best_artist
        }

    def export_to_csv(self) -> str:
        """
        파싱된 곡 정보를 CSV 파일로 저장합니다.

        Returns:
            str: 저장된 CSV 파일 경로
        """
        if not self.results:
            raise ValueError("추출된 결과가 없습니다.")

        # 모든 곡 정보 수집
        all_songs = []
        for result in self.results:
            for song in result['songs']:
                # 제목이나 아티스트 정보가 없는 경우는 건너뜁니다.
                if song.get('title') or song.get('artist'):
                    all_songs.append({
                        'frame': result['frame_number'],
                        'frame_path': result['frame_path'],
                        'title': song.get('title', ''),
                        'artist': song.get('artist', '')
                    })

        # 유사한 곡 제거
        unique_songs = []
        for song in all_songs:
            # 이미 추가된 곡과 유사한지 확인
            is_unique = True
            for unique_song in unique_songs:
                if self.is_similar_song(song, unique_song, threshold=0.7):
                    is_unique = False
                    # 더 많은 정보가 있는 경우 업데이트
                    if (len(song['title']) > len(unique_song['title']) or
                        len(song['artist']) > len(unique_song['artist'])):
                        unique_song['title'] = song['title'] if len(song['title']) > len(unique_song['title']) else unique_song['title']
                        unique_song['artist'] = song['artist'] if len(song['artist']) > len(unique_song['artist']) else unique_song['artist']
                    break
            if is_unique:
                unique_songs.append(song)

        # 결과를 데이터프레임으로 변환
        df = pd.DataFrame(unique_songs)

        # 신뢰도가 높은 결과만 남김 (빈 문자열이나 너무 짧은 텍스트 필터링)
        df = df[df['title'].str.len() > 2 | df['artist'].str.len() > 2]

        # 중복 제거 (제목과 아티스트가 동일한 행)
        df = df.drop_duplicates(subset=['title', 'artist'])

        # 결과 저장
        video_path = Path(self.video_path)
        output_path = video_path.parent / f"{video_path.stem}_playlist.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 통계 정보 로깅
        self.logger.info(f"총 {len(df)}개의 고유한 곡 정보가 추출되었습니다.")
        self.logger.info(f"결과가 저장되었습니다: {output_path}")

        return str(output_path)

    def validate_song_info(self, song: Dict[str, str]) -> bool:
        """
        추출된 곡 정보가 유효한지 검증합니다.

        Args:
            song (Dict[str, str]): 검증할 곡 정보

        Returns:
            bool: 유효하면 True, 아니면 False
        """
        title = song.get('title', '').strip()
        artist = song.get('artist', '').strip()

        # 제목이나 아티스트 둘 중 하나는 반드시 있어야 함
        if not title and not artist:
            return False

        # 텍스트 길이가 너무 짧으면 노이즈일 가능성이 높음
        if title and len(title) < 2:
            return False
        if artist and len(artist) < 2:
            return False

        # 너무 많은 특수 문자가 포함된 경우 (노이즈일 가능성 높음)
        special_chars = sum(1 for c in title + artist if not c.isalnum() and not c.isspace())
        total_chars = len(title + artist)
        if total_chars > 0 and special_chars / total_chars > 0.4:
            return False

        return True


def main():
    # 개선된 설정으로 Tesseract OCR 구성
    tesseract_config = '--oem 1 --psm 6 -l kor+eng --dpi 300'

    ocr = PlaylistOCR(tesseract_config=tesseract_config)
    try:
        if not ocr.select_video():
            print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
            return

        print("프레임 추출을 시작합니다...")
        ocr.extract_frames(interval=5)  # 프레임 간격을 짧게 설정하여 더 많은 프레임 추출

        print("OCR 처리를 시작합니다...")
        results = ocr.perform_ocr()
        output_path = ocr.export_to_csv()

        print("\nLayoutLM v3 OCR 처리가 완료되었습니다!")
        print(f"결과가 다음 위치에 저장되었습니다: {output_path}")
        print(f"프레임 이미지가 다음 위치에 저장되었습니다: {ocr.frames_dir}")

    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        logging.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()