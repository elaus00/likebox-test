import json
from tqdm import tqdm

def extract_words(file_path: str, key: str):
    words = set()
    # 파일의 총 줄 수를 미리 계산하면 tqdm 진행 바가 더 정확하게 동작할 수 있습니다.
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 대용량 파일이므로 한 줄씩 읽으며 처리한다.
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc=f"Processing {key} from {file_path}"):
            try:
                data = json.loads(line)
                if key in data:
                    # 불필요한 공백 제거 후 추가
                    word = data[key].strip()
                    if word:
                        words.add(word)
            except json.JSONDecodeError:
                continue
    return words

# 파일 경로 설정 (경로의 역슬래시(\) 대신 raw string을 사용)
artist_file = r"C:\musicbrainz_dumps\artist\mbdump\artist"
recording_file = r"C:\musicbrainz_dumps\recording\mbdump\recording"

# 각각에서 단어 추출 (아티스트의 경우 "name", 녹음의 경우 "title")
artist_words = extract_words(artist_file, 'name')
recording_words = extract_words(recording_file, 'title')

# 두 집합을 합쳐 중복을 제거합니다.
all_words = artist_words.union(recording_words)

# 결과를 텍스트 파일로 저장 (각 단어를 한 줄에 하나씩)
with open("musicbrainz_words.txt", "w", encoding="utf-8") as f:
    for word in all_words:
        f.write(word + "\n")

print(f"총 {len(all_words)}개의 단어가 사용자 사전에 저장되었습니다.")