import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("テトリス99 解析ツール（ハイブリッド精度版）")
st.write("数字と星マークの『癒着』を画像処理で切断し、2桁の数字と星を両立して認識させます。")

uploaded_file = st.file_uploader("リザルト画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 1. 1280x720にサイズを統一
    img_resized = cv2.resize(img, (1280, 720))

    # 中央リザルト全体の範囲
    y_start, y_end = 195, 655
    x_start, x_end = 370, 830
    ranking_area = img_resized[y_start:y_end, x_start:x_end]

    st.image(ranking_area, caption="解析対象エリア", use_container_width=True)

    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['ja', 'en'])

    reader = load_ocr()

    # 11行分の高さを計算
    row_height = (y_end - y_start) / 11.0
    data_rows = []

    # 【名前用】滑らかに3倍に拡大して文字の形を完全に維持する（実績100%）
    def preprocess_name(crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        return resized

    # 【レベル用・新設計】滑らかに拡大したあと、文字同士を繋ぐボヤけた境界線を「二値化」で切断する
    def preprocess_level_hybrid(crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # 滑らかに拡大
        resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        # 閾値180で二値化（文字を繋ぐ薄暗いグレーの架け橋を真っ黒にして「切断」する）
        _, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_BINARY)
        return thresh

    # 星マークの置換補正
    def clean_level_text(text):
        if not text or text == "---":
            return "---"
        for char in ['+', '*', '.', ':', '"', '`', "'", 'x', 'r', 'y', 's', 'A', 'a', 'g']:
            text = text.replace(char, '★')
        text = text.replace(' ', '')
        return text

    with st.spinner("AI画像認識を実行中..."):
        for i in range(11):
            r_y_start = int(i * row_height)
            r_y_end = int((i + 1) * row_height)
            row_img = ranking_area[r_y_start:r_y_end, :]

            # 黄金比の切り出し幅
            name_crop = row_img[:, 105:325]   # 名前
            level_crop = row_img[:, 335:430]  # レベル/ランク

            # 各エリアに最適な前処理を適用
            name_preprocessed = preprocess_name(name_crop)
            level_preprocessed = preprocess_level_hybrid(level_crop)

            # --- OCR実行 ---
            # 1. 名前の読み取り
            name_res = reader.readtext(name_preprocessed, detail=0)
            detected_name = name_res[0] if name_res else "---"

            # 2. レベルの読み取り
            level_res = reader.readtext(
                level_preprocessed,
                detail=0,
                allowlist='0123456789★☆*+.:"`xry sAa'
            )
            raw_level = level_res[0] if level_res else "---"
            detected_level = clean_level_text(raw_level)

            # 3. 順位は自動付与
            detected_rank = f"#{i+1}"

            data_rows.append({
                "順位": detected_rank,
                "名前": detected_name,
                "レベル/ランク": detected_level
            })

    # 表示用データフレーム
    df = pd.DataFrame(data_rows)
    st.write("### 抽出データ（改善版）")
    st.dataframe(df)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="このテーブルをCSVとして保存",
        data=csv,
        file_name="tetris99_complete_table.csv",
        mime="text/csv"
    )