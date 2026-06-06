import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("テトリス99 解析ツール（最高精度チューニング版）")
st.write("バイラテラル・フィルタの導入と、K.O.バッジの自動除外により、検出精度をさらに極限まで高めました。")

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


    # 【プロ仕様】エッジ（文字の輪郭）を保存したまま、背景ノイズだけを強力に潰す前処理
    def preprocess_advanced(crop_img, zoom_factor=3):
        # 1. グレースケール化
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # 2. バイラテラル・フィルタ（文字の輪郭をボカさず、背景のノイズだけを平滑化する）
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. 高品質な拡大（CUBIC）
        resized = cv2.resize(filtered, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        return resized


    # 星マークの表記ゆれ補正
    def clean_level_text(text):
        if not text or text == "---":
            return "---"
        for char in ['+', '*', '.', ':', '"', '`', "'", 'r', 'x']:
            text = text.replace(char, '★')
        text = text.replace(' ', '')
        return text


    with st.spinner("極限精度でAI画像認識を実行中..."):
        for i in range(11):
            r_y_start = int(i * row_height)
            r_y_end = int((i + 1) * row_height)
            row_img = ranking_area[r_y_start:r_y_end, :]

            # 【調整】アイコン巻き込み防止のため105開始、K.O.バッジ除外のため右端を430に制限
            name_crop = row_img[:, 105:325]  # 名前
            level_crop = row_img[:, 340:430]  # レベル/ランク（右端を430にしてK.O.マークを完全除外）

            # 高度な前処理を適用
            name_preprocessed = preprocess_advanced(name_crop, zoom_factor=3.0)
            level_preprocessed = preprocess_advanced(level_crop, zoom_factor=3.5)

            # --- OCR実行（beamWidth=10でAIの探索精度を強化） ---
            # 1. 名前の読み取り
            name_res = reader.readtext(name_preprocessed, detail=0, beamWidth=10)
            detected_name = name_res[0] if name_res else "---"

            # 2. レベルの読み取り
            level_res = reader.readtext(
                level_preprocessed,
                detail=0,
                allowlist='0123456789★☆*+.:"`xr ',
                beamWidth=10
            )
            raw_level = level_res[0] if level_res else "---"
            detected_level = clean_level_text(raw_level)

            # 3. 順位は自動付与
            detected_rank = f"#{i + 1}"

            data_rows.append({
                "順位": detected_rank,
                "名前": detected_name,
                "レベル/ランク": detected_level
            })

    # 表示用データフレーム
    df = pd.DataFrame(data_rows)
    st.write("### 抽出データ（最高精度版）")
    st.dataframe(df)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="このテーブルをCSVとして保存",
        data=csv,
        file_name="tetris99_premium_table.csv",
        mime="text/csv"
    )