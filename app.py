import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("テトリス99　リザルド画面抽出ツール")

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


    # 【名前用】文字の輪郭を損なわずに3倍拡大する
    def preprocess_name(crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        return resized


    # 【レベル用】ドット単位でシャープに拡大して数字と星の癒着を防ぐ
    def preprocess_level(crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # NEAREST（最近傍法）で、ぼかさずに3.5倍拡大
        resized = cv2.resize(gray, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_NEAREST)
        return resized


    # 星マークの化けを自動で「★」に補正
    def clean_level_text(text):
        if not text or text == "---":
            return "---"
        for char in ['+', '*', '.', ':', '"', '`', "'", 'x', 'r']:
            text = text.replace(char, '★')
        text = text.replace(' ', '')
        return text


    with st.spinner("AI画像認識を実行中..."):
        for i in range(11):
            r_y_start = int(i * row_height)
            r_y_end = int((i + 1) * row_height)
            row_img = ranking_area[r_y_start:r_y_end, :]

            # 【黄金比】アイコンを完璧に避け、右端のK.O.マークもカットするピクセル幅
            name_crop = row_img[:, 105:325]  # 名前
            level_crop = row_img[:, 335:430]  # レベル/ランク

            # 専用の画像拡大を適用
            name_preprocessed = preprocess_name(name_crop)
            level_preprocessed = preprocess_level(level_crop)

            # --- OCR実行 ---
            # 1. 名前の読み取り
            name_res = reader.readtext(name_preprocessed, detail=0)
            detected_name = name_res[0] if name_res else "---"

            # 2. レベルの読み取り（数字、★、および化けやすい記号を許可）
            level_res = reader.readtext(
                level_preprocessed,
                detail=0,
                allowlist='0123456789★☆*+.:"`xr '
            )
            raw_level = level_res[0] if level_res else "---"
            detected_level = clean_level_text(raw_level)

            # 3. 順位の動的読み取り（固定連番から、OCR自動読込に変更）
            rank_crop = row_img[:, 0:55]
            rank_preprocessed = preprocess_name(rank_crop)  # 名前の3倍滑らか拡大を再利用
            rank_res = reader.readtext(rank_preprocessed, detail=0, allowlist='0123456789#')

            raw_rank = rank_res[0] if rank_res else ""
            cleaned_rank = "".join([c for c in raw_rank if c.isdigit() or c == '#'])  # 余計な文字を排除

            if not cleaned_rank:
                detected_rank = f"#{i + 1}"  # 読み取れなかった場合のみ従来の連番をフォールバック
            else:
                # #がついていなければ自動で付与
                detected_rank = cleaned_rank if cleaned_rank.startswith('#') else f"#{cleaned_rank}"

            data_rows.append({
                "順位": detected_rank,
                "名前": detected_name,
                "レベル/ランク": detected_level
            })

    # 表示用データフレーム
    df = pd.DataFrame(data_rows)
    st.write("### 抽出データ（完成版）")
    st.dataframe(df)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="このテーブルをCSVとして保存",
        data=csv,
        file_name="tetris99_complete_table.csv",
        mime="text/csv"
    )