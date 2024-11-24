import streamlit as st
import pandas as pd
import html
from ast import literal_eval


class CSVData:
    def __init__(self, file, flatten_condition=False):
        """
        Initialize the CSVData object.
        Args:
            file: Uploaded file object (e.g., from Streamlit file uploader).
            flatten_condition: Whether to apply flatten_json to csv file
        """
        self.file_name = (file.name).replace(".csv", "")  # File name
        self.data = load_data(file)  # Data loaded as DataFrame
        if flatten_condition:
            self.data = self._flatten_json(self.data)

    def _flatten_json(self, df):
        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            records.append(record)
        return pd.DataFrame(records)

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "data": self.data.to_dict(orient="records")
        }


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


def display_answer_distribution(title, df):
    st.write(f"#### {title}")
    answer_counts = df['answer'].value_counts()
    answer_counts = answer_counts.reindex(range(1, 6), fill_value=0)
    st.bar_chart(answer_counts.sort_index())


@st.cache_data
def get_comparison_data(train_data, output_data_list):
    train_df = pd.DataFrame(train_data["data"])
    output_dfs = [pd.DataFrame(output_data["data"]) for output_data in output_data_list]
    output_file_names = [output_data["file_name"] for output_data in output_data_list]

    results = []

    for problem_id in train_df["id"]:
        record = train_df[train_df["id"] == problem_id]

        # 각 파일의 예측 값 비교
        predictions = [df[df["id"] == problem_id]["answer"] for df in output_dfs]
        if predictions[0].empty:
            continue
        statuses = [int(pred.values[0] == record["answer"].values[0]) for pred in predictions]

        # 카테고리 생성
        correct_files = [output_file_names[i] for i, status in enumerate(statuses) if status == 1]
        incorrect_files = [output_file_names[i] for i, status in enumerate(statuses) if status == 0]

        category = ""
        if len(correct_files) == len(output_file_names):  # 모든 파일이 정답
            category = "모든 output_file이 맞춤"
        elif len(incorrect_files) == len(output_file_names):  # 모든 파일이 오답
            category = "모든 output_file이 틀림"
        else:  # 특정 파일만 정답
            category = ", ".join(f"{file}만 맞춤" for file in correct_files)

        results.append({
            "id": problem_id,
            "category": category,
            "paragraph": record["paragraph"].values[0],
            "question": record["question"].values[0],
            "choices": record["choices"].values[0],
            "correct_answer": record["answer"].values[0],
            "predictions": [pred.values[0] for pred in predictions],
        })
    return pd.DataFrame(results)


tabs = st.tabs(["Upload Files", "Answer Distribution", "Output Comparison"])

### File Upload ###
with tabs[0]:
    train_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])
    train_data = None
    if train_file:
        train_data = CSVData(train_file, flatten_condition=True)

    output_files = st.file_uploader(
        "Upload your output CSV files (multiple allowed)", type=["csv"], accept_multiple_files=True
    )
    output_data_list = []
    if output_files:
        for output_file in output_files:
            output_data = CSVData(output_file, flatten_condition=False)
            output_data_list.append(output_data)

### Tabs ###
if train_data or len(output_data_list) > 0:
    # Tab 1: Answer Distribution
    with tabs[1]:
        if train_data:
            st.write("### Train Data Answer Distribution")
            display_answer_distribution(train_data.file_name, train_data.data)

        st.write("---")

        if len(output_data_list) > 0:
            st.write(f"### Output Data Answer Distribution")
            for output_data in output_data_list:
                display_answer_distribution(output_data.file_name, output_data.data)

    # Tab 2: Output Comparison
    with tabs[2]:
        if train_data and len(output_data_list) > 0:
            comparison_df = get_comparison_data(train_data.to_dict(),
                                                [output_data.to_dict() for output_data in output_data_list])
            if "output_data_length" not in st.session_state:
                st.session_state.output_data_length = len(output_data_list)

            # Sidebar: Filter options
            category_filter_options = comparison_df["category"].unique()
            category_filter = st.sidebar.selectbox("Filter by category", category_filter_options)

            # comparison_df가 바뀌었을 때만 filtered_df를 새로 갱신
            if "filtered_df" not in st.session_state or st.session_state.output_data_length != len(output_data_list):
                filtered_df = comparison_df[comparison_df["category"] == category_filter]
                filtered_df.loc[:, "id"] = filtered_df["id"].apply(lambda x: x.split("-")[-1])
                st.session_state.filtered_df = filtered_df
                st.session_state.comparison_df = comparison_df  # comparison_df를 세션 상태에 저장하여 변경 여부 추적
            else:
                # comparison_df가 바뀌지 않으면 세션 상태에서 가져오기
                filtered_df = st.session_state.filtered_df

            problem_id = st.sidebar.selectbox("Select a problem ID", filtered_df["id"])

            if problem_id:
                problem_data = filtered_df[filtered_df["id"] == problem_id].iloc[0]

                # Paragraph 박스
                st.markdown(f"""
                    <div style='padding: 15px; border: 1px solid #000;'>{problem_data['paragraph']}</div>
                """, unsafe_allow_html=True)

                # Question (문제 번호와 함께 표시)
                st.markdown(
                    f"<br><span style='font-size: 20px; font-weight: bold;'>{problem_id}. </span> {problem_data['question']}",
                    unsafe_allow_html=True)

                # Choices
                for idx, choice in enumerate(problem_data['choices']):
                    choice_number = f"&#x2460;".replace('0', str(idx))  # ①, ②, ③ 등의 유니코드 원 사용
                    if idx + 1 == problem_data["correct_answer"]:
                        st.markdown(
                            f"<div style='background-color:#FFFF00; display: inline-block; margin-bottom: 5px;'>{choice_number} {choice}</div>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div style='display: inline-block; margin-bottom: 5px;'>{choice_number} {choice}</div>",
                            unsafe_allow_html=True)

                # 선택지와 테이블 사이에 간격 추가
                st.markdown("<br><br>", unsafe_allow_html=True)

                # 테이블 생성
                table_html = "<table style='width:100%; border-collapse: collapse;'>"
                table_html += "<tr><th style='padding: 10px; text-align: left;'>Output File</th><th style='padding: 10px; text-align: left;'>Prediction</th><th style='padding: 10px; text-align: left;'>Result</th></tr>"

                # 문제별 예측 결과 표시
                for i, pred in enumerate(problem_data["predictions"]):
                    # 예측이 맞으면 강조
                    result = "✅" if pred == problem_data["correct_answer"] else "❌"

                    # 각 예측을 테이블로 추가
                    table_html += f"<tr><td style='padding: 10px; border: 1px solid #ddd; font-weight: bold;'>{output_data_list[i].file_name}</td>"
                    table_html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{pred}</td>"
                    table_html += f"<td style='padding: 10px; border: 1px solid #ddd; color: {'green' if pred == problem_data['correct_answer'] else 'red'}'>{result}</td></tr>"

                table_html += "</table>"
                # 테이블을 Streamlit에서 출력
                st.markdown(table_html, unsafe_allow_html=True)
