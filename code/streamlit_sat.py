import ast
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


def load_data(file):
    data = pd.read_csv(file)
    return data


# Function to display details of each problem entry
def display_problem_details(problem_data):
    try:
        problem_dict = ast.literal_eval(problem_data)
    except (ValueError, SyntaxError):
        st.error("Invalid format in problem data. Unable to parse.")
        return

    question = problem_dict.get('question', 'N/A')
    choices = problem_dict.get('choices', [])
    answer = problem_dict.get('answer', 'N/A')
    st.write(f"**Question:** {question}")
    st.write(f"**Choices:** {choices}")
    st.write(f"**Answer:** {answer}")


# Function to plot length distribution as a histogram
def plot_length_histogram(data, column_name, title, isDict=False, dictName=None):
    if isDict:
        lengths = data[column_name].apply(lambda x: len(str(ast.literal_eval(x).get(dictName, ''))) if pd.notnull(x) else 0)
    else:
        lengths = data[column_name].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f'{title} Length Distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


# Main Streamlit app
st.title("Korean CSAT Language Section EDA")

# File uploader for user-uploaded data
train_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])

if train_file is not None:
    # Load the uploaded data
    data = load_data(train_file)

    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Basic dataset information
    st.write("### Dataset Information")
    st.write(f"Number of entries: {len(data)}")
    st.write(f"Columns: {data.columns.tolist()}")

    # Display samples of each field
    st.write("### Field Analysis")

    # Display a sample paragraph
    st.subheader("Sample Paragraph")
    st.write(data['paragraph'].iloc[0])

    # Display a sample problem with details
    st.subheader("Sample Problem")
    display_problem_details(data['problems'].iloc[0])

    # Display a sample 'question_plus' entry
    st.subheader("Sample Question Plus")
    st.write(data['question_plus'].iloc[0])

    # Additional statistics
    st.write("### Additional Statistics")

    # Plot length distributions for each field
    plot_length_histogram(data, 'paragraph', 'Paragraph')
    plot_length_histogram(data, 'problems', 'Question', True, 'question')
    plot_length_histogram(data, 'problems', 'Choices', True, 'choices')
    plot_length_histogram(data, 'question_plus', 'Question Plus')

    # Answer count analysis
    st.write("#### Answer Distribution")
    answer_counts = data['problems'].apply(lambda x: ast.literal_eval(x)['answer']).value_counts()
    st.bar_chart(answer_counts.sort_index())


output_files = st.file_uploader(
    "Upload your output CSV files (multiple allowed)", type=["csv"], accept_multiple_files=True
)

if output_files is not None:
    for output_file in output_files:
        data = pd.read_csv(output_file)
        st.write(f"### Answer Distribution for {output_file.name}")
        answer_counts = data['answer'].value_counts()
        st.bar_chart(answer_counts.sort_index())

if train_file and output_files:
    # 파일 읽기
    train_df = pd.read_csv(train_file)
    output_dfs = [pd.read_csv(file) for file in output_files]

    # ID 기준으로 train_file에서 문제 정보 가져오기
    def get_problem_data(train_df, problem_id):
        data = train_df[train_df["id"] == problem_id].iloc[0]
        paragraph = data["paragraph"]
        problems = eval(data["problems"])  # 문자열로 저장된 dict를 다시 파싱
        return paragraph, problems

    # 문제별 비교 데이터 준비
    def get_comparison_data(train_df, output_dfs):
        results = []
        for problem_id in train_df["id"]:
            paragraph, problems = get_problem_data(train_df, problem_id)
            correct_answer = problems["answer"]

            # 각 파일의 예측 값 비교
            predictions = [df[df["id"] == problem_id]["answer"].values[0] for df in output_dfs]
            statuses = [int(pred == correct_answer) for pred in predictions]

            if sum(statuses) == len(output_dfs):  # 모두 정답
                category = "Both Correct"
            elif sum(statuses) == 0:  # 모두 오답
                category = "Both Incorrect"
            else:  # 일부 정답
                category = "One Correct"

            results.append({
                "id": problem_id,
                "category": category,
                "paragraph": paragraph,
                "question": problems["question"],
                "choices": problems["choices"],
                "correct_answer": correct_answer,
                "predictions": predictions,
            })
        return pd.DataFrame(results)

    comparison_df = get_comparison_data(train_df, output_dfs)

    # Sidebar: 문제 번호 선택
    category_filter = st.sidebar.selectbox("Filter by category", ["Both Correct", "Both Incorrect", "One Correct"])
    filtered_df = comparison_df[comparison_df["category"] == category_filter]
    problem_id = st.sidebar.selectbox("Select a problem ID", filtered_df["id"])

    # Main: 문제 정보 및 예측 값 표시
    if problem_id:
        problem_data = filtered_df[filtered_df["id"] == problem_id].iloc[0]
        st.write(f"**Paragraph:** {problem_data['paragraph']}")
        st.write(f"**Question:** {problem_data['question']}")
        st.write(f"**Choices:** {problem_data['choices']}")
        st.write(f"**Correct Answer:** {problem_data['correct_answer']}")

        st.write("### Predictions")
        for i, pred in enumerate(problem_data["predictions"]):
            st.write(f"Output File {i+1}: {pred}")
