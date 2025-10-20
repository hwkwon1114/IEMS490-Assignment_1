import utils
from langchain.prompts import ChatPromptTemplate
import time


def main():
    print("--- Running Setup ---")
    utils.setup_api_key()
    df_sampled = utils.data_sample()
    chat = utils.initialize_model()
    print("\n--- Setup Complete ---")
    print(f"Data loaded with shape: {df_sampled.shape}")
    coT_string = """You are a helpful math assistant.
    1. Solve the following question by showing your reasoning step-by-step.
    2. After all your reasoning, provide the final numerical answer on a *new line*, prefixed with "####".

    Question: {question}
    Answer: Let's think step by step.
    """
    prompt_template = ChatPromptTemplate.from_template(coT_string)
    results = []
    llm_answers = []
    num_q = 150
    REQUEST_DELAY_SECONDS = 4
    for i in range(num_q):
        prompt = prompt_template.format_messages(question = df_sampled.iloc[i,0])
        response = chat(prompt)
        answer = response.content
        llm_answers.append(answer)
        results.append(utils.parser(df_sampled.iloc[i,1])== utils.parser(answer))
        ## Added Delay because my gemini limits requests per minute
        time.sleep(REQUEST_DELAY_SECONDS)
    accuracy = sum(results) / len(results)
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Correct: {sum(results)} / {len(results)}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

    results_df = df_sampled.iloc[:num_q].copy()
    results_df['llm_answer'] = llm_answers
    results_df['is_correct'] = results
    results_df.to_csv("CoT.csv", index=False)

if __name__ == '__main__':
    main()