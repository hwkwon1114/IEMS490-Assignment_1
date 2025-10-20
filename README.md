# IEMS490:Assignment_1
Manual and Automated Prompt Engineering

Instructions on Running

On Terminal, Type 

docker build -t llm-assignment-1 .

docker run -it -e GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE" llm-assignment-1

The api key will need to be replaced using Google AI Studio's API key which can be found here: https://aistudio.google.com/

1. Dataset Overview
2. 
THe dataset is from the GSM8K and have randomly sampled 150 data points to use to predict the accuracy of the llm model. We have used the Gemini 2.5 Flash Lite model to evaluate the performance of the prompt engineering techniques. The Utils.py file contains all of the functions that initialize the chat model and the parsing function while there is a a python file associated with each part of the assignment.

3. Vanilla Prompting (No Technique)
4. 
The associated file for part 1 is Part1BasePrompt.py. The prompt that was used for the vanilla prompt was:

"""Read the following math question.
    Your task is to provide *only* the final numerical answer.
    Do not show your work. Do not use any text, units, or LaTeX.
    Format your answer on a single line, starting with the "####" prefix.

    Question: {question}
    Final Answer: """
    
The accuracy of the vanilla prompt was 28% as shown in the output documentation. The responses are documented in Vanilla_2.5.csv.

3. Manual Prompt Engineering


The associated file for part 2 is Part2CoT.py.I used the CoT technique with the role assignment technique to find the answer. While few shot technique was utilized to try optimize for the answer, the accuracy did not increase so the simple CoT technique was utilized. Here is the prompt that was used.
"""You are a helpful math assistant.
    1. Solve the following question by showing your reasoning step-by-step.
    2. After all your reasoning, provide the final numerical answer on a *new line*, prefixed with "####".

    Question: {question}
    Answer: Let's think step by step.
    """
    
We were able to achieve an accuracy of 94.67%. The responses are saved in CoT.csv.

4. Automated Prompt Engineering

The associated file for part 3 is Part3.py. The automated prompting process begins by partitioning the GSM8k dataset into a development set and a completely separate test set for final validation, ensuring no data leakage.

The algorithm is initialized with the best-performing manual prompt from Part 2, which is first evaluated on the development set to get a baseline score and an initial list of incorrectly answered questions. The core of the algorithm is an iterative loop that runs for five times:
First, a "meta-prompt" is given the set of questions the current best prompt failed on, including the model's incorrect output.
Then, the meta-prompt instructs an LLM to act as a diagnostician, analyze the failure modes, and generate a new candidate prompt designed to fix those specific errors.
Next, this new candidate is first tested against only the known failed questions.
Full Regression Test & Selection: If the candidate shows promise, it is promoted to a full "regression test" on the entire development set. This ensures the changes didn't negatively impact previously correct answers. A new prompt is only adopted as the best prompt if its overall accuracy on this full test is strictly higher than the previous prompt.
This process is repeated for five times, allowing the algorithm to learn from different wrong answers. The final, optimized prompt is then scored one last time against the unseen test set to get its final, unbiased accuracy.

The accuracy was 95.33% on the same set of test data. The responses are saved at auto_prompt_results.csv and the optimized prompt is saved at best_prompt1.txt.
