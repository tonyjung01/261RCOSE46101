import os
import json
from .parsers import Parser, evaluate_equation, validate_equation
from .gsm8k import GSM8KDataset
import re
from datasets import load_dataset
from .parsers import remove_boxed, last_boxed_only_string
import warnings

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)

class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self.create_prompt(question)
        return prompt, question, (numbers, target)


def parse_ctd_answer(raw_generation):
    def parentheses_matched(s):
        stack = []
        for char in s:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return not stack
    def extract_numbers(equation: str):
        numbers = []
        num = ''
        for ch in equation:
            if ch.isdigit():
                num += ch
            elif num:
                numbers.append(int(num))
                num = ''
                if len(numbers) > 3:  # 提前剪枝
                    break
        if num:
            numbers.append(int(num))
        return numbers
    equation = None
    try:
        equation = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        # Try to extract from answer tags
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = raw_generation

    # Replace LaTeX operators with Python operators
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

    # Check for equation with equals sign and extract only the expression part
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()

    if not re.match(r"^[0-9+\-*/() ]+$", equation) or not parentheses_matched(equation):
        return None
    numbers_in_eq = extract_numbers(equation)
    if len(numbers_in_eq) != 3:
        return None
    return equation