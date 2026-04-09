
from datasets import load_dataset
from .gsm8k import GSM8KDataset
import re
import random
from .parsers import remove_boxed, last_boxed_only_string

SVAMP_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

SVAMP_VALIDATION_SIZE = 150

class SVAMPDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SVAMP_SYSTEM_PROMPT,
        subsample=-1,
    ):
        if num_examples > 0:
            raise ValueError("num_examples must be 0 for SVAMP dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,  
            add_reasoning,
            system_prompt,
            subsample,
        )
    
    def load_test_dataset(self):
        self.dataset = load_dataset("ChilleD/SVAMP", "default", split="test")

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["Body"] + " " + self.dataset[self.subsample[idx].item()]["Question"]
        target = int(self.dataset[self.subsample[idx].item()]["Answer"])
        prompt = self.create_prompt(question)
        return prompt, question, target
    

def parse_svamp_answer(raw_generation):
    parsed_answer = None
    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass

    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass
    return parsed_answer