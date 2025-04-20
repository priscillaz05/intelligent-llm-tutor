import google.generativeai as genai
import os
import json
import time
import threading
from datetime import datetime
from collections import deque
from datasets import load_dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ------------------------------
# RateLimiter (as in your provided code)
# ------------------------------
class RateLimiter:
    def __init__(self):
        self.rpm_queue = deque(maxlen=15)
        self.daily_requests = 0
        self.last_reset_date = datetime.now().date()
        self.lock = threading.Lock()

    def can_make_request(self):
        with self.lock:
            current_time = datetime.now()
            current_date = current_time.date()

            if current_date > self.last_reset_date:
                self.daily_requests = 0
                self.last_reset_date = current_date

            while self.rpm_queue and (current_time - self.rpm_queue[0]).total_seconds() > 60:
                self.rpm_queue.popleft()

            if (len(self.rpm_queue) < 15 and self.daily_requests < 1500):
                return True
            return False

    def record_request(self):
        with self.lock:
            self.rpm_queue.append(datetime.now())
            self.daily_requests += 1

# ------------------------------
# Function to process one TACO dataset sample
# ------------------------------
def process_sample(sample, sample_index, output_directory, prompt_path, error_log_path, rate_limiter, model):
    try:
        sample["solutions"] = json.loads(sample["solutions"])
        sample["input_output"] = json.loads(sample["input_output"])
        sample["raw_tags"] = eval(sample["raw_tags"])
        sample["tags"] = eval(sample["tags"])
        sample["skill_types"] = eval(sample["skill_types"])
        
        ground_truth_solution = sample["solutions"][0]
        coding_problem = sample["question"]

        with open(prompt_path, "r") as file:
            few_shot_examples = file.read().strip()

        prompt = f"""You are provided with a coding problem from the TACO dataset. Your task is to generate a JSON object that exactly follows this format with the following keys:
- "Coding Problem"
- "Ground Truth Solution"
- "LLM CoT Steps Breakdown"
- "LLM Questions"
- "Expected Answers to LLM Questions"

Below are some few-shot examples to guide you:
{few_shot_examples}

Coding Problem: {coding_problem}
Ground Truth Solution:
{ground_truth_solution}
"""

        while not rate_limiter.can_make_request():
            time.sleep(1)
        rate_limiter.record_request()

        response = model.generate_content([prompt])

        os.makedirs(output_directory, exist_ok=True)

        if response and response.text:
            output_file = os.path.join(output_directory, f"custom_entry_{sample_index}.json")
            with open(output_file, "w") as out_file:
                json.dump(response.text, out_file, indent=6)
            print(f"Successfully processed sample {sample_index}")
        else:
            print(f"No valid response for sample {sample_index}")
    except Exception as e:
        print(f"Error processing sample {sample_index}: {str(e)}")
        with open(error_log_path, "a") as error_log:
            error_log.write(f"{datetime.now()}: Error processing sample {sample_index}: {str(e)}\n")

# ------------------------------
# Process all samples from the TACO dataset
# ------------------------------
def process_all_samples(process_func, output_directory, prompt_path, error_log_path, rate_limiter, model, dataset, start_index=0):
    total_samples = len(dataset)
    print(f"Found {total_samples} samples to process. Starting from index {start_index}.")
    for i in range(start_index, total_samples):
        sample = dataset[i]
        process_func(sample, i, output_directory, prompt_path, error_log_path, rate_limiter, model)
        time.sleep(4)

# ------------------------------
# Main entry point
# ------------------------------
if __name__ == "__main__":
    api_key = "AIzaSyCldY4woCjGjqPCdJccQydIJrfJScp40aw"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    ds = load_dataset("BAAI/TACO", split="train")

    output_dir = "taco_custom_dataset"
    prompt_file = "prompt.txt"     
    error_log_file = "error_log.txt"

    rate_limiter = RateLimiter()

    process_all_samples(
        process_func=process_sample,
        output_directory=output_dir,
        prompt_path=prompt_file,
        error_log_path=error_log_file,
        rate_limiter=rate_limiter,
        model=model,
        dataset=ds
    )

    print("Custom dataset creation complete.")