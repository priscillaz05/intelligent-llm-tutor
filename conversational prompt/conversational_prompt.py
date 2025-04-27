import google.generativeai as genai
import json
import time

if __name__ == "__main__":
    api_key = ""
    genai.configure(api_key=api_key)

    with open('confusion_generation_prompt.txt', 'r') as file:
        confusion_prompt = file.read()

    with open('coding_question.txt', 'r') as file:
        coding_question = file.read()

    confusion_generation_prompt = f"{confusion_prompt} \n\n {coding_question}"

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    confusions = model.generate_content(confusion_generation_prompt)

    print(confusions.text)
    print("\n\n")

    with open('help_generation_prompt.txt', 'r') as file:
        help_prompt = file.read()

    help_generation_prompt = f"{help_prompt} \n\n {coding_question} \n\n {confusions}"

    help = model.generate_content(help_generation_prompt)

    print(help.text)

    output_data = {
    'coding_question': coding_question,
    'confusions': confusions.text,
    'help': help.text
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"result_{timestamp}.json"

    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Response saved to {output_filename}")