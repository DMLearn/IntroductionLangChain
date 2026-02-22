# Write your solution below
# install the required packages:
# pip install langchain-core python-dotenv langchain-openai

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()

# Read planet files and store in dictionary
planets_folder = Path("planets")
planet_data = {}
for file_path in planets_folder.glob("*.txt"):
    planet_name = file_path.stem
    with open(file_path, 'r', encoding='utf-8') as f:
        planet_data[planet_name] = f.read().strip()

# Create examples from loaded planet data
examples = [{"input": name, "output": content} for name, content in planet_data.items()]

example_template = PromptTemplate.from_template("Q: {input}\nA: {output}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    suffix="Question: {question}\nAnswer:",
    input_variables=["question"]
)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), model="gpt-4o-mini")  # if using a different base_url and model, pass them as well

final_prompt = few_shot_prompt.format(question=input())
response = llm.invoke(final_prompt)
print(response.content)