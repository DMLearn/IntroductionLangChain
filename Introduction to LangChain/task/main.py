# Write your solution below
# install the required packages:
# pip install langchain-core python-dotenv langchain-openai

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
import os

dotenv.load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), model="gpt-4o-mini")  # if using a different base_url and model, pass them as well

question = input()
template = PromptTemplate.from_template(
    "You are a helpful assistant who answers questions users may have. You are asked: {question}.")

prompt = template.invoke({"question": question})
response = llm.invoke(prompt)
print(response.content)