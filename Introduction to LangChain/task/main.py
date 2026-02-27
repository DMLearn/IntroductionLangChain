# Import necessary libraries for environment variables, file paths, and LangChain integration
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables (API keys, base URLs) from .env file
load_dotenv()

# Tool 1: Retrieves distance of a planet from the Sun
@tool
def PlanetDistanceSun(planet_name: str) -> str:
    """Returns the distance of a planet from the Sun in Astronomical Units (AU).
    :rtype: str
    :param planet_name:
    :return:
    """
    planet_distances = {
        "Earth": "Earth is approximately 1 AU from the Sun.",
        "Mars": "Mars is approximately 1.5 AU from the Sun.",
        "Jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "Pluto": "Pluto is approximately 39.5 AU from the Sun."
    }

    return planet_distances.get(
        planet_name,
        f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    )


# Tool 2: Retrieves revolution period of a planet
@tool
def PlanetRevolutionPeriod(planet_name: str) -> str:
    """Returns the revolution period of a planet around the Sun in Earth years.
    :rtype: str
    :param planet_name:
    :return:
    """
    planet_periods = {
        "Earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "Mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "Jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "Pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun."
    }

    return planet_periods.get(
        planet_name,
        f"Information about the revolution period of {planet_name} is not available in this tool."
    )


# Tool 3: Performs a RAG-based search for general planet information
@tool
def PlanetGeneralInfo(planet_name: str) -> str:
    """Returns general information about a planet by performing similarity search over planet documents.
    :rtype: str
    :param planet_name:
    :return:
    """
    planets_dir = Path("planets")

    # Ensure the 'planets' directory exists
    if not planets_dir.exists():
        return f"No information available about {planet_name}."

    # Read planet data from .txt files in the directory
    planet_data = {}
    for file_path in planets_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            planet_data[file_path.stem] = file.read().strip()

    if not planet_data:
        return f"No information available about {planet_name}."

    # Convert planet content into LangChain Document objects
    documents = [
        Document(page_content=content, metadata={"planet": name})
        for name, content in planet_data.items()
    ]

    # Initialize Chroma vector store with OpenAI embeddings for similarity search
    client = chromadb.Client()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        ),
        client=client,
        collection_name="planets"
    )

    # Search for the most relevant document based on planet name
    results = vectorstore.similarity_search(planet_name, k=1)

    if results:
        return results[0].page_content
    else:
        return f"Additional information for {planet_name} is not available in this tool."


# Initialize the LLM (ChatOpenAI) using environment configuration
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Component 1: Prompt formatting chain
# Defines the system behavior and constraints for the assistant
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant that answers questions about planets. "
    "Use the provided tools to find accurate information. "
    "If you need to find a planet with a specific property (like distance or revolution period), "
    "you MUST check the specialized tools (PlanetDistanceSun or PlanetRevolutionPeriod) "
    "for multiple candidate planets (e.g., Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto) "
    "until you find the correct one. "
    "Do NOT rely on your internal knowledge or PlanetGeneralInfo for exact distances or periods.\n\n"
    "{question}"
)

# Component 2: Model chain bound with tools
# Attaches the tools to the LLM so it can decide which one to call
llm_with_tools = llm.bind_tools([PlanetGeneralInfo, PlanetDistanceSun, PlanetRevolutionPeriod])

# Component 3: Tool execution chain
# Processes the LLM's response, executes tools, and manages multi-turn tool logic
@chain
def execute_tools(model_response):
    """Executes tool calls from the model response and returns the final result."""
    tools = {
        "PlanetDistanceSun": PlanetDistanceSun,
        "PlanetRevolutionPeriod": PlanetRevolutionPeriod,
        "PlanetGeneralInfo": PlanetGeneralInfo
    }

    messages = []
    if isinstance(model_response, list):
        messages = model_response
    else:
        messages = [model_response]

    # Loop to handle iterative tool calling until a final answer is produced
    while True:
        last_message = messages[-1]

        # Stop if no more tool calls are requested by the model
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            print(last_message.content)
            return last_message.content

        # Check if all tool calls refer to known factual tools
        all_factual = all(tool_call["name"] in tools for tool_call in last_message.tool_calls)

        tool_results = []
        # Process each tool call request
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_to_call = tools.get(tool_name)

            if tool_to_call:
                print(tool_call)
                # Execute the tool and capture its result
                result = tool_to_call.invoke(tool_args)
                tool_results.append(str(result))
                # Add the result to the conversation history as a ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            else:
                error_msg = f"Error: Tool {tool_name} not found."
                tool_results.append(error_msg)
                messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))

        # If tools provide raw facts, return them directly to the user
        if all_factual:
            combined_result = "\n".join(tool_results)
            print(combined_result)
            return combined_result

        # Feed the tool results back to the LLM for further reasoning if needed
        response = llm_with_tools.invoke(messages)
        messages.append(response)


# Compose the full chain using LangChain Expression Language (LCEL)
# prompt | model | execution logic
full_chain = prompt_template | llm_with_tools | execute_tools

if __name__ == "__main__":
    # Get user input and start the process
    user_question = input("Enter your question: ")

    # Invoke the composed chain of three runnables
    result = full_chain.invoke({"question": user_question})
    print(full_chain)
