# Write your solution below

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import chromadb
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()


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


@tool
def GeneralPlanetInfo(planet_name: str) -> str:
    """Returns general information about a planet by performing similarity search over planet documents.
    :rtype: str
    :param planet_name:
    :return:
    """
    planets_dir = Path("planets")

    if not planets_dir.exists():
        return f"No information available about {planet_name}."

    planet_data = {}
    for file_path in planets_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            planet_data[file_path.stem] = file.read().strip()

    if not planet_data:
        return f"No information available about {planet_name}."

    documents = [
        Document(page_content=content, metadata={"planet": name})
        for name, content in planet_data.items()
    ]

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

    results = vectorstore.similarity_search(planet_name, k=1)

    if results:
        return results[0].page_content
    else:
        return f"Additional information for {planet_name} is not available in this tool."


llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_with_tools = llm.bind_tools([PlanetDistanceSun, PlanetRevolutionPeriod, GeneralPlanetInfo])

if __name__ == "__main__":
    user_question = input("Enter your question: ")

    messages = [HumanMessage(content=user_question)]
    tools = {
        "PlanetDistanceSun": PlanetDistanceSun,
        "PlanetRevolutionPeriod": PlanetRevolutionPeriod,
        "GeneralPlanetInfo": GeneralPlanetInfo
    }

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(response.content)
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_to_call = tools.get(tool_name)

            if tool_to_call:
                print(response.tool_calls[0])
                result = tool_to_call.invoke(tool_args)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            else:
                messages.append(ToolMessage(content=f"Error: Tool {tool_name} not found.", tool_call_id=tool_call["id"]))
