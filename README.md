# Planet Info Assistant (LangChain Reference Project)

This project demonstrates a planetary information assistant built using **LangChain**, **OpenAI**, and **ChromaDB**. It showcases key concepts of modern AI application development, including Tool Calling, Retrieval-Augmented Generation (RAG), and the LangChain Expression Language (LCEL).

## Core Concepts

### 1. Tool Calling
The assistant has access to specialized Python functions (tools) to retrieve specific data:
- `PlanetDistanceSun`: Provides the distance of a planet from the Sun.
- `PlanetRevolutionPeriod`: Provides the time it takes for a planet to orbit the Sun.
- `GeneralPlanetInfo`: Retrieves broader information using a RAG approach.

The LLM (GPT-4o or compatible) intelligently decides which tool to call based on the user's question.

### 2. Retrieval-Augmented Generation (RAG)
The `GeneralPlanetInfo` tool implements a basic RAG pipeline:
- **Documents**: Planet descriptions are stored in `planets/*.txt`.
- **Embeddings**: Uses `OpenAIEmbeddings` to convert text into vector representations.
- **Vector Store**: Uses `ChromaDB` (via `Chroma`) to store embeddings and perform similarity searches.
- **Search**: When asked about a planet, the tool finds the most relevant text snippet from the local files.

### 3. LangChain Expression Language (LCEL)
The application logic is orchestrated using LCEL, which allows for clean composition of different components:
```python
full_chain = prompt_template | llm_with_tools | execute_tools
```
- `prompt_template`: Formats the user input.
- `llm_with_tools`: The model bound with tool definitions.
- `execute_tools`: A custom `@chain` that manages the execution loop for tool calls.

### 4. Custom Execution Logic
The `execute_tools` function handles the iterative process of:
1. Receiving a tool call request from the LLM.
2. Invoking the tool.
3. Deciding whether to return the result directly (for factual tools) or feed it back to the LLM for further reasoning/summarization.

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root with the following:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: change if using a proxy
```

## Usage
Run the main script:
```bash
python task/main.py
```
Then enter a question like:
- *"How far is Jupiter from the Sun?"*
- *"Tell me something interesting about Mars."*

## Project Structure
- `task/main.py`: The main entry point containing tools and chain logic.
- `task/planets/`: Directory containing text files for RAG.
- `requirements.txt`: List of Python dependencies.
