# Autogen
# RetrieveChat with AutoGen 0.2 (Work in Progress)

This repository demonstrates how to use **RetrieveChat**—part of the AutoGen framework—for **retrieval-augmented code generation and question answering (QA)**. RetrieveChat enables agents to leverage external documentation via vector search to produce answers that go beyond the model’s training data.

> **Note:** This project is still under development. Core RetrieveChat functionality is set up, and additional features (like improved doc ingestion and error handling) are on the to-do list.

---

##  What’s Inside

- Jupyter notebook integrating **RetrieveChat** with **Qdrant** (a vector search engine) for code generation and QA.
- Agent setup using `AssistantAgent` and `RetrieveUserProxyAgent`.
- Demonstrations of retrieval-augmented workflows, including context updates and human feedback loops.
- Dependency setup and configuration instructions.

---

##  Example Agent Setup (`notebook` snippet)

```python
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

sentence_transformer_ef = SentenceTransformer("all-distilroberta-v1").encode
client = QdrantClient(url="http://localhost:6333/")

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "code",
        "docs_path": ["path/to/docs.md"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "qdrant",
        "db_config": {"client": client},
        "get_or_create": True,
        "embedding_function": sentence_transformer_ef,
    },
    code_execution_config=False,
)
