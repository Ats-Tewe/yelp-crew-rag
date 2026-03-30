import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from workspace root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=True)

# Bypass CrewAI's forced OpenAI key check
os.environ["OPENAI_API_KEY"] = "NA"

from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import JSONSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# --- LLM: Ollama phi3 locally ---
llm = LLM(model="ollama/phi3", base_url="http://localhost:11434")

# --- Knowledge Source: Yelp field glossary shared by all agents ---
with open("docs/Yelp_Data_Translation.md", "r", encoding="utf-8") as f:
    yelp_md = f.read()

knowledge_source = StringKnowledgeSource(content=yelp_md)
# --- RAG config: instructor's exact format ---
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# --- RAG Tools: unique name, description AND collection_name per tool ---
user_rag_tool = JSONSearchTool(
    json_path='data/user_subset.json',
    collection_name='v3_hf_user_data',
    config=rag_config
)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = "Retrieve a specific user's giving habits, average stars, and review counts by user_id."

review_rag_tool = JSONSearchTool(
    json_path='data/review_subset.json',
    collection_name='v3_hf_review_data',
    config=rag_config
)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = "Retrieve past review texts written by a specific user or about a specific business."

item_rag_tool = JSONSearchTool(
    json_path='data/item_subset.json',
    collection_name='v3_hf_item_data',
    config=rag_config
)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = "Retrieve a specific restaurant's location, categories, attributes, and overall stars by item_id."


@CrewBase
class FirstCrew():
    """Yelp Review Prediction Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def user_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config["user_profiler"],
            tools=[user_rag_tool, review_rag_tool],
            llm=llm,
            verbose=True
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],
            tools=[item_rag_tool],
            llm=llm,
            verbose=True
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],
            llm=llm,
            verbose=True
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_user_task"])

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_item_task"])

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],
            output_file="report.json"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[knowledge_source],
            verbose=True
        )