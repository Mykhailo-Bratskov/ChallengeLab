import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import UsageMetadataCallbackHandler

# load api key from the .env fike
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY was not loaded. Check your .env file.")

#format the LLM output for the better precision
class ExperimentSpec(BaseModel):
    name: str
    objective: str
    feature_groups: list[str]
    model: str
    validation: str
    metric: str
    priority: Literal["low", "medium", "high"] = "medium"
    notes: list[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    summary: str
    metric: str
    validation: str
    experiments: list[ExperimentSpec]
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)

parser = PydanticOutputParser(pydantic_object=PlanResponse)

# create an executable plan from research findings to code implementation
def create_plan(challenge_brief: str, dataset_profile: str, research_findings: str,):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Machine Learning Planning Agent.

Transform the provided research into a concise, actionable implementation plan
for a Machine Learning Engineer.

Do not write code.
Do not invent unavailable data.
Return only the required structured output.

{format_instructions}
"""
            ),
            (
                "human",
                """
Research findings:
{research_findings}

Dataset Profile: 
{dataset_profile}

Challenge Rules: 
{challenge_brief}
Create the implementation plan.
"""
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    usage_callback = UsageMetadataCallbackHandler()
    result = chain.invoke(
        {
            "challenge_brief": challenge_brief,
            "dataset_profile": dataset_profile,
            "research_findings": research_findings,
        },
        config={"callbacks": [usage_callback]},
    )
    
    print("--- Planner Agent Token Usage ---")
    print(f"Total Tokens: {usage_callback.usage_metadata}")

    return result.model_dump_json(indent=2), usage_callback.usage_metadata
    
