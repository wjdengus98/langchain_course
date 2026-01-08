from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """웹 검색 출처의 URL을 담는 데이터 클래스
    Agent가 정보를 찾은 웹사이트 주소를 저장합니다.
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """ Agent의 최종 응답을 구조화한 데이터 클래스"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# Agent 객체 생성 - model, tool, 구조화된 응답
llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

# Agent를 살행하고 결과를를 출력하는 메인 함수.
def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()