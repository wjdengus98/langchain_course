# React Search Agent - 구조화된 출력을 생성하는 AI 검색 에이전트
'''
version 2 - structed ouput 사용
with_structured_output()이란?
- 주어진 스키마에 맞게 출력을 구조화하는 함수
- 출력을 파싱하고 검증하는 복잡한 과정을 간소화
- 예측 결과를 명확하게 정의된 형식으로 제공
'''

# 환경 변수 로드 (API 키 등)
from dotenv import load_dotenv

load_dotenv()

# LangChain 핵심 모듈들 임포트
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# 사용자 정의 스키마 임포트
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS # 검색 에이전트의 프롬프트
from schemas import AgentResponse # 검색 결과의 스키마

# 검색 도구 초기화
tools = [TavilySearch()] # 웹 검색을 위한 Tavily API 도구

# 모델 초기화
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="") # format_instructions를 빈 문자열로 설정 (structured_llm 사용으로 불필요)

# 검색 에이전트 생성
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
# 에이전트 실행기 생성 (에이전트를 실제로 실행하고 관리)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 검색 결과 추출 및 구조화
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm

# 메인 함수 정의
def main():
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
