from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# 환경변수(.env 파일) 로드 - API 키 등을 불러오기 위함
load_dotenv()

def main():
    print("Hello from langchain-course!")
    
    # 참고 자료
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left, growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    
    """
    
    # LLM에게 보낼 프롬프트 템플릿 정의
    # 주어진 정보를 바탕으로 요약과 흥미로운 사실 2개를 요청
    summary_template = f"""
    Given the information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about the them
    """
    
    # PromptTemplate 객체 생성 (LangChain의 핵심 구성요소)
    # 프롬프트 템플릿을 사용하여 입력 변수와 템플릿을 연결
    summary_prompt_template = PromptTemplate(
        input_variables = ["information"],
        template = summary_template
    )
    
    # LLM 모델 초기화
    # 온도 설정(0은 최대한 정확한 답변, 1은 더 다양한 답변)
    llm = ChatOpenAI(temperature=0, model = "gpt-5")  
    #llm = ChatOllama(temperature=0, model = "gemma3:270m")
    
    # 프롬프트 템플릿과 LLM 모델을 연결하여 체인 생성
    chain = summary_prompt_template | llm
    
    # 체인 실행
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
