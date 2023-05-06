import discord
import openai
import os
import datetime
import pinecone

from discord import Intents
from dotenv import load_dotenv

from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationChain
from langchain.memory import (
    VectorStoreRetrieverMemory,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.agents import (
    Tool,
    AgentType,
    initialize_agent
)
from langchain.utilities import (
    GoogleSearchAPIWrapper
)

intents = Intents.all()
intents.members = True  # サーバーメンバーの取得に必要

client = discord.Client(intents=intents)
voice_client = None

load_dotenv()

DISCORD_API_KEY = os.getenv('DISCORD_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

openai.api_key = OPENAI_API_KEY


# Botのキャラ設定をテキストファイルで用意しておく
with open("char_setting.txt", "r") as f:
    settings_text = f.read().strip()

# 要約用テンプレートの読み込み
with open("summary_template.txt", "r") as f:
    summary_template = f.read().strip()

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"], template=settings_text
)

# summary_prompt = PromptTemplate(
#     input_variables=["summary", "new_lines"], template=summary_template
# )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    max_tokens=200,
    temperature=0.4
)

# llm = ChatOpenAI(
#     model_name="gpt-4",
#     max_tokens=200,
#     temperature=0.4
# )

# summarize_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# vectorstore
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone(pinecone_index, embeddings.embed_query, "text")

retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)


search = GoogleSearchAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="最新の話題について答える場合に利用することができます。また、今日の気温、天気、為替レートなど現在の状況についても確認することができます。入力は検索内容です。回答は日本語で行います。"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="計算をする場合に利用することができます。"
    )
]


# 会話チェーン
conversation = ConversationChain(memory=memory, prompt=conversation_prompt, llm=llm, verbose=True)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory, prompt=conversation_prompt)

# OpenAI APIにメッセージを送信して回答を取得する関数を定義します
async def get_openai_response(message):
    dt_now = datetime.datetime.now()
    prefix = "[now:" + dt_now.strftime('%Y/%m/%d %H:%M:%S') + "]"
    input = prefix + message.content

    # 明示的に検索を指示した場合のみagentを使って回答を得る
    if "検索して" in message.content:
        response = agent_chain.run(input=input)
    else:
        response = conversation.predict(input=input)

    return response

# ボットがメッセージを受信したときに呼び出される関数を定義します
@client.event
async def on_message(message):
    # メッセージを送信したユーザーがボット自身である場合、何もしません
    if message.author == client.user:
        return

    else:
        # メッセージをOpenAI APIに送信して、回答を取得します
        answer = await get_openai_response(message)

        if len(answer) > 2000:
            messages = [answer[i:i+1999] for i in range(0, len(answer), 1999)]
            for msg in messages:
                await message.reply(msg)

        else:
            # 回答をチャットに送信します
            await message.reply(answer)

# Discordボットを開始します
client.run(DISCORD_API_KEY)
