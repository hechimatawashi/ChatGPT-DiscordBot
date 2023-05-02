import discord
from discord import Intents
import openai

intents = Intents.all()
intents.members = True  # サーバーメンバーの取得に必要

client = discord.Client(intents=intents)
voice_client = None

# OpenAI APIの認証キーを設定します
with open("openai_key.txt", "r") as f:
    api_key = f.read().strip()

openai.api_key = api_key

# OpenAI APIにメッセージを送信して回答を取得する関数を定義します
async def get_openai_response(message):
    # Botのキャラ設定をテキストファイルで用意しておく
    with open("char_setting.txt", "r") as f:
        settings_text = f.read().strip()

    prompt = []
    system_message = {"role": "system", "content": settings_text}
    new_message = {"role": "user", "content": message.content}

    # リプライメッセージである場合
    if message.reference:
        # リプライの送信元メッセージを取得する
        # reply_message = await message.channel.fetch_message(message.reference.message_id)

        # リプライ履歴を取得する
        reply_history = await get_reply_history(message.channel, message, client.user)

        prompt += reply_history
    else:
        prompt.append(system_message)
        prompt.append(new_message)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=prompt,
            temperature=0,
            max_tokens=500
        )
        answer = response.choices[0].message.content

    except Exception as e:
        answer = e.message

    return answer

async def get_reply_history(channel,message, bot_user):
    with open("char_setting.txt", "r") as f:
        settings_text = f.read().strip()

    system_message = {"role": "system", "content": settings_text}
    # リプライチェインを格納する空の配列を作成する
    reply_chain = []

    # リプライチェインの最初のメッセージを追加する
    if message.author == bot_user:
        reply_chain.append({"role": "assistant", "content": message.content})
    else:
        reply_chain.append({"role": "user", "content": message.content})

    reply_chain.append(system_message)

    # リプライチェインに属する全てのメッセージを取得する
    while message.reference and len(reply_chain) < 8:
        reply_message = await channel.fetch_message(message.reference.message_id)
        message = reply_message

        # リプライチェインに属するメッセージを追加する
        if message.author == bot_user:
            reply_chain.append({"role": "assistant", "content": message.content})
        else:
            reply_chain.append({"role": "user", "content": message.content})

    # 取得したリプライチェインを逆順に並び替える（古いものから新しいものの順になるようにする）
    reply_chain.reverse()

    return reply_chain

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

async def on_ready():
    print("Logged in as")
    print(client.user.name)
    print(client.user.id)
    print("------")

# Discordボットを開始します
with open("discord_key.txt", "r") as f:
    discord_api_key = f.read().strip()
client.run(discord_api_key)
