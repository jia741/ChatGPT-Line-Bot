from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
)
import os
import uuid

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# 使用固定的 OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("请在环境变量中设置 OPENAI_API_KEY")

model = OpenAIModel(api_key=OPENAI_API_KEY)

youtube = Youtube(step=4)
website = Website()
memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'), memory_message_count=2)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')

    try:
        # 定义支持的命令列表
        commands = ['/指令說明', '/系統訊息', '/清除', '/圖像', '/GPT']

        # 检查消息是否以支持的命令开头
        if any(text.startswith(cmd) for cmd in commands):
            if text.startswith('/指令說明'):
                msg = TextSendMessage(text="指令：\n/系統訊息 + Prompt\n👉 Prompt 可以命令機器人扮演某個角色，例如：請你扮演擅長做總結的人\n\n/清除\n👉 當前每一次都會紀錄最後兩筆歷史紀錄，這個指令能夠清除歷史訊息\n\n/圖像 + Prompt\n👉 會調用 DALL∙E 2 Model，以文字生成圖像\n\n/GPT + Prompt\n👉 調用 ChatGPT 以文字回覆\n\n語音輸入\n👉 會調用 Whisper 模型，先將語音轉換成文字，再調用 ChatGPT 以文字回覆")

            elif text.startswith('/系統訊息'):
                memory.change_system_message(user_id, text[5:].strip())
                msg = TextSendMessage(text='輸入成功')

            elif text.startswith('/清除'):
                memory.remove(user_id)
                msg = TextSendMessage(text='歷史訊息清除成功')

            elif text.startswith('/圖像'):
                prompt = text[3:].strip()
                memory.append(user_id, 'user', prompt)
                is_successful, response, error_message = model.image_generations(prompt)
                if not is_successful:
                    raise Exception(error_message)
                url = response['data'][0]['url']
                msg = ImageSendMessage(
                    original_content_url=url,
                    preview_image_url=url
                )
                memory.append(user_id, 'assistant', url)

            elif text.startswith('/GPT'):
                prompt = text[4:].strip()
                if not prompt:
                    msg = TextSendMessage(text='請在 /GPT 指令後提供內容。')
                else:
                    memory.append(user_id, 'user', prompt)
                    url = website.get_url_from_text(prompt)
                    if url:
                        if youtube.retrieve_video_id(prompt):
                            is_successful, chunks, error_message = youtube.get_transcript_chunks(youtube.retrieve_video_id(prompt))
                            if not is_successful:
                                raise Exception(error_message)
                            youtube_transcript_reader = YoutubeTranscriptReader(model, os.getenv('OPENAI_MODEL_ENGINE'))
                            is_successful, response, error_message = youtube_transcript_reader.summarize(chunks)
                            if not is_successful:
                                raise Exception(error_message)
                            role, response = get_role_and_content(response)
                            msg = TextSendMessage(text=response)
                        else:
                            chunks = website.get_content_from_url(url)
                            if len(chunks) == 0:
                                raise Exception('無法撈取此網站文字')
                            website_reader = WebsiteReader(model, os.getenv('OPENAI_MODEL_ENGINE'))
                            is_successful, response, error_message = website_reader.summarize(chunks)
                            if not is_successful:
                                raise Exception(error_message)
                            role, response = get_role_and_content(response)
                            msg = TextSendMessage(text=response)
                    else:
                        is_successful, response, error_message = model.chat_completions(memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'))
                        if not is_successful:
                            raise Exception(error_message)
                        role, response = get_role_and_content(response)
                        msg = TextSendMessage(text=response)
                    memory.append(user_id, role, response)
        else:
            # 如果消息不以任何已知的命令开头，则不处理
            return

    except Exception as e:
        memory.remove(user_id)
        if str(e).startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='OpenAI API Key 有誤，請檢查配置。')
        elif str(e).startswith('That model is currently overloaded with other requests.'):
            msg = TextSendMessage(text='已超過負荷，請稍後再試')
        else:
            msg = TextSendMessage(text=str(e))
    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    user_id = event.source.user_id
    audio_content = line_bot_api.get_message_content(event.message.id)
    input_audio_path = f'{str(uuid.uuid4())}.m4a'
    with open(input_audio_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)

    try:
        is_successful, response, error_message = model.audio_transcriptions(input_audio_path, 'whisper-1')
        if not is_successful:
            raise Exception(error_message)
        memory.append(user_id, 'user', response['text'])
        is_successful, response, error_message = model.chat_completions(memory.get(user_id), 'gpt-3.5-turbo')
        if not is_successful:
            raise Exception(error_message)
        role, response = get_role_and_content(response)
        memory.append(user_id, role, response)
        msg = TextSendMessage(text=response)
    except Exception as e:
        memory.remove(user_id)
        if str(e).startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='OpenAI API Key 有誤，請檢查配置。')
        else:
            msg = TextSendMessage(text=str(e))
    os.remove(input_audio_path)
    line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
    return 'Hello World'


if __name__ == "__main__":
    if os.getenv('USE_MONGO'):
        mongodb.connect_to_database()
        storage = Storage(MongoStorage(mongodb.db))
    else:
        storage = Storage(FileStorage('db.json'))
    app.run(host='0.0.0.0', port=8080)
