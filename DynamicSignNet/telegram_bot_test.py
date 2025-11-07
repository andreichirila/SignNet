from telegram import Bot
import asyncio

TELEGRAM_BOT_TOKEN = '8327173184:AAGLA5pcLiAz-vMSVBq4tVJCHo7TPH3Zu8g'
CHAT_ID = '8541359800'

#Define bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)

async def run_bot(messages, chat_id):
    text = '\n'.join(messages)
    await send_message(text, chat_id)

#Test messages
messages = [
    'Product https://www.amazon.com/dp/B08C1W5N87, the price has changed from $24.99 to $26.99',
    'New negative review (rating 2) added for product https://www.amazon.com/dp/B0CL61F39H',
    'Attention! Average sales are 50% lower than usual over the last 3 hours!'
]

if messages:
    asyncio.run(run_bot(messages, CHAT_ID))