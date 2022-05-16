import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler
from telegram.ext import filters
from telegram_settings_local import TOKEN

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: CallbackContext.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot, please talk to me!")
    print("Someone started me!")

async def message(update: Update, context: CallbackContext.DEFAULT_TYPE):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    msg = update.message.text
    print(msg)
    if msg.lower().startswith("hello"):
        await send("Hello ! :3")

async def caps(update: Update, context: CallbackContext):
    text_caps = str(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), message)
    application.add_handler(message_handler)
    
    application.add_handler(CommandHandler('caps', caps))
    
    application.run_polling()
