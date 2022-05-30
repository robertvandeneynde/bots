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

async def on_message(update: Update, context: CallbackContext.DEFAULT_TYPE):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    msg = update.message.text
    print(msg)
    if msg.lower().startswith("hello"):
        await send("Hello ! :3")

async def caps(update: Update, context: CallbackContext):
    text_caps = str(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

class DatetimeText:
    days_english = "monday tuesday wednesday thursday friday saturday sunday".split() 
    days_french = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
    
    @classmethod
    def to_datetime_range(self, name, reference=None):
        from datetime import datetime, timedelta, date, time
        reference = reference or datetime.now()  # Datetime should be Brussels
        monday = datetime.combine(reference.date(), time(0))
        name = name.lower()
        if name in ("today", "auj", "aujourdhui", "aujourd'hui"):
            return monday
        while monday.isoweekday() != 1:
            monday -= timedelta(days=1)
        # TODO : add 2 letters for days
        if name in self.days_french:
            i = self.days_french.index(name)
        elif name in self.days_english:
            i = self.days_english.index(name)
        elif name in ("week", "semaine"):
            i = 0
        else:
            raise ValueError
        beg = monday + timedelta(days=i)
        if name in ("week", "semaine"):
            end = beg + timedelta(days=7)
        else:
            end = beg + timedelta(days=1)
        return beg, end
        
import sqlite3
async def add_event(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /addevent date name")
    date, *name = context.args
    name = " ".join(name)
    
    try:
        datetime, datetime_end = DatetimeText.to_datetime_range(date)
    except:
        return await send("An error occured in your command")
    
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        cursor.executescript("CREATE TABLE if not exists Events(date datetime, name text)")
        cursor.execute("INSERT INTO Events(date,name) VALUES (?,?)", (datetime, name))
    
    await send(f"Event {name!r} saved for date {datetime.date()} aka {date!r}")

async def list_events(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if len(context.args) >= 2:
        return await send("<when> must be a day of the week or week")
    
    if not context.args:
        when = "week"
    else:
        when, = context.args
    
    try:
        datetime, datetime_end = DatetimeText.to_datetime_range(when)
    except:
        return await send("An error occured in your command")
    
    beg, end = datetime, datetime_end
    
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        query = ("SELECT * FROM Events WHERE ? <= date AND date < ? ORDER BY date", (beg, end))
        
        def strptime(x:str):
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        def strftime(x:datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")
        
        msg = '\n'.join(f"{DatetimeText.days_english[strptime(date).weekday()]} {strptime(date).date():%d/%m}: {event}"
                        for date, event in cursor.execute(*query))
        await send(msg or "No events for that day !")
    
if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), on_message)
    application.add_handler(message_handler)
    
    application.add_handler(CommandHandler('caps', caps))
    application.add_handler(CommandHandler('addevent', add_event))
    application.add_handler(CommandHandler('listevents', list_events))
    
    application.run_polling()
