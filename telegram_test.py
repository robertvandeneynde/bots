import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler, ContextTypes
from telegram.ext import filters
from telegram_settings_local import TOKEN

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.getLogger('httpx').setLevel(logging.WARN)

async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot, please talk to me!")
    print("Someone started me!")

async def on_message(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    msg = update.message.text
    print(msg)
    if msg.lower().startswith("hello"):
        await send("Hello ! :3")

async def caps(update: Update, context: CallbackContext):
    text_caps = str(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

async def ru(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /ru word1 word2 word3...")
    a = "azertyuiopqsdfghjklmwxcvbn"
    b = "азертыуиопясдфгхйклмжьцвбн"
    c = "sh shch ch ye yu zh ya yo".split()
    d = "ш  щ    ч  э  ю  ж  я  ё".split()
    D = dict(zip(a,b)) | dict(zip(c,d)) | dict(zip(a.upper(), b.upper())) | dict(zip((c.upper() for c in c), (d.upper() for d in d)))
    S = sorted(D.items(), reverse=True)
    import re
    R = re.compile('|'.join(re.escape(s[0]) for s in S))
    def to_cyrilic(word):
        #return ''.join(map(lambda x: D.get(x,x), word))
        return R.sub(lambda m: (lambda x: D.get(x,x))(m.group(0)), word)
    await send(' '.join(to_cyrilic(word) for word in context.args))

async def wikt(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /wikt word1 word2 word3... /Language")
    
    # See the presence of the /Language arg (last argument)
    if context.args[-1].startswith('/'):
        language = context.args[-1][1:]
        words = context.args[:-1]
    else:
        language = ''
        words = context.args[:]
    
    if ':' in language:
        base_lang, target_lang, *_ = language.split(':')
    else:
        base_lang, target_lang = 'en', language
    def url(x):
        x = x.lower()
        return 'https://wiktionary.com/wiki/{}:{}'.format(base_lang, x) + ('#' + target_lang if target_lang != '' else '')
    return await send('\n\n'.join(url(x) for x in words))

class DatetimeText:
    days_english = "monday tuesday wednesday thursday friday saturday sunday".split() 
    days_french = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
    
    @classmethod
    def to_datetime_range(self, name, reference=None):
        from datetime import datetime, timedelta, date, time
        reference = reference or datetime.now()  # Datetime should be Brussels
        today = datetime.combine(reference.date(), time(0))
        name = name.lower()
        if name in ("today", "auj", "aujourdhui", "aujourd'hui"):
            return today, today + timedelta(days=1)
        
        if name in ("week", "semaine"):
            beg = today
            end = today + timedelta(days=7)
            return beg, end
        
        if name in ("tomorrow", "demain"):
            return today + timedelta(days=1), today + timedelta(days=2)
        
        # TODO : add few letters for days
        if name in self.days_french:
            i = self.days_french.index(name)
        elif name in self.days_english:
            i = self.days_english.index(name)
        else:
            raise ValueError(f"Unknown date {name}")
        
        the_day = today + timedelta(days=1)
        while the_day.weekday() != i:
            the_day += timedelta(days=1)
        
        beg = the_day
        end = beg + timedelta(days=1)
        return beg, end
        
import sqlite3
async def add_event(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /addevent date name")
    date, *name = context.args
    name = " ".join(name)
    
    datetime, datetime_end = DatetimeText.to_datetime_range(date)
    
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        cursor.executescript("CREATE TABLE if not exists Events(date datetime, name text)")
        cursor.execute("INSERT INTO Events(date,name) VALUES (?,?)", (datetime, name))
    
    await send(f"Event {name!r} saved for date {datetime.date()} aka {date!r}")

async def list_events(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if len(context.args) >= 2:
        return await send("<when> must be a day of the week or week")
    
    if not context.args:
        when = "week"
    else:
        when, = context.args
    
    beg, end = DatetimeText.to_datetime_range(when)
    
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        query = ("SELECT * FROM Events WHERE ? <= date AND date < ? ORDER BY date", (beg, end))
        
        from datetime import datetime, timedelta

        def strptime(x:str):
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        def strftime(x:datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")
        
        msg = '\n'.join(f"{DatetimeText.days_english[strptime(date).weekday()]} {strptime(date).date():%d/%m}: {event}"
                        for date, event in cursor.execute(*query))
        await send(msg or "No events for that day !")

def make_help(*commands):
    async def help(update, context):
        async def send(m):
            await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
        if context.args:
           raise Exception("No arguments")
        return await send('\n'.join(map("/{}".format, commands)))
    return help

async def general_error_callback(update:Update, context:CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    logging.error("Error", exc_info=context.error)
    return await send("An error occured in your command")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), on_message)
    application.add_handler(message_handler)
    
    application.add_handler(CommandHandler('caps', caps))
    application.add_handler(CommandHandler('addevent', add_event))
    application.add_handler(CommandHandler('listevents', list_events))
    application.add_handler(CommandHandler('ru', ru))
    application.add_handler(CommandHandler('wikt', wikt))
    application.add_handler(CommandHandler('help', make_help('caps', 'addevent', 'listevents', 'ru', 'wikt')))

    application.add_error_handler(general_error_callback)
    
    application.run_polling()
