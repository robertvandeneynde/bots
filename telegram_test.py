import logging
from telegram import Update, Message, Chat
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler, ContextTypes
from telegram.ext import filters
from telegram.constants import ChatType
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

def strip_botname(update: Update, context: CallbackContext):
    # TODO analyse message.entities
    bot_mention: str = '@' + context.bot.username
    if update.message.text.startswith(bot_mention):
        return update.message.text[len(bot_mention):].strip()
    return update.message.text.strip()

async def on_message(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    
    if update.message:
        logging.info("@{username}: {text} (In {group})".format(
            username=update.message.from_user.username,
            text=update.message.text,
            group='private' if update.message.chat.type == ChatType.PRIVATE else
                   "'{}'".format(update.message.chat.title) if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL) else
                   update.message.chat.type))
    else:
        logging.info("{}".format(update))

    if update.message:
        msg = strip_botname(update, context)
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

def get_or_empty(L: list, i:int) -> str | object:
    try:
        return L[i]
    except IndexError:
        return ''

async def wikt(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        if not update.message.reply_to_message:
            return await send("Usage: /wikt word1 word2 word3... /Language")
    
    words = []
    if update.message.reply_to_message:
        words += update.message.reply_to_message.text.split()
        
    if get_or_empty(context.args, -1).startswith('/'):
        language = context.args[-1][1:]
        words += context.args[:-1]
    else:
        language = ''
        words += context.args[:]
    
    if ':' in language:
        base_lang, target_lang, *_ = language.split(':')
    else:
        base_lang, target_lang = '', language
    target_lang
    def url(x):
        x = x.lower()
        return (
            'https://wiktionary.com/wiki/'
            + ('{}:'.format(base_lang) if base_lang else '')
            + x
            + ('#' + target_lang if target_lang else '')
        )
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

from decimal import Decimal
ONE_EURO_IN_BRL = Decimal("5.36")

def make_money_command(name:str, currency_base:str, currency_converted:str, rate:Decimal):
    async def money(update: Update, context: CallbackContext):
        async def send(m):
            await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
        from decimal import Decimal
        if not context.args:
            return await send(f"Usage: /{name} value")
        value, *_ = context.args
        amount_base = Decimal(value)
        amount_converted = amount_base * rate
        return await send('\n'.join([
            "{}: {:.2f}".format(currency_base.upper(), amount_base),
            "{}: {:.2f}".format(currency_converted.upper(), amount_converted),
        ]))
    return money

eur = make_money_command("eur", "eur", "brl", ONE_EURO_IN_BRL)
brl = make_money_command("brl", "brl", "eur", 1 / ONE_EURO_IN_BRL)

async def help(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    fmt = ('{} - {}' if '--botfather' in context.args else
           '/{} {}')
    
    return await send('\n'.join(fmt.format(command, COMMAND_DESC.get(command, command)) for command in COMMAND_LIST))

async def general_error_callback(update:Update, context:CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    logging.error("Error", exc_info=context.error)
    return await send("An error occured in your command")

COMMAND_DESC = {
    "help": "Help !",
    "caps": "Returns the list of parameters in capital letters",
    "addevent": "Add event",
    "listevents": "List events",
    "ru": "Latin alphabet to Cyrillic",
    "wikt": "Shows definition of each word",
    'eur': "Convert euros to other currencies",
    'brl': "Convert brazilian reals to other currencies",
}

COMMAND_LIST = ('caps', 'addevent', 'listevents', 'ru', 'wikt', 'eur', 'brl', 'help')

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
    application.add_handler(CommandHandler('eur', eur))
    application.add_handler(CommandHandler('brl', brl))
    application.add_handler(CommandHandler('help', help))

    application.add_error_handler(general_error_callback)
    
    application.run_polling()
