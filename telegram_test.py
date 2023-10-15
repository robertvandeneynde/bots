from __future__ import annotations
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

import re
MONEY_CURRENCIES_ALIAS = {
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "€": "eur",
    "brl": "brl",
    "real": "brl",
    "reais": "brl",
}
MONEY_RE = re.compile('(\\d+) ?(' + '|'.join(map(re.escape, MONEY_CURRENCIES_ALIAS)) + ')', re.I)

def strip_botname(update: Update, context: CallbackContext):
    # TODO analyse message.entities with message.parse_entity and message.parse_entities
    bot_mention: str = '@' + context.bot.username
    if update.message.text.startswith(bot_mention):
        return update.message.text[len(bot_mention):].strip()
    return update.message.text.strip()

async def hello_responder(msg:str, send:'async def'):
    if msg.lower().startswith("hello"):
        await send("Hello ! :3")

def detect_currencies(msg: str):
    return [(value, MONEY_CURRENCIES_ALIAS[currency_raw.lower()]) for value, currency_raw in MONEY_RE.findall(msg)]

async def money_responder(msg:str, send:'async def'):
    for value, currency in detect_currencies(msg):
        currency_base = currency
        currency_converted = list(filter(lambda x: x != currency, ['eur', 'brl']))[0]
        if currency == 'eur':
            rate = ONE_EURO_IN_BRL
        elif currency == 'brl':
            rate = 1 / ONE_EURO_IN_BRL
        else:
            raise ValueError("Unknown currency: '{}'".format(currency))

        amount_base = Decimal(value)
        amount_converted = amount_base * rate
        await send(format_currency(currency_base=currency_base, amount_base=amount_base, currency_converted=currency_converted, amount_converted=amount_converted))

RESPONDERS = (hello_responder, money_responder)

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

        for responder in RESPONDERS:
            try:
                await responder(msg, send)
            except Exception as e:
                await log_error(e, send)

    if update.edited_message:
        pass

async def caps(update: Update, context: CallbackContext):
    text_caps = str(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

async def ru(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /ru word1 word2 word3...")
    d1 = ("azertyuiopqsdfghjklmwxcvbn",
          "азертыуиопясдфгхйклмвхцвбн")
    d2 = ("sh shch ch ye yu zh ya yo".split(),
          "ш  щ    ч  э  ю  ж  я  ё".split())
    d3 = ("' ''".split(), 
          'ь ъ'.split())
    D = (dict(zip(*d1))
       | dict(zip(*d2))
       | dict(zip(*map(str.upper, d1)))
       | dict(zip(map(str.upper, d2[0]), map(str.upper, d2[1])))
       | dict(zip(*d3))
       | dict(zip(map(str.capitalize, d2[0]), map(str.upper, d2[1]))))
    S = sorted(D, key=len, reverse=True)
    import re
    R = re.compile('|'.join(map(re.escape, S)))
    def to_cyrilic(word):
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
    def read_my_settings(key):
        return read_settings(key, user_id=update.message.from_user.id)

    if not context.args:
        if not update.message.reply_to_message:
            return await send("Usage: /wikt word1 word2 word3... /Language")
    
    words = []
    if update.message.reply_to_message:
        words += update.message.reply_to_message.text.split()
        
    if get_or_empty(context.args, -1).startswith('/'):
        language = context.args[-1][1:]
        words += context.args[:-1]
        if ':' in language:
            base_lang, target_lang, *_ = language.split(':')
        else:
            base_lang, target_lang = '', language
    else:
        words += context.args[:]
        base_lang = read_my_settings('wikt.description')
        target_lang = read_my_settings('wikt.text')
    
    base_lang, target_lang

    def url(x):
        x = x.lower()
        return (
            'https://wiktionary.com/wiki/'
            + ('{}:'.format(base_lang) if base_lang else '')
            + x
            + ('#{}'.format(target_lang) if target_lang else '')
        )
    return await send('\n\n'.join(url(x) for x in words))

import zoneinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

class DatetimeText:
    days_english = "monday tuesday wednesday thursday friday saturday sunday".split() 
    days_french = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
    
    @classmethod
    def to_date_range(self, name, *, reference=None, tz=None) -> date:
        from datetime import datetime, timedelta, date
        reference = reference or datetime.now().astimezone(ZoneInfo("Europe/Brussels") if tz is None else tz).replace(tzinfo=None)
        today = reference.date()
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
            raise UserError(f"Unknown date {name}")
        
        the_day = today + timedelta(days=1)
        while the_day.weekday() != i:
            the_day += timedelta(days=1)
        
        beg = the_day
        end = beg + timedelta(days=1)
        return beg, end

def parse_event(args) -> (str, time | None, str):
    from datetime import date as Date, time as Time
    
    date, *rest = args
    if match := re.compile('(\\d{1,2})[:hH](\\d{2})?').fullmatch(get_or_empty(rest, 0)):
        hours, minutes = match.group(1), match.group(2)
        time = Time(int(hours), int(minutes or '0'))
        rest = rest[1:]
    else:
        time = None
    name = " ".join(rest)
    return date, time, name

import sqlite3
async def add_event(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if not context.args:
        return await send("Usage: /addevent date name")
    from datetime import datetime as Datetime, time as Time, date as Date, timedelta
    
    source_user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    date_str, time, name = parse_event(context.args)

    tz = get_my_timezone(update.message.from_user.id)
    
    date, date_end = DatetimeText.to_date_range(date_str, tz=tz)
    datetime = Datetime.combine(date, time or Time(0,0)).replace(tzinfo=tz)

    datetime_utc = datetime.astimezone(ZoneInfo('UTC'))

    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()

        def strftime(x:datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("INSERT INTO Events(date, name, chat_id, source_user_id) VALUES (?,?,?,?)", (strftime(datetime_utc), name, chat_id, source_user_id))
    
    await send('\n'.join(filter(None, [
        f"Event {name!r} saved",
        f"Date: {datetime.date()} ({date_str})",
        f"Time: {time:%H:%M}" if time else None
    ])))

async def list_events(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    if len(context.args) >= 2:
        return await send("<when> must be a day of the week or week")
    
    from datetime import date as Date, time as Time, datetime as Datetime

    chat_id = update.effective_chat.id

    if not context.args:
        when = "week"
    else:
        when, = context.args

    time = Time(0, 0)

    tz = get_my_timezone(update.message.from_user.id)
    
    beg_date, end_date = DatetimeText.to_date_range(when, tz=tz)
    beg_local, end_local = Datetime.combine(beg_date, time), Datetime.combine(end_date, time)
    
    beg, end = (x.astimezone(ZoneInfo('UTC')) for x in (beg_local, end_local))

    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        query = ("""SELECT date, name FROM Events
                    WHERE ? <= date AND date < ?
                    AND chat_id = ?
                    ORDER BY date""",
                (beg, end, chat_id))
        
        from datetime import datetime, timedelta

        def strptime(x:str):
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        def strftime(x:datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")
        
        msg = '\n'.join(f"{DatetimeText.days_english[date.weekday()]} {date:%d/%m}: {event}" if not has_hour else 
                        f"{DatetimeText.days_english[date.weekday()]} {date:%d/%m %H:%M}: {event}"
                        for date_utc, event in cursor.execute(*query)
                        for date in [strptime(date_utc).replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)]
                        for has_hour in [True])
        await send(msg or "No events for that day !")

def get_my_timezone(user_id) -> ZoneInfo:
    query = ("""SELECT timezone FROM UserTimezone WHERE user_id=?""", (user_id,))
    with sqlite3.connect('db.sqlite') as conn:
        L = conn.execute(*query).fetchall()
        if len(L) == 0:
            return None
        elif len(L) == 1:
            return ZoneInfo(L[0][0])
        else:
            raise ValueError("Unique constraint failed: Multiple timezone for user {}".format(user_id))

def set_my_timezone(user_id, tz:ZoneInfo):
    query_read = ("""SELECT timezone FROM UserTimezone WHERE user_id=?""", (user_id,))
    query_update = ("""UPDATE UserTimezone SET timezone=? WHERE user_id=?""", (str(tz), user_id))
    query_insert = ("""INSERT INTO UserTimezone(timezone, user_id) VALUES(?, ?)""", (str(tz), user_id))
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        L = conn.execute(*query_read).fetchall()
        if len(L) == 0:
            conn.execute(*query_insert)
        elif len(L) == 1:
            conn.execute(*query_update)
        else:
            raise ValueError("Unique constraint failed: Multiple timezone for user {}".format(user_id))
        conn.execute("end transaction")

def set_settings(*, user_id, key, value):
    query_read = ("""SELECT value FROM UserSettings WHERE user_id=? and key=?""", (user_id, key))
    query_update = ("""UPDATE UserSettings SET value=? WHERE user_id=? and key=?""", (value, user_id, key))
    query_insert = ("""INSERT INTO UserSettings(user_id, key, value) VALUES(?, ?, ?)""", (user_id, key, value))
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        L = conn.execute(*query_read).fetchall()
        if len(L) == 0:
            conn.execute(*query_insert)
        elif len(L) == 1:
            conn.execute(*query_update)
        else:
            raise ValueError("Unique constraint failed: Multiple settings for user {} and key {!r}".format(user_id, key))
        conn.execute("end transaction")

async def mytimezone(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    if not context.args:
        # get timezone
        tz = get_my_timezone(update.message.from_user.id)
        base_text = ("You don't have any timezone set.\nUse /mytimezone Continent/City to set it" if tz is None else
                     "Your timezone is: {}".format(tz))
        return await send(base_text)
    else:
        # set timezone
        tz_name, *_ = context.args
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            raise UserError("This time zone is not known by the system. Correct examples include America/Los_Angeles or Europe/Brussels")
        set_my_timezone(update.message.from_user.id, tz)
        return await send("Your timezone is now: {}".format(tz))

ACCEPTED_SETTINGS = ('event.timezone', 'wikt.text', 'wikt.description')

def read_settings(key, *, user_id):
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()

        query = (
            """SELECT value from UserSettings
                WHERE user_id=?
                AND key=?""",
            (user_id, key)
        )

        results = cursor.execute(*query).fetchall()
        return results[0][0] if results else None

async def settings(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    if len(context.args) not in (1, 2):
        return await send("Usage: /settings command.key\n/settings command.key value")

    if len(context.args) == 2:
        key, value = context.args
    else:
        key, value = context.args[0], None

    if key not in ACCEPTED_SETTINGS:
        return await send(f'Unknown setting: {key!r}')

    import json
    if value is None:
        # read
        value = read_settings(user_id=update.message.from_user.id, key=key)
        await send(f'Settings for {key!r}: {value}' if value is not None else
                    f'No settings for {key!r}')
    
    else:
        # write value
        set_settings(value=value, user_id=update.message.from_user.id, key=key)
        await send(f"Settings for {key!r}: {value}")

def migration0():
    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE Events(date datetime, name text)")

def migration1():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        data = conn.execute("select date, name from Events").fetchall()
        conn.execute("drop table Events")
        conn.execute("create table Events(date datetime, name text, chat_id, source_user_id)")
        conn.executemany("insert into Events(date, name, chat_id, source_user_id) values(?,?,NULL,NULL)", data)
        conn.execute("end transaction")

def migration2():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table UserTimezone(user_id, timezone text)")
        conn.execute("end transaction")

def migration3():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table UserSettings(user_id, key text, value text)")
        conn.execute("end transaction")

from decimal import Decimal
ONE_EURO_IN_BRL = Decimal("5.36")

def format_currency(*, currency_base:str, amount_base:Decimal, currency_converted:str, amount_converted:Decimal):
    return '\n'.join([
        "{}: {:.2f}".format(currency_base.upper(), amount_base),
        "{}: {:.2f}".format(currency_converted.upper(), amount_converted),
    ])

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
        return await send(format_currency(currency_base=currency_base, amount_base=amount_base, currency_converted=currency_converted, amount_converted=amount_converted))
    return money

eur = make_money_command("eur", "eur", "brl", ONE_EURO_IN_BRL)
brl = make_money_command("brl", "brl", "eur", 1 / ONE_EURO_IN_BRL)

async def help(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    fmt = ('{} - {}' if '--botfather' in context.args else
           '/{} {}')
    
    return await send('\n'.join(fmt.format(command, COMMAND_DESC.get(command, command)) for command in COMMAND_LIST))

class UserError(ValueError):
    pass

async def log_error(error, send):
    if isinstance(error, UserError):
        return await send("Error: {}".format(error))
    else:
        logging.error("Error", exc_info=error)
        return await send("An unknown error occured in your command, ask @robertvend to fix it !")

async def general_error_callback(update:Update, context:CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    
    return await log_error(context.error, send)

import unittest
import unittest.mock
from unittest import IsolatedAsyncioTestCase, TestCase

class SyncTests(TestCase):
    def test_detect_currencies(self):
        self.assertIn(('5', 'eur'), detect_currencies("This is 5€"))
        self.assertIn(('5', 'eur'), detect_currencies("This is 5 eur"))
        self.assertIn(('5', 'eur'), detect_currencies("This is 5 euros"))
        self.assertIn(('5', 'eur'), detect_currencies("This is 5 EUR"))

async def test_simple_output(function, input:list[str]):
    # setup
    context = unittest.mock.AsyncMock()
    context.args = input
    update = unittest.mock.Mock()
    update.effective_chat = Chat(type=Chat.PRIVATE, id='123')
    
    # call function
    await function(update, context)
    
    # asserts
    context.bot.send_message.assert_called_once()
    return context.bot.send_message.mock_calls[0].kwargs['text']

async def test_simple_responder(function, msg:str):
    # setup 
    send = unittest.mock.AsyncMock()
    
    # call
    await function(msg, send)
    
    # assert
    send.assert_called_once()
    return send.mock_calls[0].args[0]

async def test_multiple_responder(function, msg:str):
    # setup 
    send = unittest.mock.AsyncMock()
    
    # call
    await function(msg, send)
    
    # assert
    return [send.mock_calls[i].args[0] for i in range(len(send.mock_calls))]

class AsyncTests(IsolatedAsyncioTestCase):
    async def test_ru(self):
        self.assertEqual(await test_simple_output(ru, ['azerty']), 'азерты', "One letter mapping")
        self.assertNotEqual(await test_simple_output(ru, ['azerty']), 'лалала', "Wrong output")
        self.assertEqual(await test_simple_output(ru, ['zhina']), 'жина', "Two letters mapping")
        self.assertEqual(await test_simple_output(ru, ["hello'"]), 'хеллоь', "Soft sign")
        self.assertEqual(await test_simple_output(ru, ["hello''"]), 'хеллоъ', "Hard sign")
        self.assertEqual(await test_simple_output(ru, ["xw"]), 'хв', "x and w")
        self.assertEqual(await test_simple_output(ru, ['hello', 'shchashasha']), 'хелло щашаша', "Multiple words")
        self.assertEqual(await test_simple_output(ru, ['Chto']), 'Что', 'Mix of capital and small letters')
    
    async def test_hello_responder(self):
        self.assertIn("hello", (await test_simple_responder(hello_responder, "Hello")).lower())
        self.assertIn("hello", (await test_simple_responder(hello_responder, "Hello World !")).lower())
        self.assertEqual(0, len(await test_multiple_responder(hello_responder, "Tada")))
    
    async def test_money_responder(self):
        results = await test_multiple_responder(money_responder, "This is 5€")
        self.assertEqual(1, len(results))
        self.assertIn("EUR: 5", results[0])
        self.assertIn("BRL: 26", results[0])


COMMAND_DESC = {
    "help": "Help !",
    "caps": "Returns the list of parameters in capital letters",
    "addevent": "Add event",
    "listevents": "List events",
    "ru": "Latin alphabet to Cyrillic using Russian convention",
    "wikt": "Shows definition of each word",
    'eur': "Convert euros to other currencies",
    'brl': "Convert brazilian reals to other currencies",
    "mytimezone": "Set your timezone so that Europe/Brussels is not assumed by events commands",
    "settings": "Change user settings that are usable for commands",
}

COMMAND_LIST = ('caps', 'addevent', 'listevents', 'ru', 'wikt', 'eur', 'brl', 'mytimezone', 'settings', 'help')

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
    application.add_handler(CommandHandler('mytimezone', mytimezone))
    application.add_handler(CommandHandler('settings', settings))
    application.add_handler(CommandHandler('help', help))

    application.add_error_handler(general_error_callback)
    
    application.run_polling()
