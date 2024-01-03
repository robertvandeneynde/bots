from __future__ import annotations
import logging
from telegram import Update, Message, Chat
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler, ContextTypes
from telegram.ext import filters
from telegram.constants import ChatType
from telegram_settings_local import TOKEN
from telegram_settings_local import FRIENDS_USER

import enum
class FriendsUser(enum.StrEnum):
    FLOCON = 'flocon'
    LOUKOUM = 'loukoum'
    JOKÈRE = 'jokère'

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
from functools import partial
MONEY_CURRENCIES_ALIAS = {
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "€": "eur",
    "₺": "try",
    "try": "try",
    "tl": "try",
    "lira": "try",
    "brl": "brl",
    "real": "brl",
    "reais": "brl",
    "rub": "rub",
    "руб": "rub",
    "₽": "rub",
    '$': 'usd',
    'usd': 'usd',
    'cad': 'cad',
}
MONEY_RE = re.compile('(\\d+) ?(' + '|'.join(map(re.escape, MONEY_CURRENCIES_ALIAS)) + ')', re.I)

def read_pure_json(filename):
    import json 
    with open(filename, encoding='utf-8') as f:
        return json.load(f)

WIKTIONARY_LANGUAGES = read_pure_json('wiktionary_languages.json')
LAROUSSE_LANGUAGES = read_pure_json('larousse_languages.json')
DEFAULT_CURRENCIES = ['eur', 'usd', 'rub', 'brl', 'cad']

def strip_botname(update: Update, context: CallbackContext):
    # TODO analyse message.entities with message.parse_entity and message.parse_entities
    bot_mention: str = '@' + context.bot.username
    if update.message.text.startswith(bot_mention):
        return update.message.text[len(bot_mention):].strip()
    return update.message.text.strip()

async def hello_responder(msg:str, send:'async def', *, update, context):
    user = update.effective_user
    if user.id == FRIENDS_USER.get(FriendsUser.LOUKOUM):
        if msg.lower().startswith("hello"):
            await send("Hello my loukoum !")
        if all(word in msg.lower() for word in ('bebeğimin', 'botu')):
            await send("İyi günler Loukoum ! Çok tatlısın 🍬")
    elif user.id == FRIENDS_USER.get(FriendsUser.FLOCON):
        if msg.lower().startswith("hello"):
            await send("Bonjour flocon ! J'espère que ta journée sera artistique !")
    elif user.id == FRIENDS_USER.get(FriendsUser.JOKÈRE):
        if msg.lower().startswith("hello"):
            await send("Ʒokère ! Nous nous retrouvons ! Pas de spam en public !")
    else:
        if msg.lower().startswith("hello"):
            await send("Hello ! :3")

def detect_currencies(msg: str):
    return [(value, MONEY_CURRENCIES_ALIAS[currency_raw.lower()]) for value, currency_raw in MONEY_RE.findall(msg)]

async def money_responder(msg:str, send:'async def', *, update, context):
    detected_currencies = detect_currencies(msg)

    if detected_currencies:
        read_chat_settings = make_read_chat_settings(update, context)

        chat_currencies = set(map(str.lower, read_chat_settings('money.currencies') or DEFAULT_CURRENCIES))
        rates = get_database_euro_rates()

        for value, currency in detect_currencies(msg):
            if currency.lower() in chat_currencies:
                currencies_to_convert = [x for x in chat_currencies if x != currency]
                amount_base = Decimal(value)
                amounts_converted = [convert_money(amount_base, currency_base=currency, currency_converted=currency_to_convert, rates=rates) for currency_to_convert in currencies_to_convert]
                await send(format_currency(currency_list=[currency] + currencies_to_convert, amount_list=[amount_base] + amounts_converted))

class GetOrEmpty(list):
    def __getitem__(self, i):
        try:
            return super().__getitem__(i)
        except IndexError:
            return ''

from collections import namedtuple
NamedChatDebt = namedtuple('NamedChatDebt', 'chat_id, debitor_id, creditor_id, amount, currency')

async def sharemoney_responder(msg:str, send:'async def', *, update, context):
    chat_id = update.effective_chat.id
    read_chat_settings = make_read_chat_settings(update, context)

    setting = read_chat_settings('sharemoney.active')
    if not(setting and setting == 'on'):
        return
    
    def Amount():
        from pyparsing import Word, nums, infix_notation, opAssoc, one_of

        class EvalConstant:
            def __init__(self, tokens):
                self.value = tokens[0]

            def eval(self):
                return int(self.value)
            
        class EvalOne:
            SIGNS = {"+": 1, "-": -1}
            def __init__(self, tokens) -> None:
                self.sign, self.value = tokens[0]

            def eval(self):
                return self.SIGNS[self.sign] * self.value.eval()
        
        def operator_operands(tokenlist):
            """ generator to extract operators and operands in pairs """
            it = iter(tokenlist)
            while True:
                try:
                    yield (next(it), next(it))
                except StopIteration:
                    break

        class EvalTwo:
            OPS = {
                '+': lambda x,y: x+y,
                '-': lambda x,y: x-y,
                '*': lambda x,y: x*y,
                '/': lambda x,y: x/y,
            }
            def __init__(self, tokens):
                self.value = tokens[0]
            
            def eval(self):
                acc = self.value[0].eval()
                for op, val in operator_operands(self.value[1:]):
                    acc = self.OPS[op](acc, val.eval())
                return acc


        arithmetics = infix_notation(
            Word(nums).set_parse_action(EvalConstant),
            [
                (one_of("+ -"), 1, opAssoc.RIGHT, EvalOne),
                (one_of("* /"), 2, opAssoc.LEFT, EvalTwo),
                (one_of("+ -"), 2, opAssoc.LEFT, EvalTwo),
            ]
        )
        return arithmetics

    import regex
    name = regex.compile(r"\p{L}\w*")
    amount = Amount()
    Args = GetOrEmpty(msg.split())
    if name.fullmatch(Args[0]) and 'owes' == Args[1] and name.fullmatch(Args[2]) and amount.matches(Args[3]) and len(Args) == 4:
        first_name, _, second_name, amount_str = Args
        
        debt = NamedChatDebt(
            debitor_id=first_name,
            creditor_id=second_name,
            chat_id=chat_id,
            amount=amount.parse_string(amount_str, parse_all=True)[0].eval(),
            currency=None)
        
        simple_sql((
            'insert into NamedChatDebt(debitor_id, creditor_id, chat_id, amount, currency) values (?,?,?,?,?)',
            (debt.debitor_id, debt.creditor_id, debt.chat_id, debt.amount, debt.currency)))
        
        return await send('Debt "{d.debitor_id} owes {d.creditor_id} {d.amount}" created'.format(d=debt) if not debt.currency else
                          'Debt "{d.debitor_id} owes {d.creditor_id} {d.amount} {d.amount}" created'.format(d=debt))

RESPONDERS = (hello_responder, money_responder, sharemoney_responder)

async def on_message(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    
    # if update.message:
    #     logging.info("@{username}: {text} (In {group})".format(
    #         username=update.message.from_user.username,
    #         text=update.message.text,
    #         group='private' if update.message.chat.type == ChatType.PRIVATE else
    #                "'{}'".format(update.message.chat.title) if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL) else
    #                update.message.chat.type))
    # else:
    #     logging.info("{}".format(update))

    if update.message:
        msg = strip_botname(update, context)

        for responder in RESPONDERS:
            try:
                await responder(msg, send, update=update, context=context)
            except Exception as e:
                await log_error(e, send)

    if update.edited_message:
        pass

async def caps(update: Update, context: CallbackContext):
    text_caps = str(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

def unilinetext(x):
    import unicodedata
    return "U+{} {} {}".format(hex(ord(x))[2:].upper().zfill(4), x, unicodedata.name(x, '?'))

async def uniline(update, context):
    send = make_send(update, context)
    reply = update.message.reply_to_message
    if not reply and not context.args:
        return await send("Usage: /uniline word1 word2\nCan also be used on a reply message")
    for arg in ([reply.text] if reply else []) + list(context.args):
        S = map(unilinetext, arg)
        await send('\n'.join(S) or '[]')

async def nuniline(update, context):
    send = make_send(update, context)
    nonascii = lambda x: ord(x) > 0x7F
    reply = update.message.reply_to_message
    if not reply and not context.args:
        return await send("Usage: /nuniline word1 word2\nCan also be used on a reply message")
    for arg in ([reply.text] if reply else []) + list(context.args):
        S = map(unilinetext, filter(nonascii, arg))
        await send('\n'.join(S) or '[]')

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

def make_read_my_settings(update: Update, context: CallbackContext):
    from functools import partial
    return partial(read_settings, id=update.message.from_user.id, settings_type='user')

def make_read_chat_settings(update: Update, context: CallbackContext):
    from functools import partial
    return partial(read_settings, id=update.effective_chat.id, settings_type='chat')

async def dict_command(update: Update, context: CallbackContext, *, engine:'wikt' | 'larousse' | 'glosbe', command_name:str):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    read_my_settings = make_read_my_settings(update, context)

    if not context.args:
        if not update.message.reply_to_message:
            return await send(f"Usage: /{command_name} word1 word2 word3...\nCan also be used on a reply message")

    is_reply = False
    if update.message.reply_to_message:
        is_reply = True
        reply_message_words = update.message.reply_to_message.text.split()
        
        def any_number(items):
            import re
            number = re.compile('-?\\d+')
            return any(map(number.fullmatch, items))

        def substitute_numbers(items):
            import re
            number = re.compile('-?\\d+')
            for item in items:
                if number.fullmatch(item):
                    i = int(item) 
                    if i == 0:
                        yield item
                    else:
                        idx = i if i < 0 else i-1
                        try:
                            yield reply_message_words[idx]
                        except IndexError:
                            yield item
                else:
                    yield item

    if get_or_empty(context.args, -1).startswith('/'):
        language = context.args[-1][1:]
        parameter_words = context.args[:-1]
        if ':' in language:
            base_lang, target_lang, *_ = language.split(':')
        else:
            base_lang, target_lang = '', language
    else:
        parameter_words = context.args[:]
        base_lang = None
        target_lang = None
    
    if is_reply:
        if any_number(parameter_words):
            words = list(substitute_numbers(parameter_words))
        else:
            words = reply_message_words + parameter_words
    else:
        words = parameter_words
    
    base_lang, target_lang

    base_lang = base_lang or read_my_settings(f'{command_name}.description')
    target_lang = target_lang or read_my_settings(f'{command_name}.text')

    # lang transformation
    if engine == 'wikt':
        target_lang = WIKTIONARY_LANGUAGES.get(base_lang or 'en', {}).get(target_lang, target_lang)
    elif engine == 'larousse':
        target_lang = LAROUSSE_LANGUAGES.get('fr', {}).get(target_lang or 'fr', target_lang)
        base_lang = LAROUSSE_LANGUAGES.get('fr', {}).get(base_lang or 'fr', base_lang)
    elif engine == 'glosbe':
        pass

    # url maker
    if engine == 'wikt':
      def url(x):
        x = x.lower()
        return (
            'https://wiktionary.com/wiki/'
            + ('{}:'.format(base_lang) if base_lang else '')
            + x
            + ('#{}'.format(target_lang) if target_lang else '')
        )
    elif engine == 'larousse':
      def url(x):
        x = x.lower()
        return (
            f'https://larousse.fr/dictionnaires/{target_lang}/{x}' if target_lang == base_lang else 
            f'https://larousse.fr/dictionnaires/{target_lang}-{base_lang}/{x}'
        )
    elif engine == 'glosbe':
      def url(x):
        x = x.lower()
        return f'https://glosbe.com/{target_lang}/{base_lang}/{x}'

    return await send('\n\n'.join(url(x) for x in words))

async def wikt(update: Update, context: CallbackContext):
    return await dict_command(update, context, command_name='wikt', engine='wikt')

async def larousse(update: Update, context: CallbackContext):
    return await dict_command(update, context, command_name='larousse', engine='larousse')

async def dict_(update: Update, context: CallbackContext):
    read_my_settings = make_read_my_settings(update, context)
    engine = read_my_settings('dict.engine')
    if not engine:
        raise UserError('Engine not set for /dict command, use "/settings dict.engine wikt" for example to set wiktionary engine')
    return await dict_command(update, context, command_name='dict', engine=engine)

class UsageError(Exception):
    pass

async def flashcard(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    
    try:
      if update.message.reply_to_message:
        sentence = update.message.reply_to_message.text
        translation = ' '.join(context.args)
      else:
        def find_sentence_translation(args):
            if "/" in args:
                separator_position = args.index("/")
                sentence, translation = args[:separator_position], args[separator_position+1:]
                sentence, translation = map(' '.join, (sentence, translation))
            elif len(args) == 2:
                sentence, translation = args
            elif len(args) == 1:
                sentence, translation = args[0], ''
            else:
                raise UsageError
            return sentence, translation
        
        sentence, translation = find_sentence_translation(context.args)
    except UsageError:
        return await send("Usage:\n/flashcard word translation\n/flashcard words+ / translation+\nCan also be used on a reply message to replace the words")
    
    user_id = update.effective_user.id
    page_name = get_current_flashcard_page(user_id)
    save_flashcard(sentence, translation, user_id=user_id, page_name=page_name)

    await send(f"New flashcard:\n{sentence!r}\n-> {translation!r}")

def get_current_flashcard_page(user_id):
    with sqlite3.connect("db.sqlite") as conn:
        conn.execute("begin transaction")
        current_page_name, = conn.execute("select name from flashcardpage where user_id=? and current=1", (user_id,)).fetchone() or (None,)
        if current_page_name is None:
            current_page_name = '1'
        conn.execute("end transaction")
    return current_page_name

def save_flashcard(sentence, translation, *, user_id, page_name):
    query = ('insert into Flashcard(sentence, translation, user_id, page_name) values (?,?,?,?)', (sentence, translation, user_id, page_name))
    simple_sql(query)

def simple_sql(query):
    assert isinstance(query, (tuple, list))
    assert isinstance(query[1], (tuple, list))
    with sqlite3.connect("db.sqlite") as conn:
        return conn.execute(*query).fetchall()

def simple_sql_dict(query):
    assert isinstance(query, (tuple, list))
    assert isinstance(query[1], (tuple, list))
    with sqlite3.connect("db.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(*query).fetchall()

async def practiceflashcards(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    try:
        n = None
        if 'reversed' in context.args:
            direction = 'reversed'
        else:
            direction = 'normal'
    except UsageError:
        return await send("Usage:\n/practiceflashcards [n] [days]")
    
    user_id = update.effective_user.id
    query = ('select sentence, translation from flashcard where user_id=? and page_name=?', (user_id, get_current_flashcard_page(user_id)))
    lines = simple_sql(query)
    
    import random
    sample = random.sample(lines, n if n is not None else len(lines))
    sentences = [x[0] if direction == 'normal' else x[1] for x in sample]
    
    await send('\n'.join(map("- {}".format, sentences)))

    context.user_data['sample'] = sample
    context.user_data['direction'] = direction

    return 0

from telegram.ext import ConversationHandler
async def guessing_word(update, context):
    sample = context.user_data['sample']
    direction = context.user_data['direction']
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    answers = [x[1] if direction == 'normal' else x[0] for x in sample]
    await send('\n'.join(map("- {}".format, answers)))
    context.user_data.clear()
    return ConversationHandler.END

def make_send(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    return send

async def switchpageflashcard(update, context):
    send = make_send(update, context)
    try:
        page_name, = context.args
    except:
        return await send("Usage: /switchpageflashcard page_name")

    user_id = update.effective_user.id

    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        # 1. Remove current page, if any
        conn.execute("update flashcardpage set current=0 where user_id=?", (user_id,))
        
        # 2. Create or Update target page as current
        db_page_name, = conn.execute("select name from flashcardpage where name=? and user_id=?", (page_name, user_id)).fetchone() or (None,)
        if db_page_name is None:
            conn.execute("insert into flashcardpage(user_id, name, current) values (?,?,1)", (user_id, page_name))
        else:
            conn.execute("update flashcardpage set current=1 where user_id=? and name=?", (user_id, page_name))
        conn.execute("end transaction")

    await send(f"Your current flashcard page is now {page_name!r}")

async def exportflashcards(update, context):
    query = ('select sentence, translation from flashcard where user_id=?', (update.message.from_user.id,))
    lines = simple_sql(query)

    def export_tsv_utf8():
        import io
        file_content_io = io.StringIO()
        import csv
        csv.writer(file_content_io, dialect='excel-tab').writerows(lines)
        return file_content_io.getvalue().encode('utf-8')

    def export_xlsx():
        import openpyxl
        wb = openpyxl.Workbook()
        for line in lines:
            wb.active.append(line)
        import io
        bytes_io = io.BytesIO()
        wb.save(bytes_io)
        return bytes_io.getvalue()
    
    #file_content, extension = export_tsv_utf8(), 'tsv'
    file_content, extension = export_xlsx(), 'xlsx'

    await context.bot.send_document(update.effective_chat.id, file_content, filename="flashcards." + extension)

import zoneinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

class DatetimeText:
    days_english = "monday tuesday wednesday thursday friday saturday sunday".split() 
    days_french = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
    
    @classmethod
    def to_datetime_range(self, name, *, time=None, reference=None, tz=None):
        from datetime import datetime as Datetime, time as Time
        date, date_end = r = self.to_date_range(name, reference=reference, tz=tz)
        datetime = Datetime.combine(date, time or Time(0,0)).replace(tzinfo=tz)
        return datetime, date_end

    @classmethod
    def to_date_range(self, name, *, reference=None, tz=None) -> date:
        from datetime import datetime, timedelta, date, date as Date
        reference = reference or datetime.now().astimezone(ZoneInfo("Europe/Brussels") if tz is None else tz).replace(tzinfo=None)
        today = reference.date()
        name = name.lower()
        
        if match := re.match("(\d{4})-(\d{2})-(\d{2})", name):
            day = date(*map(int, match.groups()))
            return day, day + timedelta(days=1)
        
        if name in ("today", "auj", "aujourdhui", "aujourd'hui"):
            return today, today + timedelta(days=1)
        
        if name in ("week", "semaine"):
            beg = today
            end = today + timedelta(days=7)
            return beg, end
        
        if name in ("tomorrow", "demain"):
            return today + timedelta(days=1), today + timedelta(days=2)
        
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

from collections import namedtuple
ParsedEvent = namedtuple('ParsedEvent', 'date time name')

def parse_event(args) -> (str, time | None, str):
    from datetime import date as Date, time as Time, timedelta as Timedelta

    date, *rest = args
    if match := re.compile('(\\d{1,2})[:hH](\\d{2})?').fullmatch(get_or_empty(rest, 0)):
        hours, minutes = match.group(1), match.group(2)
        time = Time(int(hours), int(minutes or '0'))
        rest = rest[1:]
    else:
        time = None
    name = " ".join(rest)

    return ParsedEvent(date, time, name)

import sqlite3
async def add_event(update: Update, context: CallbackContext):
    send = make_send(update, context)
    if not context.args:
        return await send("Usage: /addevent date name")
    from datetime import datetime as Datetime, time as Time, date as Date, timedelta
    
    source_user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    date_str, time, name = parse_event(context.args)

    tz = get_my_timezone(update.message.from_user.id) or ZoneInfo("Europe/Brussels")
    
    date, date_end = DatetimeText.to_date_range(date_str, tz=tz)
    datetime = Datetime.combine(date, time or Time(0,0)).replace(tzinfo=tz)

    datetime_utc = datetime.astimezone(ZoneInfo('UTC'))

    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()

        def strftime(x:datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("INSERT INTO Events(date, name, chat_id, source_user_id) VALUES (?,?,?,?)", (strftime(datetime_utc), name, chat_id, source_user_id))
    
    read_chat_settings = make_read_chat_settings(update, context)
    chat_timezones = read_chat_settings("event.timezones")
    await send('\n'.join(filter(None, [
        f"Event {name!r} saved",
        f"Date: {datetime.date()} ({date_str})",
        (f"Time: {time:%H:%M} ({tz})" if chat_timezones and set(chat_timezones) != {tz} else
         f"Time: {time:%H:%M}") if time else None
    ] + ([
        f"Time: {datetime_tz:%H:%M} ({timezone})" if datetime_tz.date() == datetime.date() else
        f"Time: {datetime_tz:%H:%M} on {datetime_tz.date()} ({timezone})"
        for timezone in chat_timezones or []
        if timezone != tz
        for datetime_tz in [datetime.astimezone(timezone)]
    ] if time else []))))

def sommeil(s, *, command) -> (datetime, datetime):
    if m := re.match("/%s (.*)" % command, s):
        s = m.group(1)
    args = s.split()
    a,tiret,b = args
    assert tiret == '-'
    a = map(int, a.split(":"))
    b = map(int, b.split(":"))
    ax,ay = a
    bx,by = b
    from datetime import time
    at = time(hour=ax, minute=ay)
    bt = time(hour=bx, minute=by)
    now = datetime.now().astimezone()
    adt = datetime.combine(now, at)
    bdt = datetime.combine(now, bt)
    if not adt < bdt:
        adt -= timedelta(days=1)
    assert adt < bdt
    return adt, bdt

async def sleep_(update, context):
    send = make_send(update, context)
    from_dt, to_dt = sommeil(update.effective_message.text, command='sleep')
    await send(str(to_dt - from_dt))
    
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

    tz = get_my_timezone(update.message.from_user.id) or ZoneInfo("Europe/Brussels")
    
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
        
        read_chat_settings = make_read_chat_settings(update, context)
        chat_timezones = read_chat_settings("event.timezones")
        msg = '\n'.join(f"{DatetimeText.days_english[date.weekday()]} {date:%d/%m}: {event}" if not has_hour else 
                        f"{DatetimeText.days_english[date.weekday()]} {date:%d/%m %H:%M}: {event}"
                        for date_utc, event in cursor.execute(*query)
                        for date in [strptime(date_utc).replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)]
                        for has_hour in [True])
        if msg and chat_timezones and set(chat_timezones) != {tz}:
            msg += '\n\n' + f"Timezone: {tz}"
        await send(msg or "No events for that day !")

def n_to_1_dict(x:dict|Iterable):
    gen = x.items() if isinstance(x, dict) else x
    
    def is_cool_iterable(it):
        import collections.abc
        return (isinstance(it, collections.abc.Sequence) 
          and not isinstance(it, str))

    d = {}
    for key, value in gen:
        if is_cool_iterable(key):
            for v in key:
                d[v] = value
        else:
            d[key] = value
    return d

def fetch_event(key):
    return None

async def timedifference(update, context, command):
    from datetime import timedelta, datetime
    send = make_send(update, context)
    conversions = n_to_1_dict({
        ('minutes', 'min', 'minute'): timedelta(minutes=1), 
        ('days', 'day'): timedelta(days=1),
        ('h', 'hours', 'hour'): timedelta(hours=1),
        ('s', 'seconds', 'second'): timedelta(seconds=1),
        ('weeks', 'week'): timedelta(weeks=1),
    })
    is_timedelta = re.compile('|'.join(map(re.escape, conversions)))
    di = next((i for i, x in enumerate(context.args) if is_timedelta.fullmatch(x)), None)
    units = ('days' if di is None else
             context.args[di])
    dtunits = conversions[units]
    eventkey, = context.args
    event: datetime = fetch_event(eventkey) or DatetimeText.to_datetime_range(eventkey)[0]
    delta = event - datetime.now()
    if command == 'timesince':
        delta = -delta
    await send('{:.2f} {}'.format(delta / dtunits, units))

async def timeuntil(update, context):
    return await timedifference(update, context, command='timeuntil')

async def timesince(update, context):
    return await timedifference(update, context, command='timesince')

async def deletevent(update, context):
    send = make_send(update, context)
    key, = context.args
    await send("Not implemented yet!")

def get_my_timezone_from_timezone_table(user_id) -> ZoneInfo:
    query = ("""SELECT timezone FROM UserTimezone WHERE user_id=?""", (user_id,))
    with sqlite3.connect('db.sqlite') as conn:
        L = conn.execute(*query).fetchall()
        if len(L) == 0:
            return None
        elif len(L) == 1:
            return ZoneInfo(L[0][0])
        else:
            raise ValueError("Unique constraint failed: Multiple timezone for user {}".format(user_id))

def get_my_timezone(user_id) -> ZoneInfo:
    return (read_settings('event.timezone', id=user_id, settings_type='user')
            or get_my_timezone_from_timezone_table(user_id))

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

def set_settings(*, id, key, value_raw:any, settings_type:'chat' | 'user'):
    conversion = CONVERSION_SETTINGS[settings_type][key]['to_db']

    value: any = conversion(value_raw)
    table = SettingsInfo.TABLES[settings_type]
    field_id = SettingsInfo.FIELDS[settings_type]
    query_read = (f"""SELECT value FROM {table} WHERE {field_id}=? and key=?""", (id, key))
    query_update = (f"""UPDATE {table} SET value=? WHERE {field_id}=? and key=?""", (value, id, key))
    query_insert = (f"""INSERT INTO {table}({field_id}, key, value) VALUES(?, ?, ?)""", (id, key, value))
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        L = conn.execute(*query_read).fetchall()
        if len(L) == 0:
            conn.execute(*query_insert)
        elif len(L) == 1:
            conn.execute(*query_update)
        else:
            raise ValueError("Unique constraint failed: Multiple settings for {} {} and key {!r}".format(settings_type, id, key))
        conn.execute("end transaction")

def delete_settings(*, id, key, settings_type:'chat' | 'user'):
    table = SettingsInfo.TABLES[settings_type]
    field_id = SettingsInfo.FIELDS[settings_type]
    query_delete = (f"""DELETE FROM {table} WHERE {field_id}=? and key=?""", (id, key))
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute(*query_delete)

async def mytimezone(update: Update, context: CallbackContext):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    if not context.args:
        # get timezone
        tz = get_my_timezone_from_timezone_table(update.message.from_user.id)
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

ACCEPTED_SETTINGS_USER = (
    'event.timezone',
    'wikt.text',
    'wikt.description',
    'larousse.text',
    'larousse.description',
    'dict.text',
    'dict.description',
    'dict.engine',
)
ACCEPTED_SETTINGS_CHAT = (
    'money.currencies',
    'sharemoney.active',
    'event.timezones',
)

def CONVERSION_SETTINGS_BUILDER():
    import json
    default_serializer = {
        'from_db': lambda x:x,
        'to_db': lambda x:x,
    }
    json_serializer = {
        'from_db': json.loads,
        'to_db': json.dumps,
    }
    timezone_serializer = {
        'from_db': ZoneInfo,
        'to_db': lambda x:x,
    }
    list_of_timezone_serializer = {
        'from_db': lambda s: list(map(ZoneInfo, json.loads(s))),
        'to_db': lambda L: json.dumps(list(map(str, L)))
    }
    list_of_currencies_serializer = {
        'from_db': lambda s: list(map(str.upper, json.loads(s))),
        'to_db': lambda L: json.dumps(list(map(str.upper, L)))
    }
    mapping_chat = {
        'money.currencies': list_of_currencies_serializer,
        'event.timezones': list_of_timezone_serializer,
    }
    mapping_user = {
        'event.timezone': timezone_serializer,
    }
    from collections import defaultdict
    return {
        'chat': defaultdict(lambda: default_serializer, mapping_chat),
        'user': defaultdict(lambda: default_serializer, mapping_user)
    }

CONVERSION_SETTINGS = CONVERSION_SETTINGS_BUILDER()
assert {'chat', 'user'} <= CONVERSION_SETTINGS.keys(), f'Missing keys in {CONVERSION_SETTINGS}'
assert all({'from_db', 'to_db'} <= x.keys() for y in CONVERSION_SETTINGS.values() for x in y.values()), f"Missing 'from_db' or 'to_db' in {CONVERSION_SETTINGS=}"

class SettingsInfo:
    TABLES = {'chat': 'ChatSettings', 'user': 'UserSettings'}
    FIELDS = {'chat': 'chat_id', 'user': 'user_id'}

def read_settings(key, *, id, settings_type:'chat' | 'user'):
    conversion = CONVERSION_SETTINGS[settings_type][key]['from_db']

    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()

        table_name = SettingsInfo.TABLES[settings_type]
        field_id = SettingsInfo.FIELDS[settings_type]
        query = (
            f"""SELECT value from {table_name}
                WHERE {field_id}=?
                AND key=?""",
            (id, key)
        )

        results = cursor.execute(*query).fetchall()
        return conversion(results[0][0]) if results else None

async def listallsettings(update: Update, context: CallbackContext):
    send = make_send(update, context)
    await send('\n'.join("- {} ({})".format(
            setting,
            '|'.join(['user'] * (setting in ACCEPTED_SETTINGS_USER) + ['chat'] * (setting in ACCEPTED_SETTINGS_CHAT)))
        for setting in sorted(ACCEPTED_SETTINGS_USER + ACCEPTED_SETTINGS_CHAT)))

async def settings_command(update: Update, context: CallbackContext, *, command_name: str, settings_type:'chat' | 'user', accepted_settings:list[str]):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)

    async def print_usage():
        await send(f"Usage:\n/{command_name} command.key\n/{command_name} command.key value")

    if len(context.args) == 0:
        return await print_usage()

    key, *rest = context.args

    if key not in accepted_settings:
        return await send(f'Unknown settings: {key!r}\n\nType /listallsettings for complete list of settings (hidden command)')

    if key in ('money.currencies', 'event.timezones'):
        value = list(rest)
    else:
        # default, single value no space string
        if len(rest) not in (0, 1):
            return await print_usage()
        value = rest[0] if rest else None

    if settings_type == 'user':
        id = update.message.from_user.id
    elif settings_type == 'chat':
        id = update.effective_chat.id
    else:
        raise ValueError(f'Invalid settings_type: {settings_type}')

    if value is None:
        # read
        value = read_settings(id=id, key=key, settings_type=settings_type)
        await send(f'Settings: {key} = {value}' if value is not None else
                    f'No settings for {key!r}')
    
    else:
        # write value
        set_settings(value_raw=value, id=id, key=key, settings_type=settings_type)
        await send(f"Settings: {key} = {value}")

async def delsettings_command(update:Update, context: CallbackContext, *, accepted_settings:list[str], settings_type:'chat' | 'id', command_name:str):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    
    if len(context.args) != 1:
        return await send(f"Usage: /{command_name} command.key")
    
    key, = context.args

    if key not in accepted_settings:
        return await send(f'Unknown setting: {key!r}')

    if settings_type == 'user':
        id = update.message.from_user.id
    elif settings_type == 'chat':
        id = update.effective_chat.id
    else:
        raise ValueError(f'Invalid settings_type: {settings_type}')
     
    delete_settings(id=id, key=key, settings_type=settings_type)
    await send(f"Deleted settings for {key!r}")

settings = partial(settings_command, accepted_settings=ACCEPTED_SETTINGS_USER, settings_type='user', command_name='settings')
delsettings = partial(delsettings_command, accepted_settings=ACCEPTED_SETTINGS_USER, settings_type='user', command_name='delsettings')
chatsettings = partial(settings_command, accepted_settings=ACCEPTED_SETTINGS_CHAT, settings_type='chat', command_name='chatsettings')
delchatsettings = partial(delsettings_command, accepted_settings=ACCEPTED_SETTINGS_CHAT, settings_type='chat', command_name='delchatsettings')

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

def migration4():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table EuroRates(datetime, rates)")
        conn.execute("end transaction")

def migration5():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table ChatSettings(chat_id, key text, value text)")
        conn.execute("end transaction")

def migration6():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table Flashcard(user_id, sentence, translation)")
        conn.execute("end transaction")

def migration7():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute("begin transaction")
        conn.execute("create table FlashcardPage(user_id, name, current)")
        conn.execute("alter table Flashcard add page_name default '1'")
        conn.execute("end transaction")

def migration8():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute('begin transaction')
        conn.execute("create table NamedChatDebt(chat_id, debitor_id, creditor_id, amount, currency)") # debitor owes creditor
        conn.execute('end transaction')


def get_latest_euro_rates_from_api() -> json:
    import requests
    from telegram_settings_local import FIXER_TOKEN
    response = requests.get(f'http://data.fixer.io/api/latest?access_key={FIXER_TOKEN}&base=EUR').json()
    assert response['success']
    assert response['base'] == 'EUR'
    # response['date']
    return response['rates']

class DatetimeDbSerializer:
    @staticmethod
    def strptime(x:str):
        from datetime import datetime
        return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    @staticmethod
    def strftime(x:datetime):
        return x.strftime("%Y-%m-%d %H:%M:%S")

    def to_db(self, x: Datetime):
        return self.strftime(x)

    def from_db(self, x: any):
        return self.strptime(x)

class JsonDbSerializer:
    def to_db(self, x: json):
        import json
        return json.dumps(x)

    def from_db(self, x: any):
        import json
        return json.loads(x)

def get_database_euro_rates() -> Rates:
    query_get_last_date = ('''select MAX(datetime), rates from EuroRates''', ())

    from datetime import datetime as Datetime, timedelta as Timedelta

    with sqlite3.connect('db.sqlite') as conn:
        latest_date_string, rates_string = conn.execute(*query_get_last_date).fetchone() or (None, None)
        latest_date: Datetime = latest_date_string and DatetimeDbSerializer().from_db(latest_date_string)
        rates: json = rates_string and JsonDbSerializer().from_db(rates_string)
    
    now = Datetime.utcnow()
    if latest_date is None or now - latest_date > Timedelta(days=1):
        rates = get_latest_euro_rates_from_api()
        with sqlite3.connect('db.sqlite') as conn:
            conn.execute('''INSERT INTO EuroRates(datetime, rates) VALUES(?, ?)''', (DatetimeDbSerializer().to_db(now), JsonDbSerializer().to_db(rates)))
        return rates
    else:
        return rates


from decimal import Decimal

def format_currency(*, currency_list:list[str], amount_list:list[Decimal]):
    return '\n'.join(
        "{}: {:.2f}".format(currency.upper(), amount)
        for currency, amount in zip(currency_list, amount_list))

def convert_money(amount: Decimal, currency_base:str, currency_converted:str, rates:Rates):
    currency_base, currency_converted = currency_base.upper(), currency_converted.upper()
    if currency_base == 'EUR':
        return amount * Decimal(rates[currency_converted])
    if currency_converted == 'EUR':
        return amount / Decimal(rates[currency_base.upper()])
    return convert_money(convert_money(amount, currency_base, 'EUR', rates=rates), 'EUR', currency_converted, rates=rates)

def make_money_command(name:str, currency:str):
    async def money(update: Update, context: CallbackContext):
        async def send(m):
            await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
        read_chat_settings = make_read_chat_settings(update, context)

        chat_currencies = read_chat_settings('money.currencies') or DEFAULT_CURRENCIES

        from decimal import Decimal
        value, *_ = context.args or ['1']
        amount_base = Decimal(value)
        rates = get_database_euro_rates()
        currencies_to_convert = [x for x in chat_currencies if x != currency]
        amounts_converted = [convert_money(amount_base, currency_base=currency, currency_converted=currency_to_convert, rates=rates) for currency_to_convert in currencies_to_convert]
        return await send(format_currency(currency_list=[currency] + currencies_to_convert, amount_list=[amount_base] + amounts_converted))
    return money

eur = make_money_command("eur", "eur")
brl = make_money_command("brl", "brl")
rub = make_money_command("rub", "rub")

async def convertmoney(update, context):
    async def send(m):
        await context.bot.send_message(text=m, chat_id=update.effective_chat.id)
    read_chat_settings = make_read_chat_settings(update, context)

    try:
        if len(context.args) == 2:
            value, currency = context.args
            currency = currency.upper()
            mode = 'to_chat_currencies'
            direction, currency_converted = None, None
        elif len(context.args) == 4:
            value, currency, direction, currency_converted = context.args
            currency = currency.upper()
            currency_converted = currency_converted.upper()
            mode = 'to_one_currency'
            assert direction == 'to'
        else:
            raise Exception
    except:
        return await send("Usage: /convertmoney value currency [to currency]")

    amount_base = Decimal(value)
    rates = get_database_euro_rates()
    
    if mode == 'to_chat_currencies':
        chat_currencies = read_chat_settings('money.currencies') or DEFAULT_CURRENCIES
        currencies_to_convert = [x.upper() for x in chat_currencies if x.upper() != currency.upper()]
    elif mode == 'to_one_currency':
        currencies_to_convert = [currency_converted.upper()]

    try:
        amounts_converted = [convert_money(amount_base, currency_base=currency, currency_converted=currency_to_convert, rates=rates) for currency_to_convert in currencies_to_convert]
    except KeyError as e:
        raise UserError(f"Unknown currency: {e}")
    await send(format_currency(currency_list=[currency] + currencies_to_convert, amount_list=[amount_base] + amounts_converted))

async def sharemoney(update, context):
    send = make_send(update, context)
    Args = GetOrEmpty(context.args)
    on_off, = context.args
    try:
        if on_off == 'on':
            return await send("Run: /chatsettings sharemoney.active on")
        elif on_off == 'off':
            return await send("Run: /delchatsettings sharemoney.active")
        else:
            raise UsageError
    except UsageError:
        return await send('Usage: /sharemoney on|off')

async def listdebts(update, context):
    send = make_send(update, context)
    chat_id = update.effective_chat.id
    lines = simple_sql_dict(('select chat_id, debitor_id, creditor_id, amount, currency from NamedChatDebt where chat_id=?', (chat_id,)))
    debts_sum = {}
    
    for debt in (NamedChatDebt(**x) for x in lines):
        if debt.currency:
            return await send("I cannot deal with debt with currencies atm...")
        
        if (debt.debitor_id, debt.creditor_id) in debts_sum or (debt.creditor_id, debt.debitor_id) in debts_sum:
            key = ((debt.debitor_id, debt.creditor_id) if (debt.debitor_id, debt.creditor_id) in debts_sum else
                   (debt.creditor_id, debt.debitor_id))
            
            sign = (+1 if (debt.debitor_id, debt.creditor_id) in debts_sum else
                    -1)
            
            debts_sum[key] += sign * Decimal(debt.amount)
        else:
            debts_sum[debt.debitor_id, debt.creditor_id] = Decimal(debt.amount)
    
    return await send('\n'.join(
        "{} owes {} {}".format(debitor, creditor, amount) if amount > 0 else
        "{} owes {} {}".format(creditor, debitor, -amount) if amount < 0 else
        "{} and {} are even".format(debitor, creditor)
        for (debitor, creditor), amount in debts_sum.items()) or 'No debts in this chat !')
    

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
        if update and update.effective_chat:
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
    
    @unittest.skip
    async def test_hello_responder(self):
        self.assertIn("hello", (await test_simple_responder(hello_responder, "Hello")).lower())
        self.assertIn("hello", (await test_simple_responder(hello_responder, "Hello World !")).lower())
        self.assertEqual(0, len(await test_multiple_responder(hello_responder, "Tada")))
    
    @unittest.skip
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
    "dict": "Shows definition of each word using dictionary and settings engine",
    "wikt": "Shows definition of each word using wiktionary",
    "larousse": "Show definition of each word using french dictionary Larousse.fr",
    'eur': "Convert euros to other currencies",
    'brl': "Convert brazilian reals to other currencies",
    'rub': "Convert russian rubles to other currencies",
    'convertmoney': 'Convert money to chat currencies or to specific currency',
    "mytimezone": "Set your timezone so that Europe/Brussels is not assumed by events commands",
    "settings": "Change user settings that are usable for commands",
    "delsettings": "Delete user settings that are usable for commands",
    "chatsettings": "Change chat settings that are usable for commands",
    "delchatsettings": "Delete chat settings that are usable for commands",
    "flashcard": "Add a new flashcard to help memorize words more easily",
    "exportflashcards": "Export your flashcards in excel format",
    "praticeflashcards": "Practice your flashcards to train your memory",
    "switchpageflashcard": "Switch to a page to group your flashcards",
    "uniline": "Describe in Unicode each character or symbol or emoji",
    "nuniline": "Describe in Unicode each non ascii character or symbol or emoji",
    "timeuntil": "Tell the time until an event",
    "timesince": "Tell the elapsed time since an event",
    "sleep": "Record personal sleep cycle and make graphs", 
    "sharemoney": "Manage money between users (shared bank account, add a debt)",
    "listdebts": "List debts between users (sharemoney)",
}

COMMAND_LIST = (
    'caps',
    'addevent', 'listevents',
    'ru',
    'dict', 'wikt', 'larousse',
    'convertmoney', 'eur', 'brl', 'rub',
    'mytimezone', 'settings', 'delsettings', 'chatsettings', 'delchatsettings',
    'flashcard', 'exportflashcards', 'practiceflashcards', 'switchpageflashcard',
    'help',
    'uniline', 'nuniline',
    #'sleep',
    'sharemoney', 'listdebts',
)

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
    application.add_handler(CommandHandler('dict', dict_))
    application.add_handler(CommandHandler('wikt', wikt))
    application.add_handler(CommandHandler('larousse', larousse))
    application.add_handler(CommandHandler('eur', eur))
    application.add_handler(CommandHandler('brl', brl))
    application.add_handler(CommandHandler('rub', rub))
    application.add_handler(CommandHandler('convertmoney', convertmoney))
    application.add_handler(CommandHandler('mytimezone', mytimezone))
    application.add_handler(CommandHandler('listallsettings', listallsettings))
    application.add_handler(CommandHandler('settings', settings))
    application.add_handler(CommandHandler('delsettings', delsettings))
    application.add_handler(CommandHandler('chatsettings', chatsettings))
    application.add_handler(CommandHandler('delchatsettings', delchatsettings))
    application.add_handler(CommandHandler('flashcard', flashcard))
    application.add_handler(CommandHandler('switchpageflashcard', switchpageflashcard))
    application.add_handler(CommandHandler('exportflashcards', exportflashcards))
    application.add_handler(CommandHandler('sharemoney', sharemoney))
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler('practiceflashcards', practiceflashcards)],
        states={
            0: [MessageHandler(filters.TEXT, guessing_word)],
        },
        fallbacks=[]
    ), group=1)
    application.add_handler(CommandHandler('help', help))
    application.add_handler(CommandHandler('uniline', uniline))
    application.add_handler(CommandHandler('nuniline', nuniline))
    application.add_handler(CommandHandler('timeuntil', timeuntil))
    application.add_handler(CommandHandler('timesince', timesince))
    #application.add_handler(CommandHandler('sleep', sleep_))
    application.add_handler(CommandHandler('listdebts', listdebts))

    application.add_error_handler(general_error_callback)
    
    application.run_polling()

