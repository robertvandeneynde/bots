from __future__ import annotations
import logging
from telegram import Update, Message, Chat, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler, MessageHandler, ContextTypes, CallbackQueryHandler
from telegram.ext import filters
from telegram.constants import ChatType
from telegram_settings_local import TOKEN
from telegram_settings_local import FRIENDS_USER

import json

import enum
class FriendsUser(enum.StrEnum):
    FLOCON = 'flocon'
    KOROLEVA_LION = 'koroleva-lion'
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

EVENT_ICS_TEMPLATE = '''\
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//hacksw/handcal//NONSGML v1.0//EN
BEGIN:VEVENT
DTSTAMP:{dt_created_utc:%Y%m%dT%H%M%SZ}
DTSTART:{dt_start_utc:%Y%m%dT%H%M%SZ}
DTEND:{dt_end_utc:%Y%m%dT%H%M%SZ}
SUMMARY:{name_ical_formatted}
END:VEVENT
END:VCALENDAR\
''' # UID:uid1@example.com, GEO:48.85299;2.36885, ORGANIZER;CN=John Doe:MAILTO:john.doe@example.com

def get_reply(message):
    if not message.is_topic_message:
        return message.reply_to_message
    elif message.reply_to_message.id == message.message_thread_id:
        return None
    else:
        return message.reply_to_message

def strip_botname(update: Update, context: CallbackContext):
    # TODO analyse message.entities with message.parse_entity and message.parse_entities
    bot_mention: str = '@' + context.bot.username
    if update.message.text.startswith(bot_mention):
        return update.message.text[len(bot_mention):].strip()
    return update.message.text.strip()

async def hello_responder(msg:str, send: AsyncSend, *, update, context):
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
    elif user.id == FRIENDS_USER.get(FriendsUser.KOROLEVA_LION):
        if msg.lower().startswith("hello"):
            await send("Hellow you wild sladkij ^^ Hope your day will improve your life !")
    else:
        if msg.lower().startswith("hello"):
            await send("Hello ! :3")

def detect_currencies(msg: str):
    return [(value, MONEY_CURRENCIES_ALIAS[currency_raw.lower()]) for value, currency_raw in MONEY_RE.findall(msg)]

from typing import Callable, Awaitable
AsyncSend = Callable[[Update, CallbackContext], Awaitable[None]]

async def money_responder(msg:str, send: AsyncSend, *, update, context):
    detected_currencies = detect_currencies(msg)

    if detected_currencies:
        read_chat_settings = make_read_chat_settings(update, context)

        chat_currencies = list(remove_dup_keep_order(read_chat_settings('money.currencies') or DEFAULT_CURRENCIES))
        rates = get_database_euro_rates()

        for value, currency_lower in detect_currencies(msg):
            if (currency := currency_lower.upper()) in chat_currencies:
                currencies_to_convert = [x for x in chat_currencies if x != currency]
                amount_base = Decimal(value)
                amounts_converted = [convert_money(amount_base, currency_base=currency, currency_converted=currency_to_convert, rates=rates) for currency_to_convert in currencies_to_convert]
                await send(format_currency(currency_list=[currency] + currencies_to_convert, amount_list=[amount_base] + amounts_converted))

async def whereisanswer_responder(msg:str, send: AsyncSend, *, update, context):
    reply = get_reply(update.message)

    class DoNotAnswer(Exception):
        pass

    try:
        assert_true(reply and reply.text, DoNotAnswer)
        assert_true(reply.text.startswith('/whereis') or reply.text.startswith('/whereis@' + context.bot.username), DoNotAnswer)
    except DoNotAnswer:
        return
    
    key = ' '.join(InfiniteEmptyList(reply.text.split())[1:])
    value = msg
    
    await save_thereis(key, value, update=update, context=context)
    
    
class GetOrEmpty(list):
    def __getitem__(self, i):
        try:
            return super().__getitem__(i)
        except IndexError:
            return ''
InfiniteEmptyList = GetOrEmpty

from collections import namedtuple
NamedChatDebt = namedtuple('NamedChatDebt', 'chat_id, debitor_id, creditor_id, amount, currency')

async def sharemoney_responder(msg:str, send: AsyncSend, *, update, context):
    chat_id = update.effective_chat.id

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

RESPONDERS = (
    (hello_responder, 'hello', 'on'),
    (money_responder, 'money', 'on'),
    (sharemoney_responder, 'sharemoney', 'off'),
    (whereisanswer_responder, 'whereisanswer', 'on'),
)

async def on_message(update: Update, context: CallbackContext):
    send = make_send(update, context)
    
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
        read_settings = make_read_chat_settings(update, context)

        for responder, setting, default in RESPONDERS:
            if (read_settings(setting + '.active') or default) == 'off':
                continue
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
    if not (reply := get_reply(update.message)) and not context.args:
        return await send("Usage: /uniline word1 word2\nCan also be used on a reply message")
    for arg in ([reply.text] if reply else []) + list(context.args):
        S = map(unilinetext, arg)
        await send('\n'.join(S) or '[]')

async def nuniline(update, context):
    send = make_send(update, context)
    nonascii = lambda x: ord(x) > 0x7F
    if not (reply := get_reply(update.message)) and not context.args:
        return await send("Usage: /nuniline word1 word2\nCan also be used on a reply message")
    for arg in ([reply.text] if reply else []) + list(context.args):
        S = map(unilinetext, filter(nonascii, arg))
        await send('\n'.join(S) or '[]')

async def befluent(update, context):
    send = make_send(update, context)

    await send("Hello")

    return ConversationHandler.END

async def ru(update: Update, context: CallbackContext):
    send = make_send(update, context)
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

DICT_ENGINES = ('wikt', 'larousse', 'glosbe')

async def dict_command(update: Update, context: CallbackContext, *, engine:Literal['wikt'] | Literal['larousse'] | Literal['glosbe'], command_name:str):
    send = make_send(update, context)
    read_my_settings = make_read_my_settings(update, context)

    reply = get_reply(update.message)
    if not context.args:
        if not reply:
            return await send(f"Usage: /{command_name} word1 word2 word3...\nCan also be used on a reply message")

    if reply:
        reply_message_words = reply.text.split()
        
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
    
    if reply:
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
    else:
        raise UsageError("Engine is misconfigured, please run /settings dict.engine {}".format('|'.join(DICT_ENGINES)))

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
    send = make_send(update, context)
    try:
      if reply := get_reply(update.message):
        sentence = reply.text 
        translation = ' '.join(context.args)
      else:
        def find_sentence_translation(args):
            if any(x in args for x in ("=", "/")):
                separator_position = args.index("=" if "=" in args else "/")
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
        return await send("Usage:\n/flashcard word translation\n/flashcard words+ = translation+\n/flashcard words+ / translation+\nCan also be used on a reply message to replace the words")
    
    user_id = update.effective_user.id
    page_name = get_current_flashcard_page(user_id)
    save_flashcard(sentence, translation, user_id=user_id, page_name=page_name)

    await send(f"New flashcard:\n{sentence!r}\n→ {translation!r}")

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

def simple_sql(query, *, connection=None):
    conn = connection
    assert isinstance(query, (tuple, list))
    assert isinstance(query[1], (tuple, list))
    if conn:
        return conn.execute(*query).fetchall()
    with sqlite3.connect("db.sqlite") as conn:
        return conn.execute(*query).fetchall()

def simple_sql_dict(query):
    assert isinstance(query, (tuple, list))
    assert isinstance(query[1], (tuple, list))
    with sqlite3.connect("db.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(*query).fetchall()

async def practiceflashcards(update, context):
    send = make_send(update, context)

    try:
        n = None
        if 'reversed' in context.args:
            direction = 'reversed'
        else:
            direction = 'normal'
    except UsageError:
        return await send("Usage:\n/practiceflashcards [n] [days]")
    
    user_id = update.effective_user.id
    query = ('select sentence, translation from flashcard where user_id=? and page_name=?', (user_id, current_page := get_current_flashcard_page(user_id)))
    lines = simple_sql(query)

    if not lines:
        return await send(f"No flashcards for page {current_page}")
    
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
    send = make_send(update, context)
    answers = [x[1] if direction == 'normal' else x[0] for x in sample]
    await send('\n'.join(map("- {}".format, answers)))
    context.user_data.clear()
    return ConversationHandler.END

SendSaveInfo = namedtuple('SendSaveInfo', 'chat_id thread_id')
def make_send(update: Update, context: CallbackContext, *, save_info: SendSaveInfo = None) -> AsyncSend:
    if not save_info:
        save_info = make_send_save_info(update, context)
    async def send(m, **kwargs):
        await context.bot.send_message(
            text=m,
            chat_id=save_info.chat_id,
            message_thread_id=save_info.thread_id,
            **kwargs)
    return send

def make_send_save_info(update: Update, context: CallbackContext) -> SendSaveInfo:
    return SendSaveInfo(
        chat_id=update.effective_chat.id,
        thread_id=update.message.message_thread_id if update.message and update.message.is_topic_message else None,
    )

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
    
    #file_content = export_tsv_utf8()
    #extension = 'tsv'
    file_content: bytes = export_xlsx()
    extension: str = 'xlsx'

    await context.bot.send_document(update.effective_chat.id, file_content, filename="flashcards." + extension)

async def export_event(update, context, *, name, datetime_utc):
    from datetime import date, time, datetime, timedelta
    
    file_content_str = EVENT_ICS_TEMPLATE.format(
        dt_created_utc=datetime.utcnow(),
        dt_start_utc=datetime_utc,
        dt_end_utc=datetime_utc + timedelta(hours=1),
        name_ical_formatted=name)
    
    file_content: bytes = file_content_str.encode('utf-8')
    
    await context.bot.send_document(
        update.effective_chat.id,
        file_content,
        filename="event.ics",
        message_thread_id=update.message.message_thread_id if update.message.is_topic_message else None)

import zoneinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
UTC = ZoneInfo('UTC')

class DatetimeText:
    days_english = "monday tuesday wednesday thursday friday saturday sunday".split() 
    days_french = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()

    months_english = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december"
    ]

    months_french = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre"
    ]

    months_value = {
        x: i for i, x in enumerate(months_english, start=1)
    } | {
        x: i for i, x in enumerate(months_french, start=1)
    }
    
    @classmethod
    def to_datetime_range(self, name, *, time=None, reference=None, tz=None):
        from datetime import datetime as Datetime, time as Time
        date, date_end = r = self.to_date_range(name, reference=reference, tz=tz)
        datetime = Datetime.combine(date, time or Time(0,0)).replace(tzinfo=tz)
        return datetime, date_end

    @classmethod
    def to_date_range(self, name, *, reference=None, tz=None) -> datetime.date:
        from datetime import datetime, timedelta, date, date as Date
        reference = reference or datetime.now().astimezone(tz).replace(tzinfo=None)
        today = reference.date()
        name = name.lower()
        
        if match := re.fullmatch("(\d{4})-(\d{2})-(\d{2})", name):
            day = date(*map(int, match.groups()))
            return day, day + timedelta(days=1)

        if (match_eu := re.fullmatch(r"(\d{1,2}) (%s)( (\d{4}))?" % '|'.join(map(re.escape, self.months_value)), name)) or \
           (match_us := re.fullmatch(r"(%s) (\d{1,2})( (\d{4}))?" % '|'.join(map(re.escape, self.months_value)), name)):
            if match := match_eu:
                dstr,mstr,_,ystr = match.groups()
            elif match := match_us:
                mstr,dstr,_,ystr = match.groups()
            if not ystr:
                y = today.year
            else:
                y = int(ystr)
            d = int(dstr)
            m = self.months_value[mstr]
            day = date(y, m, d)
            return day, day + timedelta(days=1)
        
        if name in ("today", "auj", "aujourdhui", "aujourd'hui"):
            return today, today + timedelta(days=1)
        
        if name in ("week", "semaine"):
            beg = today
            end = today + timedelta(days=7)
            return beg, end
        
        if name in ("tomorrow", "demain"):
            return today + timedelta(days=1), today + timedelta(days=2)
        
        if name in ('future', 'futur'):
            return today, date.max - timedelta(days=7)
        
        if name in ('past', 'passé'):
            return date.min + timedelta(days=7), today
        
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
ParsedEventMiddle = namedtuple('ParsedEventMiddle', 'date time name day_of_week')
ParsedEventFinal = namedtuple('ParsedEventFinal', 'date_str, time, name, date, date_end, datetime, datetime_utc, tz')

def parse_event_date(args):
    """
    ['Something', 'A', 'B', 'C'] -> 'Something', ['A', 'B', 'C']  # n = 1
    ['25', 'November', 'A', 'B', 'C'] -> '25 November', ['A', 'B', 'C']  # n = 2
    ['25', 'November', '2023', 'A', 'B', 'C'] -> '25 November 2023', ['A', 'B', 'C']  # n = 3
    """
    Args = GetOrEmpty(args)
    if Args[0].lower() in DatetimeText.days_english + DatetimeText.days_french:
        day_of_week = Args[0]
        args = args[1:]
        Args = GetOrEmpty(args)
    else:
        day_of_week = ''

    if Args[0].isdecimal() and Args[1].lower() in DatetimeText.months_value \
    or Args[1].isdecimal() and Args[0].lower() in DatetimeText.months_value:
        if Args[2].isdecimal() and len(Args[2]) == 4:
            n = 3
        else:
            n = 2
    else:
        if day_of_week:
            n = 0
        else:
            n = 1
    
    return day_of_week, ' '.join(args[:n]) if n > 0 else day_of_week, args[n:]

def parse_event(args) -> tuple[str, datetime.time | None, str]:
    from datetime import date as Date, time as Time, timedelta as Timedelta

    date: str
    rest: list
    day_of_week, date, rest = parse_event_date(args)
    if match := re.compile('(\\d{1,2})[:hH](\\d{2})?').fullmatch(get_or_empty(rest, 0)):
        hours, minutes = match.group(1), match.group(2)
        time = Time(int(hours), int(minutes or '0'))
        rest = rest[1:]
    else:
        time = None
    name = " ".join(rest)

    return ParsedEventMiddle(date=date, time=time, name=name, day_of_week=day_of_week)

def raise_error(error):
    raise error


def induce_my_timezone(*, user_id, chat_id):
    if tz := get_my_timezone(user_id):
        return tz
    elif tzs := read_settings("event.timezones", id=chat_id, settings_type='chat'):
        if len(tzs) == 1:
            return tzs[0]
    raise UserError(
        "I don't know your timezone and the chat doesn't have one and only one timezone.\n"
        "\n"
        "Set your timezone by typing...\n"
        "- This: /mytimezone TIMEZONE\n"
        "- Example: /mytimezone Europe/Brussels\n"
        "- Example: /mytimezone America/Los_Angeles\n"
        "\n"
        "Or set the chat timezone by typing...\n"
        "- This: /chatsettings event.timezones TIMEZONE\n"
        "- Example: /chatsettings event.timezones Europe/Brussels\n")

def parse_datetime_point(update, context, when_infos=None, what_infos=None):
    from datetime import datetime as Datetime, time as Time, date as Date, timedelta
    tz = induce_my_timezone(user_id=update.message.from_user.id, chat_id=update.effective_chat.id)
    
    if when_infos is None and what_infos is None:
        date_str, time, name, day_of_week = parse_event(context.args)
    else:
        date_str, time, name_from_when_part, day_of_week = parse_event(when_infos.split())
        if name_from_when_part:
            raise UserError("Too much infos in the When part")
        name = what_infos or ''

    date, date_end = DatetimeText.to_date_range(date_str, tz=tz)
    datetime = Datetime.combine(date, time or Time(0,0)).replace(tzinfo=tz)
    datetime_utc = datetime.astimezone(ZoneInfo('UTC'))
    Loc = locals()

    if day_of_week:
        if not is_correct_day_of_week(date, day_of_week):
            raise UserError(f"{date_str!r} is not a {day_of_week!r}")

    return ParsedEventFinal(**{x: Loc[x] for x in ParsedEventFinal._fields})

def is_correct_day_of_week(date, day_of_week):
    return date.weekday() == (DatetimeText.days_english + DatetimeText.days_french).index(day_of_week.lower()) % 7

import sqlite3
async def add_event(update: Update, context: CallbackContext):
    send = make_send(update, context)
    read_chat_settings = make_read_chat_settings(update, context)
    read_my_settings = make_read_my_settings(update, context)
    
    if not context.args:
        if reply := get_reply(update.message):
            infos_event = addevent_analyse(update, context)
        else:
            return await send("Usage: /addevent date name\nUsage: /addevent date hour name")
    else:
        infos_event = None

    if infos_event is not None:
        other_infos = {k: infos_event[k] for k in infos_event.keys() - {'when', 'what', 'where'}}        
        when_infos = infos_event.get('when') or ''
        what_infos = ' '.join(filter(None, [
            infos_event.get('what') or '',
        ] + [
            '{%s: %s}' % (item[0].capitalize(), item[1])
            for item in other_infos.items()
        ] + [
            "@ " + infos_event['where'] if infos_event.get('where') else '',
        ]))
    else:
        when_infos = None
        what_infos = None

    source_user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    date_str, time, name, date, date_end, datetime, datetime_utc, tz = parse_datetime_point(update, context, when_infos=when_infos, what_infos=what_infos)
    
    chat_timezones = read_chat_settings("event.timezones")
    
    if chat_timezones and tz and tz not in chat_timezones:
        raise UserError('\n'.join([
            'Your timezone is not in chat timezones, this can be confusing, change your timezone or add your timezone to the chat timezones.',
            '- Your timezone: {tz}'.format(tz=tz),
            '- Chat timezones: {chat_timezone_str}'.format(chat_timezone_str=", ".join(map(str, chat_timezones))),
        ]))

    with sqlite3.connect('db.sqlite') as conn:
        cursor = conn.cursor()

        strftime = DatetimeDbSerializer.strftime

        cursor.execute("INSERT INTO Events(date, name, chat_id, source_user_id) VALUES (?,?,?,?)", (strftime(datetime_utc), name, chat_id, source_user_id))
    
    emojis = EventFormatting.emojis

    # 1. Send info in text

    await send('\n'.join(filter(None, [
        f"Event added:",
        f"{emojis.Name} {name}",
        f"{emojis.Date} {datetime:%A} {datetime.date():%d/%m/%Y} ({date_str})",
        (f"{emojis.Time} {time:%H:%M} ({tz})" if chat_timezones and set(chat_timezones) != {tz} else
         f"{emojis.Time} {time:%H:%M}") if time else None
    ] + ([
        f"{emojis.Time} {datetime_tz:%H:%M} ({timezone})" if datetime_tz.date() == datetime.date() else
        f"{emojis.Time} {datetime_tz:%H:%M} on {datetime_tz.date():%d/%m/%Y} ({timezone})"
        for timezone in chat_timezones or []
        if timezone != tz
        for datetime_tz in [datetime.astimezone(timezone)]
    ] if time else []))))
    
    if do_unless_setting_off(read_chat_settings('event.addevent.display_file')):
        # 2. Send info as clickable ics file to add to calendar
        if do_unless_setting_off(read_chat_settings('event.addevent.help_file')):
            await send('Click the file below to add the event to your calendar:')
        await export_event(update, context, name=name, datetime_utc=datetime_utc)

def natural_filter(x):
    return filter(None, x)

import yaml
def addevent_analyse_yaml(update, context, text:str) -> {'what': str, 'when': str}:
    text = '\n'.join(l for l in text.splitlines() if ':' in l)
    Y = yaml.safe_load(text)
    if not isinstance(Y, dict):
        raise EventAnalyseError('Each line should have a colon symbol, example:\n\nWhat: Party\nWhen: Tomorrow 16h')
    
    Y = {k.lower(): v for k,v in Y.items()}

    result = {}
    keys_lower = {k.lower(): k for k in Y.keys()}
    possibles = EventInfosAnalyse.possibles
    for field in possibles:
        if field in keys_lower:
            result[possibles[field].lower()] = Y.get(keys_lower[field], '')
    
    for field in Y.keys() - possibles.keys():
        result[field.lower()] = Y[field]
    
    if not result.get('when'):
        raise EventAnalyseError("When is mandatory")
    
    Interval = re.compile('(\d{2}:\d{2}) - (\d{2}:\d{2})')
    if match := Interval.search(result['when']):
        result['what'] = ' '.join(natural_filter([result.get('what'), '({})'.format(match.group(0))]))
        result['when'] = Interval.sub(match.group(1), result['when'])

    return result

def only_one(it, error=ValueError):
    if len(L := list(it)) == 1:
        return L[0]
    else:
        raise error

def only_one_with_error(error):
    return partial(only_one, error=error)

def addevent_analyse_from_bot(update, context, text:str) -> {'what': str, 'when': str}:
    my_timezone = induce_my_timezone(user_id=update.message.from_user.id, chat_id=update.effective_chat.id)

    lines = GetOrEmpty(text.splitlines())
    if lines[0] in ("Event!", "Event added:"):
        del lines[0]
        
    re_pattern = (
        '^'
        + '({})'.format('|'.join(map(re.escape, EventInfosAnalyse.emojis_meaning)))
        + '\\s*'
        + '(.*)'
    )
    from collections import defaultdict
    infos_raw = defaultdict(list)
    Re = re.compile(re_pattern, re.I)
    for line in lines:
        if match := Re.match(line):
            infos_raw[EventInfosAnalyse.emojis_meaning[match.group(1)].lower()].append(match.group(2))

    def deal_with_timezones(infos_raw):
        infos_raw = infos_raw.copy()

        def extract_timezone(data):
            Re2 = re.compile(
                '(.*)'
                + '\s*'
                + re.escape('(') + '(' + '.*?' + ')' + re.escape(')')
            )
            if match := Re2.search(data):
                time_str, tz = match.group(1), match.group(2)
                try:
                    tz = ZoneInfo(tz)
                except ZoneInfoNotFoundError:
                    tz = None
                if not tz:
                    return time_str.strip(), None
                return time_str, tz
            return data.strip(), None
        
        if 'date' in infos_raw:
            ldate = infos_raw['date']
            local_only_one = only_one_with_error(EventAnalyseError(f'Multiple values for timezone {my_timezone!s}'))
            all_timezone_info = [local_only_one(x[0] for x in infos_raw['date'] if my_timezone == x[1])]
            local_time, local_time_tz = extract_timezone(all_timezone_info)
            infos_raw['date'] = [ '{} ({})'.format(local_time, local_time_tz) ]

        return infos_raw
    
    def reduce_multi_values(infos_raw):
        return {k: ' & '.join(v) for k,v in infos_raw.items()} 

    # infos_raw = deal_with_timezones(infos_raw)  # TODO
    infos_raw = reduce_multi_values(infos_raw)

    what = infos_raw.get('name', '')
    try:
        when = infos_raw['date'] + ((' ' + infos_raw['time']) if infos_raw.get('time') else '')
    except KeyError:
        raise EventAnalyseError("Missing Date in message")

    if match := re.search('\s*' + re.escape('(') + '.*' + re.escape(')'), when):
        when = when[:match.span(0)[0]] + when[match.span(0)[1]:]

    if match := re.match('|'.join(map(re.escape, DatetimeText.days_english)), when, re.I):
        when = when[:match.span(0)[0]] + when[match.span(0)[1]:]

    if match := re.search('(' + '\d+' + re.escape('/') + '\d+' + re.escape('/') + '\d+' + ')', when):
        when = when[:match.span(0)[0]] + '-'.join(reversed(match.group(1).split('/'))) + when[match.span(0)[1]:]

    when = when.strip()

    return {
        'what': what,
        'when': when,
    }

def enrich_event_with_where(event):
    if event.get('where'):
        return event
    event = dict(event)
    event['where'] = GetOrEmpty(re.compile('(?:[ ]|^)[@][ ]').split(event['what']))[1]
    if not event.get('where'):
        del event['where']
    return event

def addevent_analyse(update, context):
    if not (reply := get_reply(update.message)):
        raise UserError("Cannot analyse if there is nothing to analyse")

    exceptions = []
    try:
        return addevent_analyse_yaml(update, context, reply.text)
    except yaml.error.YAMLError as e:
        exceptions.append(EventAnalyseError("YAML Error:" + str(e)))
    except EventAnalyseError as e:
        exceptions.append(e)
    
    try:
        return addevent_analyse_from_bot(update, context, reply.text)
    except EventAnalyseError as e:
        exceptions.append(e)
    
    if exceptions:
        raise exceptions[0] if len(exceptions) == 1 else EventAnalyseMultipleError(exceptions)
    else:
        raise EventAnalyseError("I cannot interpret this message as an event")

async def whereisto(update, context, *, command: Literal['whereis', 'whereto']):
    send = make_send(update, context)

    key = None
    if reply := get_reply(update.message):
        try:
            infos_event = addevent_analyse(update, context)
            infos_event = enrich_event_with_where(infos_event)
            key = infos_event.get('where', None)  
        except UserError as e:
            reply_error = e

    if key is None:
        try:
            keys = context.args
            key = ' '.join(keys)
            if not key:
                raise ValueError
        except ValueError:
            return await send("Usage: /whereis place\n/whereis (on a event message)")
    
    key: str

    if command == 'whereis':
        results = simple_sql(('select value from EventLocation where chat_id=? and LOWER(key)=LOWER(?)', (chat_id := update.effective_chat.id, key,)))
        await send("I don't know ! :)" if not results else "→ " + only_one(results)[0])
    
    elif command == 'whereto':
        current_key = key
        results = []
        while True:
            current_result = simple_sql(('select value from EventLocation where chat_id=? and LOWER(key)=LOWER(?)', (chat_id := update.effective_chat.id, current_key, )))
            if not current_result:
                break
            current_key = current_result[0][0]
            if current_key in results:
                results.append(current_key)
                break  # we stop because there is a loop
            else:
                results.append(current_key)
                continue
        await send("I don't know ! :)" if not results else '\n'.join(map("→ {}".format, results)))

    else:
        raise AssertionError

async def whereis(update:Update, context:CallbackContext):
    return await whereisto(update:=update, context=context, command='whereis') 

async def whereto(update:Update, context:CallbackContext):
    return await whereisto(update:=update, context=context, command='whereto') 

async def thereis(update:Update, context:CallbackContext):
    if reply := get_reply(update.message):
        if reply.text.startswith('/whereis') or reply.text.startswith("/whereis@" + context.bot.username):
            value = ' '.join(context.args)
            key = GetOrEmpty(reply.text.split(maxsplit=1))[1]
        else:
            value = reply.text
            key = ' '.join(context.args)

    else:
        for tries in (1, 2, 3):
            try:
                match tries:
                    case 1:
                        key, value = context.args
                    case 2:
                        i = context.args.index('=')
                        keys, values = context.args[:i], context.args[i+1:]
                        key = ' '.join(keys)
                        value = ' '.join(values)
                    case 3:
                        key, *values = context.args
                        value = ' '.join(values)
                break
            except ValueError:
                continue
        else:
            return await send("Usage: /thereis place location")
    
    await save_thereis(key, value, update=update, context=context)

async def save_thereis(key, value, *, update, context):
    send = make_send(update, context)
    chat_id = update.effective_chat.id
    
    assert_true(key and value, UserError("Key and Values must be non null"))
    
    with sqlite3.connect("db.sqlite") as conn:
        my_simple_sql = partial(simple_sql, connection=conn)
        conn.execute('begin transaction')
        
        if my_simple_sql(('select * from EventLocation where chat_id=? and LOWER(key)=LOWER(?)', (chat_id, key))):
            my_simple_sql(('delete from EventLocation where chat_id=? and LOWER(key)=LOWER(?)', (chat_id, key)))
        my_simple_sql(('insert into EventLocation(key, value, chat_id) VALUES (?,?,?)', (key, value, chat_id)))
        conn.execute('end transaction')

    await send(f"Elephant remembers location:\n{key!r}\n→ {value!r}")

from datetime import datetime, timedelta
def sommeil(s, *, command) -> tuple[datetime, datetime]:
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

def parse_datetime_range(update, *, args, default="week"):
    if len(args) >= 2:
        raise UserError("<when> must be a day of the week or week")
    
    from datetime import date as Date, time as Time, datetime as Datetime

    if not args:
        when = default
    else:
        when, = args  # beware of the ","
    time = Time(0, 0)
    tz = induce_my_timezone(user_id=update.message.from_user.id, chat_id=update.effective_chat.id)
    
    beg_date, end_date = DatetimeText.to_date_range(when, tz=tz)
    beg_local, end_local = Datetime.combine(beg_date, time), Datetime.combine(end_date, time)
    
    beg, end = (x.replace(tzinfo=tz).astimezone(ZoneInfo('UTC')) for x in (beg_local, end_local))
    
    return dict(beg_utc=beg, end_utc=end, tz=tz, when=when, beg_local=beg_local, end_local=end_local)  # | {x: locals()[x] for x in ()}

async def next_or_last_event(update: Update, context: CallbackContext, n:int):
    from datetime import datetime as Datetime
    send = make_send(update, context)
    read_chat_settings = make_read_chat_settings(update, context)

    datetime_str = None
    skip_n = None
    if len(context.args) == 0:
        pass
    elif len(context.args) == 1:
        first_arg, = context.args
        try:
            skip_n = int(first_arg)
        except ValueError:
            pass
        if not skip_n > 0:
            raise UserError('n must be > 0')
        if skip_n is None:
            datetime_str = first_arg
            if len(datetime_str) <= len('2020-01-01'):
                datetime_str += ' ' + '00:00:00'
    elif len(context.args) == 2:
        date, hour = context.args
        if len(hour) <= len('08:00'):
            hour += ':00'
        datetime_str = date + ' ' + hour
    else:
        raise UserError("Usage: /nextevent\n/nextevent datetime\n/nextevent n")
    if skip_n is None:
        skip_n = 1
    
    chat_timezones = read_chat_settings("event.timezones")
    tz = induce_my_timezone(user_id=update.message.from_user.id, chat_id=update.effective_chat.id)
    now = Datetime.now(UTC) if not datetime_str else DatetimeDbSerializer.strptime(datetime_str.replace('T', ' ')).replace(tzinfo=tz).astimezone(ZoneInfo('UTC'))

    events = simple_sql_dict(('''
        SELECT date as date_utc, name as name
        FROM Events
        WHERE %s
        AND chat_id=?
        ORDER BY date %s, rowid DESC
        LIMIT 1
        OFFSET ?
    ''' % ({1: "date >= ?", -1: "date <= ?"}[n], {1: 'ASC', -1:'DESC'}[n]), (now, update.effective_chat.id, skip_n - 1)))

    if len(events) == 0:
        return await send("No event !")

    strptime = DatetimeDbSerializer.strptime

    date_utc, name = events[0]
    datetime = strptime(date_utc).replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)
    date, time = datetime.date(), datetime.time()

    emojis = EventFormatting.emojis
    await send('\n'.join(natural_filter([
        f"Event!",
        f"{emojis.Name} {name}",
        f"{emojis.Date} {datetime:%A} {datetime.date():%d/%m/%Y}",
        (f"{emojis.Time} {time:%H:%M} ({tz})" if chat_timezones and set(chat_timezones) != {tz} else
         f"{emojis.Time} {time:%H:%M}") if time else None
    ] + ([
        f"{emojis.Time} {datetime_tz:%H:%M} ({timezone})" if datetime_tz.date() == datetime.date() else
        f"{emojis.Time} {datetime_tz:%H:%M} on {datetime_tz.date()} ({timezone})"
        for timezone in chat_timezones or []
        if timezone != tz
        for datetime_tz in [datetime.astimezone(timezone)]
    ] if time else []))))

async def last_event(update, context):
    return await next_or_last_event(update, context, -1)

async def next_event(update, context):
    return await next_or_last_event(update, context, 1)

async def list_days(update: Update, context: CallbackContext):
    return await list_days_or_today(update, context, mode='list')

def setting_on_off(s, default):
    return (s if isinstance(s, bool) else
            True if isinstance(s, str) and s.lower() == 'on' else
            False if isinstance(s, str) and s.lower() == 'off' else 
            setting_on_off(default, default=False) if isinstance(default, str) and default.lower() in ('on', 'off') else
            default)

def do_if_setting_on(setting):
    return setting_on_off(setting, default=False)

def do_unless_setting_off(setting):
    return setting_on_off(setting, default=True)

from typing import Literal
async def list_days_or_today(update: Update, context: CallbackContext, mode: Literal['list', 'today']):
    assert mode in ('list', 'today')

    send = make_send(update, context)
    
    datetime_range = parse_datetime_range(update, args=context.args if mode == 'list' else ('today',) if mode == 'today' else raise_error(AssertionError('mode must be a correct value')))
    beg, end, tz, when = (datetime_range[x] for x in ('beg_utc', 'end_utc', 'tz', 'when'))

    strptime = DatetimeDbSerializer.strptime
    strftime = DatetimeDbSerializer.strftime

    events = simple_sql_dict(('''
        SELECT date, name
        FROM Events
        WHERE ? <= date AND date < ?
        AND chat_id=?
        ORDER BY date''',
        (strftime(beg), strftime(end), update.effective_chat.id,)))

    from collections import defaultdict
    days = defaultdict(list)
    for event in events:
        date = strptime(event['date']).replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        event_name = event['name']
        days[date.timetuple()[:3]].append((date, event_name))

    read_chat_settings = make_read_chat_settings(update, context)
    display_time_marker = False if mode == 'list' else do_unless_setting_off(read_chat_settings('event.listtoday.display_time_marker'))

    now_tz = datetime.now().astimezone(tz)
    def is_past(event_date):
        return event_date <= now_tz
    
    days_as_lines = []
    for day in sorted(days):
        date = days[day][0][0]
        day_of_week = DatetimeText.days_english[date.weekday()]
        days_as_lines.append(
            f"{day_of_week.capitalize()} {date:%d/%m}"
            + "\n"
            + "\n".join(f"-{marker} {event_date:%H:%M}: {event_name}" for event_date, event_name in days[day] for marker in ['>' if display_time_marker and is_past(event_date) else '']))
    
    msg = '\n\n'.join(days_as_lines)

    chat_timezones = read_chat_settings("event.timezones")

    if msg and chat_timezones and set(chat_timezones) != {tz}:
        msg += '\n\n' + f'Timezone: {tz}'

    await send(msg or (
        "No events for the next 7 days !" if when == 'week' else
        f"No events for {when} !" + (" 😱" if "today" in (mode, when) else "")
    ))

async def list_today(update: Update, context: CallbackContext):
    return await list_days_or_today(update, context, mode='today')

async def list_events(update: Update, context: CallbackContext):
    send = make_send(update, context)
    
    datetime_range = parse_datetime_range(update, args=context.args)
    beg, end, tz, when = (datetime_range[x] for x in ('beg_utc', 'end_utc', 'tz', 'when'))

    chat_id = update.effective_chat.id
    with sqlite3.connect('db.sqlite') as conn:
        strptime = DatetimeDbSerializer.strptime
        strftime = DatetimeDbSerializer.strftime
        
        cursor = conn.cursor()
        query = ("""SELECT date, name
                    FROM Events
                    WHERE ? <= date AND date < ?
                    AND chat_id = ?
                    ORDER BY date""",
                (strftime(beg), strftime(end), chat_id))
        
        read_chat_settings = make_read_chat_settings(update, context)
        chat_timezones = read_chat_settings("event.timezones")
        msg = '\n'.join(f"- {DatetimeText.days_english[date.weekday()]} {date:%d/%m}: {event}" if not has_hour else 
                        f"- {DatetimeText.days_english[date.weekday()]} {date:%d/%m %H:%M}: {event}"
                        for date_utc, event in cursor.execute(*query)
                        for date in [strptime(date_utc).replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)]
                        for has_hour in [True])
        if msg and chat_timezones and set(chat_timezones) != {tz}:
            msg += '\n\n' + f"Timezone: {tz}"
        await send(msg or (
            "No events for the next 7 days !" if when == 'week' else
            f"No events for {when} !" + (" 😱" if "today" == when else "")
        ))

async def delevent(update, context):
    send = make_send(update, context)

    if reply := get_reply(update.message):
        return await send("Not implemented yet but will allow to deleent an event by responding to it.")

    strptime = DatetimeDbSerializer.strptime

    strftime = DatetimeDbSerializer.strftime
    strftime_minutes = DatetimeDbSerializer.strftime_minutes

    datetime_range = parse_datetime_range(update, args=context.args, default="future")
    beg, end, tz = datetime_range['beg_utc'], datetime_range['end_utc'], datetime_range['tz']
    events = simple_sql_dict(('''
        SELECT rowid, date, name
        FROM Events
        WHERE ? <= date AND date < ?
        AND chat_id=?
        ORDER BY date''',
        (strftime(beg), strftime(end), update.effective_chat.id,)))
   
    saved_info_dict: dict = make_send_save_info(update, context)._asdict()

    keyboard = [
        [InlineKeyboardButton("{} - {}".format(
            strftime_minutes(strptime(event['date']).replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)),
            event['name']
        ), callback_data=json.dumps(saved_info_dict | dict(rowid=str(event['rowid']))))]
        for event in events
    ]

    if not keyboard:
        await send("No events to delete !")
        return ConversationHandler.END
    
    cancel = [[InlineKeyboardButton("/cancel", callback_data=json.dumps(saved_info_dict | dict(rowid="null")))]]

    await send("Choose an event to delete:", reply_markup=InlineKeyboardMarkup(keyboard + cancel))

    return 0

async def do_delete_event(update, context):
    query = update.callback_query
    await query.answer()

    data_dict: dict = json.loads(query.data)
    send = make_send(update, context, save_info=SendSaveInfo(chat_id=data_dict['chat_id'], thread_id=data_dict['thread_id']))
    
    rowid = data_dict["rowid"]
    if rowid == "null":
        await send("Cancelled: No event deleted")
    else:
        await db_delete_event(update, context, send, chat_id=update.effective_chat.id, event_id=rowid)

    # await query.edit_message_text
    # await query.edit_message_reply_markup()
    await query.delete_message()
        
    return ConversationHandler.END

async def db_delete_event(update, context, send, *, chat_id, event_id):
    read_chat_settings = make_read_chat_settings(update, context)

    if read_chat_settings('event.delevent.display'):
        infos = dict(only_one(simple_sql_dict(('select date, name from Events where chat_id = ? and rowid = ?', (chat_id, event_id, )))))
    else:
        infos = None

    simple_sql(('delete from Events where chat_id = ? and rowid = ?', (chat_id, event_id)))
    
    await send(f"Event deleted" if infos is None else f"Event deleted: {infos}")

from typing import Iterable
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

def set_settings(*, id, key, value_raw:any, settings_type:Literal['chat'] | Literal['user']):
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

def delete_settings(*, id, key, settings_type:Literal['chat'] | Literal['user']):
    table = SettingsInfo.TABLES[settings_type]
    field_id = SettingsInfo.FIELDS[settings_type]
    query_delete = (f"""DELETE FROM {table} WHERE {field_id}=? and key=?""", (id, key))
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute(*query_delete)

async def mytimezone(update: Update, context: CallbackContext):
    send = make_send(update, context)

    if not context.args:
        # get timezone
        tz = get_my_timezone_from_timezone_table(update.message.from_user.id)
        base_text = ("You don't have any timezone set.\n"
                     "Use /mytimezone Continent/City to set it.\n"
                     "Example: /mytimezone Europe/Brussels\n"
                     "Example: /mytimezone America/Los_Angeles" if tz is None else
                     "Your timezone is: {}".format(tz))
        return await send(base_text)
    else:
        # set timezone
        for tries in (1, 2):
            match tries:
                case 1:
                    tz_name, *_ = context.args
                case 2:
                    tz_continent, tz_city, *_ = context.args
                    tz_name = tz_continent + "/" + tz_city
            try:
                tz = ZoneInfo(tz_name)
                break
            except ZoneInfoNotFoundError:
                continue
            except Exception as e:
                if isinstance(e, IsADirectoryError):
                    continue
                else:
                    raise e
        else:
            raise UserError("This timezone is not known by the system.\nCorrect examples include:\n- America/Los_Angeles\n- Europe/Brussels")
        set_my_timezone(update.message.from_user.id, tz)
        return await send("Your timezone is now: {}".format(tz))

def remove_dup_keep_order(it):
    S = set()
    for x in it:
        if x not in S:
            S.add(x)
            yield x

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
    'event.timezones',
    'event.addevent.help_file',
    'event.addevent.display_file',
    'event.listtoday.display_time_marker',
    'event.delevent.display',
) + tuple(
    remove_dup_keep_order(setting + '.active' for _, setting, _ in RESPONDERS)
)

assert len(set(ACCEPTED_SETTINGS_USER)) == len(ACCEPTED_SETTINGS_USER), "Duplicates in ACCEPTED_SETTINGS_USER" 
assert len(set(ACCEPTED_SETTINGS_CHAT)) == len(ACCEPTED_SETTINGS_CHAT), "Duplicates in ACCEPTED_SETTINGS_CHAT" 

def assert_true(condition, error=AssertionError):
    if not condition:
        raise error
    return True

def is_timezone(x: str) -> bool:
    try:
        ZoneInfo(x)
        return True
    except ZoneInfoNotFoundError:
        return False

def CONVERSION_SETTINGS_BUILDER():
    import json
    # serializers
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
        'to_db': lambda x: assert_true(is_timezone(x), UserError(f"{x} is not a timezone"))
                 and x,
    }
    list_of_timezone_serializer = {
        'from_db': lambda s: list(map(ZoneInfo, json.loads(s))),
        'to_db': lambda L: json.dumps(list(map(timezone_serializer['to_db'], L)))
    }
    list_of_currencies_serializer = {
        'from_db': lambda s: list(map(str.upper, json.loads(s))),
        'to_db': lambda L: json.dumps(list(map(str.upper, L)))
    }
    on_off_serializer = {
        'from_db': lambda x: x != 'off',
        'to_db': lambda x: assert_true(isinstance(x, str) and x.lower() in ('on', 'off', 'true', 'false', 'yes', 'no'), UserError(f"{x} must be on/off"))
                 and {'true': 'on', 'false': 'off', 'yes': 'on', 'no': 'off'}.get(x.lower(), x.lower()),
    }
    # mappings
    mapping_chat = {
        'money.currencies': list_of_currencies_serializer,
        'event.timezones': list_of_timezone_serializer,
        'event.addevent.display_file': on_off_serializer,
        'event.delevent.display': on_off_serializer,
    }
    mapping_user = {
        'event.timezone': timezone_serializer,
        'dict.engine': {
            'from_db': lambda x: x,
            'to_db': lambda x: x if assert_true(x in DICT_ENGINES, UserError("{!r} is not a known engine.\nAvaiblable options: {}".format(x, ', '.join(DICT_ENGINES)))) else None
        }
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

SettingType = Literal['chat', 'user']

def read_settings(key, *, id, settings_type: SettingType):
    conversion = CONVERSION_SETTINGS[settings_type][key]['from_db']
    raw = read_raw_settings(key, id=id, settings_type=settings_type)
    return conversion(raw) if raw is not None else None

def read_raw_settings(key, *, id, settings_type: SettingType):
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
        return results[0][0] if results else None

async def listallsettings(update: Update, context: CallbackContext):
    send = make_send(update, context)
    await send('\n'.join("- {} ({})".format(
            setting,
            '|'.join(['user'] * (setting in ACCEPTED_SETTINGS_USER) + ['chat'] * (setting in ACCEPTED_SETTINGS_CHAT)))
        for setting in sorted(ACCEPTED_SETTINGS_USER + ACCEPTED_SETTINGS_CHAT)))

async def settings_command(update: Update, context: CallbackContext, *, command_name: str, settings_type:Literal['chat'] | Literal['user'], accepted_settings:list[str]):
    send = make_send(update, context)

    async def print_usage():
        await send(f"Usage:\n/{command_name} command.key\n/{command_name} command.key value")

    if len(context.args) == 0:
        return await print_usage()

    key, *rest = context.args

    if key not in accepted_settings:
        return await send(f'Unknown settings: {key!r}\n\nType /listallsettings for complete list of settings (hidden command)')

    if rest and rest[0] == '=':
        rest = rest[1:]
        if not rest:
            raise UserError('Must sepcify a value when setting a value with "a.b = c"')

    if key in ('money.currencies', 'event.timezones'):
        value = ([] if list(rest) in [['()'], ['[]']] else
                 None if not rest else
                 list(rest))
    else:
        if len(rest) not in (0, 1):
            return await print_usage()
        # default, single value no space string
        value = rest[0] if rest else None

    if settings_type == 'user':
        id = update.message.from_user.id
    elif settings_type == 'chat':
        id = update.effective_chat.id
    else:
        raise ValueError(f'Invalid settings_type: {settings_type}')

    if value is None:
        # read
        value = read_raw_settings(id=id, key=key, settings_type=settings_type)
        await send(f'Settings: {key} = {value}' if value is not None else
                    f'No settings for {key!r}')
    
    else:
        # write value
        set_settings(value_raw=value, id=id, key=key, settings_type=settings_type)
        await send(f"Settings: {key} = {value}")

async def delsettings_command(update:Update, context: CallbackContext, *, accepted_settings:list[str], settings_type:Literal['chat'] | Literal['id'], command_name:str):
    send = make_send(update, context)
    
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

def migration9():
    with sqlite3.connect('db.sqlite') as conn:
        conn.execute('begin transaction')
        conn.execute('create table EventLocation(key, value, chat_id)')
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

    def to_db(self, x: datetime):
        return self.strftime(x)

    def from_db(self, x: any):
        return self.strptime(x)
    
    @staticmethod
    def strftime_minutes(x:datetime):
        return x.strftime("%Y-%m-%d %H:%M")

class JsonDbSerializer:
    def to_db(self, x: json):
        import json
        return json.dumps(x)

    def from_db(self, x: any):
        import json
        return json.loads(x)

Rates = dict
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
        send = make_send(update, context)
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
    send = make_send(update, context)
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
    on_off = Args[0]
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
    send = make_send(update, context)

    fmt = ('{} - {}' if '--botfather' in context.args else
           '/{} {}')
    
    return await send('\n'.join(fmt.format(command, COMMAND_DESC.get(command, command)) for command in COMMAND_LIST))

class UserError(ValueError):
    pass

class DictJsLike(dict):
    def __getattribute__(self, x):
        if x in self:
            return self[x]
        return super().__getattribute__(x)

class EventFormatting:
    emojis = DictJsLike(
         Name="📃",
         Time="⌚",
         Date="🗓️",
         Location="📍",
    )

class EventInfosAnalyse:
    possibles = {
        'what': 'what', 'when': 'when', 'where': 'where',
        'quand': 'when', 'quoi': 'what', 'où': 'where',
        'name': 'what', 'location': 'where', 
    }

    emojis_meaning = {y:x for x,y in EventFormatting.emojis.items()}

class EventAnalyseError(UserError):
    pass

class EventAnalyseMultipleError(EventAnalyseError):
    def __init__(self, exceptions):
        self.exceptions = exceptions
        
    def __str__(self):
        return '\n---\n'.join(map(str, self.exceptions))

async def log_error(error, send):
    if isinstance(error, UserError):
        return await send("Error: {}".format(error))
    else:
        logging.error("Error", exc_info=error)
        return await send("An unknown error occured in your command, ask @robertvend to fix it !")

async def general_error_callback(update:Update, context:CallbackContext):
    send = make_send(update, context)
    async def send_on_error(m):
        if update and update.effective_chat:
            await send(m)
    
    return await log_error(context.error, send_on_error)

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
    "nextevent": "Display the next event in emoji row format",
    "lastevent": "Display the last event in emoji row format",
    "listevents": "List events",
    "listdays": "List events grouped by days",
    "listtoday": "Shortcut for /listdays today, can add time marker",
    "today": "Shortcut for /listtoday",
    "whereis": "Remember a place/directions for events",
    "thereis": "Set a place a place/directions for events",
    "whereto": "Remember a place/directions for events in cascade mode",
    "delevent": "Delete event",
    "ru": "Latin alphabet to Cyrillic using Russian convention",
    "dict": "Shows definition of each word using dictionary and settings engine",
    "wikt": "Shows definition of each word using wiktionary",
    "larousse": "Show definition of each word using french dictionary Larousse.fr",
    'eur': "Convert euros to other currencies",
    'brl': "Convert brazilian reals to other currencies",
    'rub': "Convert russian rubles to other currencies",
    'convertmoney': 'Convert money to chat currencies or to specific currency',
    "mytimezone": "Set your timezone to use events commands",
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
    'addevent', 'nextevent', 'lastevent', 'listevents', 'listdays', 'listtoday', 'today', 'delevent',
    'whereis', 'thereis',
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
    
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler('befluent', befluent)],
        states = {
            
        },
        fallbacks=[

        ]
    ))
    application.add_handler(CommandHandler('caps', caps))
    application.add_handler(CommandHandler('addevent', add_event))
    application.add_handler(CommandHandler('nextevent', next_event))
    application.add_handler(CommandHandler('listevents', list_events))
    application.add_handler(CommandHandler('listdays', list_days))
    application.add_handler(CommandHandler('listoday', list_today)) # hidden command, for typo
    application.add_handler(CommandHandler('listtoday', list_today))
    application.add_handler(CommandHandler('today', list_today))
    application.add_handler(CommandHandler('lastevent', last_event))
    application.add_handler(CommandHandler('whereis', whereis))
    application.add_handler(CommandHandler('whereto', whereto))
    application.add_handler(CommandHandler('thereis', thereis))
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("delevent", delevent)],
        states={
            0: [CallbackQueryHandler(do_delete_event)],
        },
        fallbacks=[],
    ), group=2)
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

