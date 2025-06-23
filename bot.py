import random
import pickle
import os
import logging
import traceback
from enum import Enum
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_date, extract_destination, extract_airline, extract_price, \
    is_date_in_range, Stats, logger, lemmatize_phrase, analyze_sentiment
from rapidfuzz import process, fuzz

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')


# Состояния бота
class BotState(Enum):
    NONE = "NONE"
    WAITING_FOR_FLIGHT = "WAITING_FOR_FLIGHT"
    WAITING_FOR_DATE = "WAITING_FOR_DATE"
    WAITING_FOR_INTENT = "WAITING_FOR_INTENT"


# Намерения
class Intent(Enum):
    HELLO = "hello"
    BYE = "bye"
    YES = "yes"
    NO = "no"
    FLIGHT_TYPES = "flight_types"
    FLIGHT_PRICE = "flight_price"
    FLIGHT_AVAILABILITY = "flight_availability"
    FLIGHT_RECOMMENDATION = "flight_recommendation"
    FILTER_FLIGHTS = "filter_flights"
    FLIGHT_INFO = "flight_info"
    ORDER_FLIGHT = "order_flight"
    COMPARE_FLIGHTS = "compare_flights"


# Типы ответов
class ResponseType(Enum):
    INTENT = "intent"
    GENERATE = "generate"
    FAILURE = "failure"


# Класс бота
class Bot:
    def __init__(self):
        """Инициализация моделей."""
        try:
            with open('models/intent_model.pkl', 'rb') as f:
                self.clf = pickle.load(f)
            with open('models/intent_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('models/dialogues_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open('models/dialogues_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open('models/dialogues_answers.pkl', 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Не найдены файлы модели: {e}\n{traceback.format_exc()}")
            raise

    def _update_context(self, context, replica, answer, intent=None, flight_name=None):
        """Обновляет контекст пользователя."""
        context.user_data.setdefault('state', BotState.NONE.value)
        context.user_data.setdefault('current_flight', None)
        context.user_data.setdefault('last_bot_response', None)
        context.user_data.setdefault('last_intent', None)
        context.user_data.setdefault('history', [])

        context.user_data['history'].append(replica)
        context.user_data['history'] = context.user_data['history'][-CONFIG['history_limit']:]
        context.user_data['last_bot_response'] = answer
        if intent:
            context.user_data['last_intent'] = intent
        if flight_name:
            context.user_data['current_flight'] = flight_name

    def classify_intent(self, replica):
        """Классифицирует намерение пользователя."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized:
            return None
        vectorized = self.vectorizer.transform([replica_lemmatized])
        intent = self.clf.predict(vectorized)[0]
        best_score = 0
        best_intent = None
        for intent_key, data in CONFIG['intents'].items():
            examples = [lemmatize_phrase(ex) for ex in data.get('examples', []) if lemmatize_phrase(ex)]
            if not examples:
                continue
            match = process.extractOne(replica_lemmatized, examples, scorer=fuzz.ratio)
            if match and match[1] / 100 > best_score and match[1] / 100 >= CONFIG['thresholds']['intent_score']:
                best_score = match[1] / 100
                best_intent = intent_key
        logger.info(
            f"Classify intent: replica='{replica_lemmatized}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
        return best_intent or intent if best_score >= CONFIG['thresholds']['intent_score'] else None

    def _get_flight_response(self, intent, flight_name, replica, context):
        """Обрабатывает запросы, связанные с конкретным рейсом."""
        if flight_name not in CONFIG['flights']:
            return "Извините, такого рейса нет в каталоге."
        responses = CONFIG['intents'][intent]['responses']
        answer = random.choice(responses)
        flight_data = CONFIG['flights'][flight_name]
        answer = answer.replace('[flight_name]', flight_name)
        answer = answer.replace('[price]', str(flight_data['price']))
        answer = answer.replace('[date]', flight_data['date'])
        answer = answer.replace('[description]', flight_data.get('description', 'удобный рейс'))
        answer = answer.replace('[from]', flight_data['route']['from'])
        answer = answer.replace('[to]', flight_data['route']['to'])

        # Добавляем реакцию на тональность
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            answer += " Кажется, вы не в духе. Может, путешествие поднимет настроение? 😊"

        return f"{answer} Что ещё интересует?"

    def _find_flight_by_context(self, replica, context):
        """Ищет рейс на основе контекста или маршрута."""
        last_response = context.user_data.get('last_bot_response', '')
        last_intent = context.user_data.get('last_intent', '')
        flight_name = extract_destination(replica)

        if flight_name:
            return flight_name
        if last_response and 'Кстати, у нас есть' in last_response:
            return extract_destination(last_response)
        if last_intent in [Intent.FLIGHT_TYPES.value, Intent.FILTER_FLIGHTS.value, Intent.FLIGHT_RECOMMENDATION.value]:
            for hist in context.user_data.get('history', [])[::-1]:
                hist_flight = extract_destination(hist)
                if hist_flight:
                    return hist_flight
        return context.user_data.get('current_flight')

    def _handle_filter_flights(self, date, price, flight_name, context):
        """Обрабатывает фильтрацию рейсов по дате, цене и маршруту."""
        suitable_flights = []
        for flight, data in CONFIG['flights'].items():
            matches_date = not date or is_date_in_range(date, data['date'])
            matches_price = not price or data['price'] <= price
            matches_route = not flight_name or flight == flight_name or (
                    flight_name and lemmatize_phrase(flight_name) in lemmatize_phrase(
                f"{data['route']['from']} {data['route']['to']}"))
            if matches_date and matches_price and matches_route:
                suitable_flights.append(flight)

        recent_flights = [extract_destination(h) for h in context.user_data.get('history', []) if
                          extract_destination(h)]
        suitable_flights = [t for t in suitable_flights if t not in recent_flights]

        if not suitable_flights:
            conditions = []
            if date:
                conditions.append(f"даты {date}")
            if price:
                conditions.append(f"до {price} рублей")
            if flight_name:
                conditions.append(f"маршрута {flight_name}")
            return f"Извините, нет рейсов для {', '.join(conditions)}."

        flights_list = ', '.join(suitable_flights)
        if len(suitable_flights) == 1:
            flight_name = suitable_flights[0]
            context.user_data['current_flight'] = flight_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Нашёлся рейс {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']} на {CONFIG['flights'][flight_name]['date']}. Хотите узнать цену или описание?"
        return f"Вот что нашлось: {flights_list}."

    def get_answer_by_intent(self, intent, replica, context):
        """Генерирует ответ на основе намерения."""
        flight_name = context.user_data.get('current_flight')
        last_intent = context.user_data.get('last_intent', '')
        date = extract_date(replica)
        price = extract_price(replica)
        new_flight = extract_destination(replica)

        if intent not in CONFIG['intents']:
            return None
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        # Добавляем реакцию на тональность
        sentiment = analyze_sentiment(replica)
        sentiment_suffix = ""
        if sentiment == 'positive':
            sentiment_suffix = " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            sentiment_suffix = " Кажется, вы не в духе. Давайте подберём отличный рейс! 😊"

        if intent in [Intent.FLIGHT_PRICE.value, Intent.FLIGHT_AVAILABILITY.value, Intent.FLIGHT_INFO.value,
                      Intent.ORDER_FLIGHT.value]:
            if not flight_name:
                flight_name = self._find_flight_by_context(replica, context)
                if not flight_name and new_flight:
                    flight_name = new_flight
                if flight_name:
                    context.user_data['current_flight'] = flight_name
                    context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                    return self._get_flight_response(intent, flight_name, replica, context)
                context.user_data['state'] = BotState.WAITING_FOR_FLIGHT.value
                return f"Укажите маршрут, например, 'из Москвы в Питер'.{sentiment_suffix}"
            return self._get_flight_response(intent, flight_name, replica, context)

        elif intent == Intent.FLIGHT_RECOMMENDATION.value:
            if date or new_flight:
                answer = self._handle_filter_flights(date, None, new_flight, context)
            else:
                context.user_data['state'] = BotState.WAITING_FOR_DATE.value
                return f"На какую дату нужен рейс?{sentiment_suffix}"

        elif intent == Intent.FILTER_FLIGHTS.value:
            if date or price or new_flight:
                answer = self._handle_filter_flights(date, price, new_flight, context)
            else:
                return f"Укажите дату, цену или маршрут для фильтрации.{sentiment_suffix}"

        elif intent == Intent.FLIGHT_TYPES.value:
            routes = random.sample(
                [f"{data['route']['from']} - {data['route']['to']}" for flight, data in CONFIG['flights'].items()],
                min(3, len(CONFIG['flights'])))
            flights = random.sample(list(CONFIG['flights'].keys()), min(2, len(CONFIG['flights'])))
            answer = f"У нас есть рейсы {', '.join(set(routes))} и рейсы вроде {', '.join(flights)}. Куда хотите?{sentiment_suffix}"
            context.user_data['current_flight'] = None

        elif intent == Intent.COMPARE_FLIGHTS.value:
            flight1 = random.choice(list(CONFIG['flights'].keys()))
            flight2 = random.choice([t for t in CONFIG['flights'].keys() if t != flight1])
            answer = answer.replace('[flight1]', flight1).replace('[flight2]', flight2)
            context.user_data['current_flight'] = flight1
            answer += f" Что интересует: {flight1} или {flight2}?{sentiment_suffix}"

        elif intent == Intent.YES.value:
            if last_intent == Intent.HELLO.value:
                routes = random.sample(
                    [f"{data['route']['from']} - {data['route']['to']}" for flight, data in CONFIG['flights'].items()],
                    min(3, len(CONFIG['flights'])))
                answer = f"Отлично! У нас есть рейсы {', '.join(set(routes))}. Куда хотите полететь?{sentiment_suffix}"
            elif last_intent in [Intent.FLIGHT_PRICE.value, Intent.FLIGHT_INFO.value, Intent.FLIGHT_AVAILABILITY.value,
                                 Intent.ORDER_FLIGHT.value]:
                if flight_name:
                    answer = f"Цена на рейс {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']} — {CONFIG['flights'][flight_name]['price']} рублей. Что ещё интересует?{sentiment_suffix}"
                else:
                    answer = f"Назови маршрут, чтобы я рассказал подробнее!{sentiment_suffix}"
            elif last_intent == Intent.FLIGHT_TYPES.value:
                flights = random.sample(list(CONFIG['flights'].keys()), min(2, len(CONFIG['flights'])))
                answer = f"У нас есть рейсы {', '.join(flights)}. Назови один, чтобы узнать больше!{sentiment_suffix}"
            elif last_intent == 'offtopic':
                answer = f"Хорошо, давай продолжим! Хочешь узнать про рейсы?{sentiment_suffix}"
            else:
                answer = f"Хорошо, что интересует? Рейсы, цены или что-то ещё?{sentiment_suffix}"

        elif intent == Intent.NO.value:
            context.user_data['current_flight'] = None
            context.user_data['state'] = BotState.NONE.value
            answer = f"Хорошо, какой рейс обсудим теперь?{sentiment_suffix}"

        if intent in [Intent.HELLO.value, Intent.FLIGHT_TYPES.value] and random.random() < 0.2:
            ad_flight = random.choice([t for t in CONFIG['flights'].keys() if t != flight_name])
            answer += f" Кстати, у нас есть рейс {ad_flight} из {CONFIG['flights'][ad_flight]['route']['from']} в {CONFIG['flights'][ad_flight]['route']['to']} — отличный выбор на {CONFIG['flights'][ad_flight]['date']}!{sentiment_suffix}"

        context.user_data['last_intent'] = intent
        return answer

    def generate_answer(self, replica, context):
        """Генерирует ответ на основе диалогов."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized or not self.answers:
            return None
        if not is_meaningful_text(replica):
            return None
        replica_vector = self.tfidf_vectorizer.transform([replica_lemmatized])
        similarities = cosine_similarity(replica_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] > CONFIG['thresholds']['dialogues_similarity']:
            answer = self.answers[best_idx]
            logger.info(
                f"Found in dialogues.txt: replica='{replica_lemmatized}', answer='{answer}', similarity={similarities[best_idx]}")
            # Добавляем реакцию на тональность
            sentiment = analyze_sentiment(replica)
            if sentiment == 'positive':
                answer += " Рад, что ты в хорошем настроении! 😊"
            elif sentiment == 'negative':
                answer += " Кажется, ты не в духе. Может, рейс в отпуск поднимет настроение? 😊"
            if random.random() < 0.3:
                ad_flight = random.choice(list(CONFIG['flights'].keys()))
                answer += f" Кстати, у нас есть рейс {ad_flight} из {CONFIG['flights'][ad_flight]['route']['from']} в {CONFIG['flights'][ad_flight]['route']['to']} — отличный выбор на {CONFIG['flights'][ad_flight]['date']}!"
            context.user_data['last_intent'] = 'offtopic'
            return answer
        logger.info(f"No match in dialogues.txt for replica='{replica_lemmatized}'")
        return None

    def get_failure_phrase(self, replica):
        """Возвращает фразу при неудачном запросе с учетом тональности."""
        flight_name = random.choice(list(CONFIG['flights'].keys()))
        answer = random.choice(CONFIG['failure_phrases']).replace('[flight_name]', flight_name)
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Ты в отличном настроении, давай найдем крутой рейс! 😊"
        elif sentiment == 'negative':
            answer += " Не переживай, давай подберем что-то интересное! 😊"
        return answer

    def _process_none_state(self, replica, context):
        """Обрабатывает состояние NONE."""
        flight_name = extract_destination(replica)
        if flight_name:
            context.user_data['current_flight'] = flight_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Рад, что ты в хорошем настроении! 😊" if sentiment == 'positive' else " Кажется, ты не в духе. Давай найдем что-то крутое? 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду рейс {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']}? Хотите узнать цену, описание или наличие?{suffix}"

        intent = self.classify_intent(replica)
        if intent:
            return self.get_answer_by_intent(intent, replica, context)

        return self.generate_answer(replica, context) or self.get_failure_phrase(replica)

    def _process_waiting_for_flight(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_FLIGHT."""
        flight_name = extract_destination(replica)
        if flight_name:
            context.user_data['current_flight'] = flight_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отличное настроение, да? 😊" if sentiment == 'positive' else " Давай найдем классный рейс! 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду рейс {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']}? Хотите узнать цену, описание или наличие?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " Отлично, давай продолжим! 😊" if sentiment == 'positive' else " Не переживай, уточним! 😊" if sentiment == 'negative' else ""
        return f"Пожалуйста, уточните маршрут, например, 'из Москвы в Питер'.{suffix}"

    def _process_waiting_for_date(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_DATE."""
        date = extract_date(replica)
        if date:
            context.user_data['state'] = BotState.NONE.value
            return self._handle_filter_flights(date, None, None, context)
        sentiment = analyze_sentiment(replica)
        suffix = " В хорошем настроении? 😊" if sentiment == 'positive' else " Не переживай, уточним! 😊" if sentiment == 'negative' else ""
        return f"Укажите дату, например, '15 декабря'.{suffix}"

    def _process_waiting_for_intent(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_INTENT."""
        flight_name = extract_destination(replica) or context.user_data.get('current_flight')
        if flight_name and flight_name in CONFIG['flights']:
            context.user_data['current_flight'] = flight_name
        else:
            flight_name = context.user_data.get('current_flight', None)
            if not flight_name:
                context.user_data['state'] = BotState.WAITING_FOR_FLIGHT.value
                sentiment = analyze_sentiment(replica)
                suffix = " В хорошем настроении? 😊" if sentiment == 'positive' else " Не переживай, уточним! 😊" if sentiment == 'negative' else ""
                return f"Укажите маршрут, например, 'из Москвы в Питер'.{suffix}"

        intent = self.classify_intent(replica)
        if intent in [Intent.FLIGHT_PRICE.value, Intent.FLIGHT_AVAILABILITY.value, Intent.FLIGHT_INFO.value,
                      Intent.ORDER_FLIGHT.value]:
            context.user_data['state'] = BotState.NONE.value
            return self._get_flight_response(intent, flight_name, replica, context)
        if intent == Intent.YES.value:
            context.user_data['state'] = BotState.NONE.value
            sentiment = analyze_sentiment(replica)
            suffix = " Рад твоему настроению! 😊" if sentiment == 'positive' else " Давай поднимем настроение! 😊" if sentiment == 'negative' else ""
            return f"Цена на рейс {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']} — {CONFIG['flights'][flight_name]['price']} рублей. Что ещё интересует?{suffix}"
        if intent == Intent.NO.value:
            context.user_data['current_flight'] = None
            context.user_data['state'] = BotState.NONE.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отлично, продолжаем! 😊" if sentiment == 'positive' else " Не грусти, найдем другое! 😊" if sentiment == 'negative' else ""
            return f"Хорошо, какой рейс обсудим теперь?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " В хорошем настроении? 😊" if sentiment == 'positive' else " Не переживай, найдем что-то классное! 😊" if sentiment == 'negative' else ""
        return f"Что хотите узнать про {flight_name} из {CONFIG['flights'][flight_name]['route']['from']} в {CONFIG['flights'][flight_name]['route']['to']}: цену, описание или наличие?{suffix}"

    def process(self, replica, context):
        """Обрабатывает запрос пользователя."""
        stats = Stats(context)
        if not is_meaningful_text(replica):
            answer = self.get_failure_phrase(replica)
            self._update_context(context, replica, answer)
            stats.add(ResponseType.FAILURE.value, replica, answer, context)
            return answer

        date = extract_date(replica)
        price = extract_price(replica)
        flight_name = extract_destination(replica)
        if date or price or flight_name:
            answer = self._handle_filter_flights(date, price, flight_name, context)
            self._update_context(context, replica, answer, Intent.FILTER_FLIGHTS.value, flight_name)
            stats.add(ResponseType.INTENT.value, replica, answer, context)
            return answer

        state = context.user_data.get('state', BotState.NONE.value)
        logger.info(
            f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}', current_flight='{context.user_data.get('current_flight')}'")

        if state == BotState.WAITING_FOR_FLIGHT.value:
            answer = self._process_waiting_for_flight(replica, context)
        elif state == BotState.WAITING_FOR_DATE.value:
            answer = self._process_waiting_for_date(replica, context)
        elif state == BotState.WAITING_FOR_INTENT.value:
            answer = self._process_waiting_for_intent(replica, context)
        else:
            answer = self._process_none_state(replica, context)

        self._update_context(context, replica, answer, flight_name=flight_name)
        stats.add(ResponseType.INTENT.value if self.classify_intent(
            replica) else ResponseType.GENERATE.value if 'dialogues.txt' in answer else ResponseType.FAILURE.value,
                  replica, answer, context)
        return answer


# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        import signal
        def signal_handler(signum, frame):
            raise TimeoutError("Speech recognition timed out")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)  # Таймаут 5 секунд
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except (sr.UnknownValueError, sr.RequestError, TimeoutError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}\n{traceback.format_exc()}")
        return None
    finally:
        signal.alarm(0)
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')


# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}\n{traceback.format_exc()}")
        return None


# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = Intent.HELLO.value
    await update.message.reply_text(answer)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get('stats', {ResponseType.INTENT.value: 0, ResponseType.GENERATE.value: 0,
                                            ResponseType.FAILURE.value: 0})
    answer = (
        f"Статистика:\n"
        f"Обработано намерений: {stats[ResponseType.INTENT.value]}\n"
        f"Ответов из диалогов: {stats[ResponseType.GENERATE.value]}\n"
        f"Неудачных запросов: {stats[ResponseType.FAILURE.value]}"
    )
    await update.message.reply_text(answer)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    bot = context.bot_data.setdefault('bot', Bot())
    answer = bot.process(user_text, context)
    await update.message.reply_text(answer)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    bot = context.bot_data.setdefault('bot', Bot())
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot.process(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}\n{traceback.format_exc()}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')


def run_bot():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()


if __name__ == '__main__':
    run_bot()
