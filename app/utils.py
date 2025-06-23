import logging
import nltk
from rapidfuzz import process, fuzz
from data.config import CONFIG
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


# Загрузка тонального словаря
def load_tonal_dict():
    tonal_dict = {}
    try:
        with open('data/tonal_dict.txt', encoding='utf-8') as f:
            for line in f:
                word, score = line.strip().split('\t')
                tonal_dict[word] = float(score)
    except FileNotFoundError:
        logger.error("Файл tonal_dict.txt не найден")
    return tonal_dict


TONAL_DICT = load_tonal_dict()


# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = '1234567890qwertyuiopasdfghjklzxcvbnmабвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()


# Лемматизация и морфологический анализ
def lemmatize_phrase(phrase):
    if not phrase:
        return ""
    cleaned_phrase = clear_phrase(phrase)
    if not cleaned_phrase:
        return ""
    doc = Doc(cleaned_phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmatized_words = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma if token.lemma else token.text
        lemmatized_words.append(lemma)
    return ' '.join(lemmatized_words)


# Анализ тональности
def analyze_sentiment(phrase):
    if not phrase:
        return 'neutral'
    lemmatized = lemmatize_phrase(phrase)
    words = lemmatized.split()
    sentiment_score = 0
    count = 0
    for word in words:
        if word in TONAL_DICT:
            sentiment_score += TONAL_DICT[word]
            count += 1
    if count == 0:
        return 'neutral'
    avg_score = sentiment_score / count
    if avg_score > 0.3:
        return 'positive'
    elif avg_score < -0.3:
        return 'negative'
    return 'neutral'


# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)


# Извлечение даты
def extract_date(replica):
    replica = lemmatize_phrase(replica)
    logger.info(f"Extracting date from: '{replica}'")
    words = replica.split()
    months = {'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04', 'май': '05', 'июнь': '06',
              'июль': '07', 'август': '08', 'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12'}
    for i, word in enumerate(words):
        if word.isdigit() and i + 1 < len(words) and words[i + 1] in months:
            day = word.zfill(2)
            month = months[words[i + 1]]
            return f"{day}.{month}"
    logger.info("Date not found")
    return None


# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
    logger.info(f"Extracting price from: '{replica}'")
    if not replica:
        return None
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (
                i + 1 < len(words) and words[i + 1] in ['рублей', 'руб'] or 'до' in words[:i] or 'дешевле' in words[
                                                                                                              :i]):
            logger.info(f"Found price: {word}")
            return int(word)
    logger.info("Price not found")
    return None


# Извлечение маршрута
def extract_destination(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    # Проверка полного совпадения с названием рейса или синонимами
    for flight in CONFIG['flights'].keys():
        flight_lemmatized = lemmatize_phrase(flight)
        if flight_lemmatized in replica:
            return flight
        synonyms_lemmatized = [lemmatize_phrase(syn) for syn in CONFIG['flights'][flight].get('synonyms', [])]
        if any(syn in replica for syn in synonyms_lemmatized):
            return flight
        candidates = [flight] + CONFIG['flights'][flight].get('synonyms', []) + [
            f"{data['route']['from']} {data['route']['to']}" for flight, data in CONFIG['flights'].items()]
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > CONFIG['thresholds']['fuzzy_match_flight']:
            for f in CONFIG['flights']:
                if best_match[0] in [f] + CONFIG['flights'][f].get('synonyms', []) or best_match[
                    0] == f"{CONFIG['flights'][f]['route']['from']} {CONFIG['flights'][f]['route']['to']}":
                    return f
    # Попытка извлечь маршрут из текста (например, "из Москвы в Питер")
    words = replica.split()
    for i, word in enumerate(words):
        if word == 'из' and i + 1 < len(words):
            from_city = words[i + 1]
            to_city = None
            if i + 3 < len(words) and words[i + 2] == 'в':
                to_city = words[i + 3]
            elif i + 2 < len(words) and words[i + 2] in ['в', 'на']:
                to_city = words[i + 2]
            if to_city:
                for flight, data in CONFIG['flights'].items():
                    route = data.get('route', {})
                    from_match = lemmatize_phrase(from_city) in lemmatize_phrase(route.get('from', ''))
                    to_match = lemmatize_phrase(to_city) in lemmatize_phrase(route.get('to', ''))
                    if from_match and to_match:
                        return flight
    # Попытка извлечь маршрут без предлогов (например, "Москва Париж")
    if len(words) >= 2:
        from_city = words[0]
        to_city = words[1]
        for flight, data in CONFIG['flights'].items():
            route = data.get('route', {})
            from_match = lemmatize_phrase(from_city) in lemmatize_phrase(route.get('from', ''))
            to_match = lemmatize_phrase(to_city) in lemmatize_phrase(route.get('to', ''))
            if from_match and to_match:
                return flight
    # Попытка извлечь только город назначения
    for flight, data in CONFIG['flights'].items():
        to_city = lemmatize_phrase(data['route']['to'])
        if to_city in replica:
            return flight
    return None


# Извлечение авиакомпании
def extract_airline(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    for flight, data in CONFIG['flights'].items():
        for airline in data.get('airlines', []):
            airline_lemmatized = lemmatize_phrase(airline)
            airline_synonyms = data.get('airline_synonyms', {}).get(airline, [])
            synonyms_lemmatized = [lemmatize_phrase(syn) for syn in airline_synonyms]
            if airline_lemmatized in replica or any(syn in replica for syn in synonyms_lemmatized):
                return airline
    return None


# Проверка даты в диапазоне
def is_date_in_range(date, flight_date):
    try:
        input_date = datetime.strptime(date, '%d.%m')
        flight_date = datetime.strptime(flight_date, '%d.%m')
        return input_date.date() == flight_date.date()
    except (ValueError, TypeError):
        return False


# Класс для управления статистикой
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")
