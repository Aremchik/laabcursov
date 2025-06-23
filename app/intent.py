import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from data.settings import SETTINGS
from tools import clean_text, get_lemmas, log

log.info("Инициализация процесса тренировки классификатора намерений")

# Сбор и подготовка обучающих данных
samples = []
labels = []
for intent_name, intent_data in SETTINGS['intents'].items():
    for phrase in intent_data['examples']:
        processed_phrase = get_lemmas(clean_text(phrase))
        samples.append(processed_phrase)
        labels.append(intent_name)

# Преобразование текста в векторы
text_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    lowercase=True
)
vectorized_samples = text_vectorizer.fit_transform(samples)

# Тренировка классификатора
intent_classifier = LinearSVC()
intent_classifier.fit(vectorized_samples, labels)

# Проверка и создание папки для моделей
if not os.path.exists('model_files'):
    os.mkdir('model_files')

# Экспорт обученных моделей
model_file = 'model_files/intent_classifier.pkl'
vec_file = 'model_files/text_vectorizer.pkl'

with open(model_file, 'wb') as model_out:
    pickle.dump(intent_classifier, model_out)
    
with open(vec_file, 'wb') as vec_out:
    pickle.dump(text_vectorizer, vec_out)

log.info(f"Обучение завершено. Модели сохранены в {model_file} и {vec_file}")