# ./app/conversation_model_trainer.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import sanitize_text, perform_lemmatization, log_handler

log_handler.log("Инициализация процесса обучения модели диалогов")

conversation_pairs = []
try:
    with open('data/dialogues.txt', 'r', encoding='utf-8') as file:
        raw_data = file.read()
    
    raw_dialogues = [dialogue.split('\n')[:2] 
                    for dialogue in raw_data.split('\n\n') 
                    if len(dialogue.split('\n')) >= 2]
    
    conversation_pairs = [
        (question[1:].strip() if question.startswith('-') else question.strip(),
         response[1:].strip() if response.startswith('-') else response.strip())
        for question, response in raw_dialogues
    ]
    
except IOError as error:
    log_handler.error(f"Не удалось обработать файл диалогов: {error}")
    raise SystemExit(1)

processed_questions = [perform_lemmatization(sanitize_text(question)) 
                      for question, _ in conversation_pairs]
bot_responses = [response for _, response in conversation_pairs]

# Конфигурация и обучение векторайзера
text_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    lowercase=True,
    min_df=0.001
)
question_vectors = text_vectorizer.fit_transform(processed_questions)

# Экспорт обученных моделей
model_files = {
    'vectorizer': 'models/dialogues_vectorizer.pkl',
    'matrix': 'models/dialogues_matrix.pkl',
    'responses': 'models/dialogues_answers.pkl'
}

try:
    with open(model_files['vectorizer'], 'wb') as vec_file:
        pickle.dump(text_vectorizer, vec_file)
    
    with open(model_files['matrix'], 'wb') as mat_file:
        pickle.dump(question_vectors, mat_file)
    
    with open(model_files['responses'], 'wb') as resp_file:
        pickle.dump(bot_responses, resp_file)
        
except IOError as save_error:
    log_handler.error(f"Ошибка сохранения моделей: {save_error}")
    raise SystemExit(1)

log_handler.log("Обучение завершено. Модели сохранены в директории ./models/")