# ใช้ newenv
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import requests

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Neo4j database connection details
URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

# Function to run Neo4j queries
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

# Load greeting and question corpus from Neo4j
def load_corpora():
    greeting_query = '''
    MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
    '''
    question_query = '''
    MATCH (q:Question) RETURN q.name as name, q.msg_reply as reply;
    '''
    
    greetings = run_query(greeting_query)
    questions = run_query(question_query)
    
    greeting_corpus = [record['name'] for record in greetings]
    question_corpus = [record['name'] for record in questions]
    
    greeting_dict = {record['name']: record['reply'] for record in greetings}
    question_dict = {record['name']: record['reply'] for record in questions}
    
    return greeting_corpus, question_corpus, greeting_dict, question_dict

greeting_corpus, question_corpus, greeting_dict, question_dict = load_corpora()

# Function to compute response and indicate the source
def compute_response(sentence):
    # Encode the question corpus and the input sentence
    question_vec = model.encode(question_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    
    # Compute cosine similarity
    question_scores = util.cos_sim(question_vec, ask_vec).cpu().numpy()
    
    # Get the index of the most similar question
    max_question_index = np.argmax(question_scores)
    max_question_score = question_scores[max_question_index]

    if max_question_score > 0.8:
        Match_question = question_corpus[max_question_index]
        response = question_dict.get(Match_question, "ไม่พบคำตอบในระบบ")
        return f"คำถาม: {sentence}\nคำตอบ (จาก Neo4j): {response}"
    else:
        # Handle greeting
        greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
        greeting_scores = util.cos_sim(greeting_vec, ask_vec).cpu().numpy()
        
        max_greeting_index = np.argmax(greeting_scores)
        max_greeting_score = greeting_scores[max_greeting_index]

        if max_greeting_score > 0.8:
            Match_greeting = greeting_corpus[max_greeting_index]
            response = greeting_dict.get(Match_greeting, "ไม่พบคำตอบในระบบ")
            return f"คำถาม: {sentence}\nคำตอบ (จาก Neo4j): {response}"
        else:
            # Fallback to Ollama API if no response found
            llama_response = get_ollama_response(sentence)
            return f"คำถาม: {sentence}\nคำตอบ (จาก OLLaMA): {llama_response}"

# Flask app to handle Line bot messages remains unchanged


# Function to get response from Ollama API
def get_ollama_response(prompt):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",
        "prompt": prompt + "คำตอบไม่เกิน 20 คำ เรื่องกฎหมายครอบครัวเท่านั้น",
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.text
        data = json.loads(response_data)
        return data.get("response", "ไม่สามารถตอบคำถามได้")
    else:
        return "ไม่สามารถติดต่อโมเดลได้"

# Flask app to handle Line bot messages
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'A0aobhKEnLStv6/dpp/xmcDjLdNTKiMHAV7hHrfESiRpTGcQmlT3258Jphc4CK8qvNRhs3eYTJ7k4tPEruCPy/jxRvDwq80UmRnfyA/LzZkR+eAUMCUelxrOCuCSw+VHU6QyIfn7zbx5oqqh/9ahjgdB04t89/1O/w1cDnyilFU='
        secret = '0230e7a07ae588d4db5a2e206bbec687'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        response_msg = compute_response(msg)
        
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
    except Exception as e:
        print(f"Error: {e}")
        print(body)
    return 'OK'

if __name__ == '__main__':
    # For Debugging
    compute_response("กฎหมายครอบครัวคืออะไร?")
    app.run(port=5000)
