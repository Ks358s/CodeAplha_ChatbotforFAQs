import re
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# --- STEP 1: Collect FAQs ---
FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day hassle-free return policy. Items must be returned in their original packaging and unused condition."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard domestic shipping takes between 3 to 5 business days. International delivery can take anywhere from 7 to 14 business days."
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order is shipped, you will receive an email containing a unique tracking link. You can also view it in your Account Profile."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit/debit cards (Visa, MasterCard, American Express), UPI, Net Banking, and digital wallets like PayPal."
    },
    {
        "question": "Can I change or cancel my order after placing it?",
        "answer": "Orders can be modified or canceled within 1 hour of placement by reaching out directly to our live support helpdesk."
    }
]

# Extract questions list for vector matching
faq_questions = [item["question"] for item in FAQ_DATA]

# --- STEP 2: Preprocess the Text ---
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "you", "your", "i", "me",
    "my", "we", "us", "our", "this", "these", "those", "what", "which",
    "who", "whom", "do", "does", "did", "but", "if", "or", "not"
}

def load_stop_words():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return DEFAULT_STOPWORDS

stop_words = load_stop_words()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

cleaned_faq_questions = [clean_text(q) for q in faq_questions]

# --- STEP 3: Match User Questions & Handle API ---
@app.route('/chat', methods=['POST'])
def chat():
    request_data = request.get_json(silent=True) or {}
    user_message = str(request_data.get("message", "")).strip()
    
    if not user_message:
        return jsonify({"reply": "I didn't catch that. Could you please ask something?"})
    
    cleaned_user_query = clean_text(user_message)
    
    temporary_corpus = cleaned_faq_questions + [cleaned_user_query]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(temporary_corpus)
    
    faq_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    
    similarity_scores = cosine_similarity(user_vector, faq_vectors).flatten()
    
    best_match_idx = similarity_scores.argsort()[-1]
    highest_score = similarity_scores[best_match_idx]
    
    SIMILARITY_THRESHOLD = 0.25
    if highest_score >= SIMILARITY_THRESHOLD:
        reply = FAQ_DATA[best_match_idx]["answer"]
    else:
        reply = "I'm sorry, I couldn't find an exact answer to that question in our system. Would you like to speak to a customer care executive?"
        
    return jsonify({"reply": reply})

# --- STEP 4: Render Front UI Layer ---
@app.route('/')
def home():
    return render_template_string(HTML_UI)

# --- STEP 5: Web UI HTML with Chat Interface ---
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQ Helpdesk Bot</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-100 font-sans h-screen flex flex-col justify-between">

    <div class="container mx-auto max-w-lg my-auto p-4 flex flex-col h-[85vh] bg-white rounded-2xl shadow-2xl border border-slate-200">
        
        <div class="flex items-center gap-3 pb-4 border-b border-slate-100">
            <div class="w-10 h-10 bg-indigo-600 rounded-full flex items-center justify-center text-white text-lg shadow-md shadow-indigo-100">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div>
                <h2 class="font-bold text-slate-800 text-base">Support Assistant</h2>
                <span class="text-xs text-emerald-500 font-medium flex items-center gap-1">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 inline-block animate-pulse"></span> Online & Ready
                </span>
            </div>
        </div>

        <div id="chatLog" class="flex-grow overflow-y-auto py-4 space-y-4 pr-1">
            <div class="flex items-start gap-2.5 max-w-[85%]">
                <div class="bg-slate-100 text-slate-700 text-sm rounded-2xl rounded-tl-none p-3.5 shadow-sm">
                    Hello! 👋 I'm your FAQ assistant. Ask me anything about returns, shipping, or payments!
                </div>
            </div>
        </div>

        <div class="border-t border-slate-100 pt-3 flex gap-2">
            <input type="text" id="userInput" placeholder="Ask a question..." 
                class="flex-grow bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-700">
            <button id="sendBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white p-3.5 rounded-xl transition-colors shadow-md shadow-indigo-100 active:scale-95 flex items-center justify-center">
                <i class="fa-solid fa-paper-plane text-sm"></i>
            </button>
        </div>
    </div>

    <script>
        const chatLog = document.getElementById('chatLog');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function appendMessage(text, isUser = false) {
            const wrap = document.createElement('div');
            wrap.className = isUser ? "flex items-start gap-2.5 max-w-[85%] ml-auto justify-end" : "flex items-start gap-2.5 max-w-[85%]";
            
            const box = document.createElement('div');
            box.className = isUser 
                ? "bg-indigo-600 text-white text-sm rounded-2xl rounded-tr-none p-3.5 shadow-md shadow-indigo-100" 
                : "bg-slate-100 text-slate-700 text-sm rounded-2xl rounded-tl-none p-3.5 shadow-sm";
            box.textContent = text;
            
            wrap.appendChild(box);
            chatLog.appendChild(wrap);
            chatLog.scrollTop = chatLog.scrollHeight;
        }

        async function handleSendMessage() {
            const messageText = userInput.value.trim();
            if (!messageText) return;

            appendMessage(messageText, true);
            userInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: messageText })
                });
                const data = await response.json();
                
                appendMessage(data.reply, false);
            } catch (err) {
                appendMessage("Oops, I lost connection to the server. Please try again.", false);
            }
        }

        sendBtn.addEventListener('click', handleSendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSendMessage();
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
