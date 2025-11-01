import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. DAS TRAINIERTE MODELL LADEN ---
try:
    model = tf.keras.models.load_model('spam_classifier_model.keras')
    print("Modell 'spam_classifier_model.keras' erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit()

# --- 2. DEN TOKENIZER REPRODUZIEREN (WICHTIG!) ---
# Wir müssen den exakt gleichen Tokenizer wie im Training verwenden.
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"
df = pd.read_csv(filepath, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
labels = df['label'].values
messages = df['message'].values
msg_train, _, _, _ = train_test_split(messages, labels, test_size=0.2, random_state=42, stratify=labels)

# Tokenizer mit den gleichen Parametern wie in Schritt 3 initialisieren
vocab_size = 5000
max_length = 100
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(msg_train) # Nur auf den Trainingsdaten fitten!
print("Tokenizer wurde erfolgreich reproduziert.")

# --- 3. FINALE EVALUIERUNG AUF DEM TEST-SET ---
with np.load('processed_data.npz') as data:
    padded_test = data['test_data']
    label_test = data['test_labels']

loss, accuracy = model.evaluate(padded_test, label_test)
print("\nFinale Evaluierung des Modells auf den Testdaten:")
print(f"Verlust (Loss): {loss:.4f}")
print(f"Genauigkeit (Accuracy): {accuracy:.4f}")

# --- 4. VORHERSAGE-FUNKTION ---
def predict_messages(messages_to_predict):
    """Nimmt eine Liste von Texten und gibt Vorhersagen zurück."""
    # Texte in Sequenzen umwandeln und padden
    sequences = tokenizer.texts_to_sequences(messages_to_predict)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Vorhersage treffen
    predictions = model.predict(padded)
    
    print("\n--- Vorhersage-Ergebnisse ---")
    for i, text in enumerate(messages_to_predict):
        score = predictions[i][0]
        label = "Spam" if score > 0.5 else "Ham"
        print(f"Nachricht: '{text}'")
        print(f"  -> Vorhersage: {label} (Score: {score:.4f})")
        
# --- 5. EIGENE NACHRICHTEN TESTEN ---
my_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to [http://bit.ly/spam-link](http://bit.ly/spam-link) to claim now.",
    "Hey mom, I'm going to be late for dinner tonight. See you in a bit.",
    "URGENT: Your account has been suspended. Please verify your details immediately to avoid closure.",
    "Can you please call me back when you have a moment?"
]

predict_messages(my_messages)
