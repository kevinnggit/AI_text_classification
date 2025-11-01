Bauanleitung: Dein eigener KI-Spam-Filter von Grund auf

Dies ist eine vollständige Schritt-für-Schritt-Anleitung, um ein neuronales Netz zu bauen und zu trainieren, das lernt, Spam-Nachrichten von normalen Nachrichten (Ham) zu unterscheiden. Wir verwenden dafür ausschließlich Python und gängige Open-Source-Bibliotheken, ohne auf externe APIs wie OpenAI oder Google AI angewiesen zu sein.

Schritt 1: Die Umgebung einrichten

Das Fundament jedes Projekts ist eine saubere Entwicklungsumgebung.

Voraussetzungen:

Ein Linux-basiertes System (z.B. Ubuntu oder WSL unter Windows).

Python Version 3.11. Neuere Versionen sind oft noch nicht mit TensorFlow kompatibel.

1. Python 3.11 installieren (Beispiel für Ubuntu/WSL):

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv


2. Projektordner und virtuelle Umgebung erstellen:
Eine virtuelle Umgebung (.venv) isoliert die für dieses Projekt benötigten Bibliotheken von anderen Projekten.

# Ordner erstellen und dorthin wechseln
mkdir -p ~/ai-spam-filter && cd ~/ai-spam-filter

# Virtuelle Umgebung mit Python 3.11 erstellen
python3.11 -m venv .venv

# Umgebung aktivieren
source .venv/bin/activate

# Nach der Aktivierung sollte dein Terminal (.venv) anzeigen


3. Notwendige Bibliotheken installieren:

# Pip, das Paket-Tool von Python, aktualisieren
python -m pip install --upgrade pip

# Die vier Kern-Bibliotheken installieren
pip install tensorflow pandas numpy scikit-learn


Schritt 2: Daten beschaffen und analysieren

Jede KI braucht Daten zum Lernen. Wir verwenden einen öffentlichen Datensatz mit SMS-Nachrichten.

Code (daten_laden.py):
Dieser Code lädt die Daten mit der Pandas-Bibliothek und gibt uns einen ersten Überblick.

import pandas as pd

# Pfad zur lokal gespeicherten CSV-Datei
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"

try:
    # Lade die Daten, gib den Spalten sinnvolle Namen
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("--- Erste 5 Zeilen der Daten ---")
    print(df.head())
    print("\n--- Verteilung der Nachrichten ---")
    print(df['label'].value_counts())

except FileNotFoundError:
    print(f"FEHLER: Die Datei unter '{filepath}' wurde nicht gefunden.")


Schritt 3: Text in Zahlen umwandeln (Vorverarbeitung)

Ein Computer versteht keine Wörter, nur Zahlen. Dieser Schritt ist entscheidend, um den Text für das neuronale Netz aufzubereiten.

Konzepte:

Label-Kodierung: Wir wandeln 'ham' in 0 und 'spam' in 1 um.

Tokenization: Wir zerlegen Sätze in Wörter (Tokens).

Sequenzierung: Jedes einzigartige Wort erhält eine feste Zahl.

Padding: Wir füllen alle Sätze mit Nullen auf, damit sie die gleiche Länge haben.

Code (daten_verarbeiten.py):

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Daten laden und Labels umwandeln
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"
df = pd.read_csv(filepath, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Daten in Trainings- und Test-Sets aufteilen (80% / 20%)
messages = df['message'].values
labels = df['label'].values
msg_train, msg_test, label_train, label_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. Tokenizer erstellen und auf die Trainingsdaten anwenden
vocab_size = 5000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(msg_train)

# 4. Texte in Zahlensequenzen umwandeln und padden
sequences_train = tokenizer.texts_to_sequences(msg_train)
padded_train = pad_sequences(sequences_train, maxlen=max_length, padding='post', truncating='post')

sequences_test = tokenizer.texts_to_sequences(msg_test)
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding='post', truncating='post')

# 5. Verarbeitete Daten für den nächsten Schritt speichern
np.savez('processed_data.npz', 
         train_data=padded_train, 
         train_labels=label_train, 
         test_data=padded_test, 
         test_labels=label_test)

print("Vorverarbeitung abgeschlossen. Daten in 'processed_data.npz' gespeichert.")


Schritt 4: Das neuronale Netz bauen und trainieren

Jetzt bauen wir das "Gehirn" unserer KI. [Bild einer einfachen neuronalen Netzarchitektur]

Architektur:

Embedding-Schicht: Lernt, Wörtern mit ähnlicher Bedeutung ähnliche Zahlenvektoren zuzuordnen.

GlobalAveragePooling1D-Schicht: Fasst den Vektor einer ganzen Nachricht zu einem einzigen Vektor zusammen.

Dense-Schicht (versteckt): Eine Schicht von Neuronen, die Muster lernt.

Dense-Schicht (Output): Ein einzelnes Neuron, das eine Wahrscheinlichkeit (0 bis 1) ausgibt, ob es sich um Spam handelt.

Code (modell_trainieren.py):

import tensorflow as tf
import numpy as np

# 1. Verarbeitete Daten laden
with np.load('processed_data.npz') as data:
    padded_train = data['train_data']
    label_train = data['train_labels']
    padded_test = data['test_data']
    label_test = data['test_labels']

# 2. Modell-Architektur definieren
vocab_size = 5000
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Modell kompilieren
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 4. Modell trainieren
num_epochs = 10
history = model.fit(
    padded_train, 
    label_train, 
    epochs=num_epochs, 
    validation_data=(padded_test, label_test)
)

# 5. Trainiertes Modell speichern
model.save('spam_classifier_model.keras')
print("\nTraining abgeschlossen! Modell gespeichert.")


Schritt 5: Das fertige Modell anwenden

Der letzte Schritt: Wir laden unser trainiertes Modell und testen es mit eigenen Sätzen.

Code (modell_anwenden.py):

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Trainiertes Modell laden
model = tf.keras.models.load_model('spam_classifier_model.keras')

# 2. Den Tokenizer exakt wie im Training reproduzieren
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"
df = pd.read_csv(filepath, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
messages = df['message'].values
labels = df['label'].values
msg_train, _, _, _ = train_test_split(messages, labels, test_size=0.2, random_state=42, stratify=labels)

vocab_size = 5000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(msg_train)

# 3. Vorhersage-Funktion
def predict_messages(messages_to_predict):
    sequences = tokenizer.texts_to_sequences(messages_to_predict)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    predictions = model.predict(padded)
    
    print("\n--- Vorhersage-Ergebnisse ---")
    for i, text in enumerate(messages_to_predict):
        score = predictions[i][0]
        label = "Spam" if score > 0.5 else "Ham"
        print(f"Nachricht: '{text}'\n  -> Vorhersage: {label} (Score: {score:.4f})")
        
# 4. Eigene Nachrichten testen
my_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to [http://bit.ly/spam-link](http://bit.ly/spam-link) to claim now.",
    "Hey mom, I'm going to be late for dinner tonight. See you in a bit.",
    "URGENT: Your account has been suspended. Please verify your details immediately.",
    "Can you please call me back when you have a moment?"
]

predict_messages(my_messages)


Schritt 6: Wie kann das Projekt perfektioniert werden?

Du hast ein solides Grundmodell gebaut. Hier sind einige professionelle nächste Schritte, um es zu verbessern und zu erweitern:

Tokenizer speichern und laden:

Problem: In Schritt 5 müssen wir den Tokenizer jedes Mal neu erstellen. Das ist ineffizient und fehleranfällig.

Lösung: Speichere den trainierten Tokenizer nach Schritt 3 mit der pickle-Bibliothek von Python. In Schritt 5 kannst du ihn dann einfach laden, anstatt die Daten erneut verarbeiten zu müssen.

# Nach tokenizer.fit_on_texts in Schritt 3 speichern:
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# In Schritt 5 laden:
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


Modellarchitektur verbessern:

Dropout: Füge tf.keras.layers.Dropout(0.5) zwischen den Dense-Schichten hinzu. Ein Dropout-Layer deaktiviert während des Trainings zufällig einige Neuronen. Das zwingt das Netzwerk, robuster zu werden und verhindert, dass es sich zu sehr auf einzelne Merkmale verlässt ("Overfitting").

LSTM / GRU Layer: Ersetze die GlobalAveragePooling1D-Schicht durch eine tf.keras.layers.LSTM(64) oder tf.keras.layers.GRU(64). Diese Schichten (sogenannte rekurrente neuronale Netze) sind speziell dafür gemacht, sequenzielle Daten wie Text zu verarbeiten und die Reihenfolge der Wörter besser zu berücksichtigen. Dies kann die Genauigkeit oft erheblich verbessern.

Hyperparameter-Tuning:

Experimentiere mit den Werten, die wir festgelegt haben: vocab_size, embedding_dim, max_length, die Anzahl der Neuronen in den Dense-Schichten und die num_epochs. Jede Änderung kann die Leistung des Modells beeinflussen.

Eine interaktive Anwendung erstellen:

Verwandle das Skript aus Schritt 5 in ein einfaches Kommandozeilen-Tool. Verwende die input()-Funktion von Python, um einen Benutzer nach einer Nachricht zu fragen, und gib die Vorhersage direkt aus.

# Am Ende von Schritt 5 hinzufügen:
while True:
    user_input = input("\nGib eine Nachricht ein (oder 'exit' zum Beenden): ")
    if user_input.lower() == 'exit':
        break
    predict_messages([user_input])

