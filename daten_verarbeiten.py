import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- 1. DATEN LADEN (wie in Schritt 2) ---
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"
try:
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print("Daten erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
    exit()

# --- 2. LABELS IN ZAHLEN UMWANDELN (0 und 1) ---
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
labels = df['label'].values
messages = df['message'].values
print("Labels wurden zu 0 (ham) und 1 (spam) umgewandelt.")

# --- 3. DATEN IN TRAININGS- UND TEST-SETS AUFTEILEN ---
# Wir halten 20% der Daten für den finalen Test zurück.
msg_train, msg_test, label_train, label_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)
# stratify=labels sorgt dafür, dass das Verhältnis von Spam zu Ham in beiden Sets gleich ist.

print(f"Daten aufgeteilt: {len(msg_train)} Trainingsnachrichten, {len(msg_test)} Testnachrichten.")

# --- 4. TOKENIZER ERSTELLEN UND ANWENDEN ---
# Hyperparameter für die Vorverarbeitung
vocab_size = 5000  # Wir betrachten nur die 5000 häufigsten Wörter
max_length = 100   # Jede Nachricht wird auf 100 Wörter gekürzt/aufgefüllt
oov_tok = "<OOV>"  # Ein spezieller Token für Wörter, die nicht im Vokabular sind

# Initialisiere den Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Baue das Wörterbuch nur auf den Trainingsdaten auf!
tokenizer.fit_on_texts(msg_train)

# Wandle die Trainings- und Testnachrichten in Zahlensequenzen um
sequences_train = tokenizer.texts_to_sequences(msg_train)
sequences_test = tokenizer.texts_to_sequences(msg_test)

# --- 5. SEQUENZEN AUFFÜLLEN (PADDING) ---
padded_train = pad_sequences(sequences_train, maxlen=max_length, padding='post', truncating='post')
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding='post', truncating='post')

print("\nVorverarbeitung abgeschlossen!")
print("Beispiel einer Original-Trainingsnachricht:")
print(msg_train[0])
print("\nDieselbe Nachricht als aufgefüllte Zahlensequenz:")
print(padded_train[0])

# Speichere die verarbeiteten Daten für den nächsten Schritt
# WICHTIG: np.savez() verwenden, um mehrere Arrays zu speichern.
np.savez('processed_data.npz', train_data=padded_train, train_labels=label_train, test_data=padded_test, test_labels=label_test)
print("\nVerarbeitete Daten wurden in 'processed_data.npz' gespeichert.")

