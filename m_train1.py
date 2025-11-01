import tensorflow as tf
import numpy as np

# --- 1. VERARBEITETE DATEN LADEN ---
try:
    with np.load('processed_data.npz') as data:
        padded_train = data['train_data']
        label_train = data['train_labels']
        padded_test = data['test_data']
        label_test = data['test_labels']
    print("Verarbeitete Daten erfolgreich geladen.")
except FileNotFoundError:
    print("FEHLER: 'processed_data.npz' nicht gefunden. Bitte führe zuerst das Skript aus Schritt 3 aus.")
    exit()

# --- 2. MODELL-ARCHITEKTUR DEFINIEREN ---
# Hyperparameter aus Schritt 3 wiederverwenden
vocab_size = 5000
max_length = 100
embedding_dim = 16 # Dimension für die Wort-Vektoren

model = tf.keras.Sequential([
    # 1. Embedding-Schicht: Wandelt Wort-Indizes in dichte Vektoren um
    # Das Argument `input_length` ist nicht mehr notwendig und wird automatisch abgeleitet.
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    
    # 2. Pooling-Schicht: Fasst die Vektoren für die ganze Sequenz zusammen
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # 3. Eine dichte Schicht zum Musterlernen
    tf.keras.layers.Dense(24, activation='relu'),
    
    # 4. Output-Schicht: Gibt eine Wahrscheinlichkeit zwischen 0 und 1 aus
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- 3. MODELL KOMPILIEREN ---
# Hier legen wir fest, WIE das Modell lernen soll
model.compile(
    loss='binary_crossentropy', # Verlustfunktion für Ja/Nein-Entscheidungen
    optimizer='adam',           # Ein effizienter Algorithmus zur Anpassung der Gewichte
    metrics=['accuracy']        # Wir wollen die Genauigkeit während des Trainings beobachten
)

# Zeige eine Zusammenfassung der Architektur an
model.summary()

# --- 4. MODELL TRAINIEREN ---
num_epochs = 10 # Wie oft das Modell die gesamten Trainingsdaten sehen soll
history = model.fit(
    padded_train, 
    label_train, 
    epochs=num_epochs, 
    validation_data=(padded_test, label_test), # Daten zur Validierung nach jeder Epoche
    verbose=2
)

# --- 5. TRAINIERTES MODELL SPEICHERN ---
model.save('spam_classifier_model1_1.keras')
print("\nTraining abgeschlossen! Das Modell wurde als 'spam_classifier_model.keras' gespeichert.")

