import pandas as pd

# URL des Datensatzes
filepath = "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?select=spam.csv"

# Lade die Daten mit Pandas. Wir geben an, dass die Spalten durch ein Komma getrennt sind.
# Wir geben den Spalten auch direkt die Namen 'label' und 'message'.
try:
    df = pd.read_csv(filepath, encoding='latin-1')
    # Behalte nur die relevanten Spalten und benenne sie um
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("Daten erfolgreich geladen!")
    print("---------------------------------")

    # Zeige die ersten 5 Zeilen an, um einen Eindruck zu bekommen
    print("Die ersten 5 Nachrichten im Datensatz:")
    print(df.head())
    print("\n---------------------------------")

    # Zeige eine Zusammenfassung der Daten an
    print("Informationen über den Datensatz:")
    df.info()
    print("\n---------------------------------")

    # Zähle, wie viele 'spam' und 'ham' Nachrichten wir haben
    print("Verteilung der Nachrichten:")
    print(df['label'].value_counts())

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
    print("Stelle sicher, dass du eine aktive Internetverbindung hast.")

