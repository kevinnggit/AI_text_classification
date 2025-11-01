import pandas as pd

# Lokaler Dateipfad zu deinem Datensatz.
# Ändere diesen Pfad, falls deine Datei woanders liegt.
filepath = "/home/kevin/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"

# Lade die Daten mit Pandas.
try:
    df = pd.read_csv(filepath, encoding='latin-1')
    # Behalte nur die relevanten Spalten und benenne sie um
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("Daten erfolgreich von lokalem Pfad geladen!")
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

except FileNotFoundError:
    print(f"FEHLER: Die Datei wurde unter dem Pfad '{filepath}' nicht gefunden.")
    print("Bitte überprüfe, ob der Pfad korrekt ist und du die nötigen Leserechte hast.")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
