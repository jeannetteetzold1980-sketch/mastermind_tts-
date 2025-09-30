# TTS-Datensatz-Formatter

Dieses Tool ist eine grafische Benutzeroberfläche (GUI) zur Verarbeitung von Audiodateien, um hochwertige Datensätze für das Training von Text-to-Speech (TTS)-Modellen zu erstellen. Die Pipeline ist darauf ausgelegt, Audiodateien intelligent zu segmentieren, zu filtern, zu bereinigen und zu transkribieren.

## Funktionen

- **Grafische Benutzeroberfläche**: Einfache Bedienung per Drag & Drop und Klick.
- **Gender-Filterung**: Automatische Erkennung und Filterung von männlichen und weiblichen Stimmen.
- **Intelligente Segmentierung**: Teilt Audiodateien basierend auf stillen Pausen und Prosodie-Analyse (Satzmelodie) in optimale Längen (standardmäßig 2.5-12 Sekunden).
- **Professionelles Audio-Preprocessing**:
  - Rauschunterdrückung (Spectral Gating)
  - Normalisierung (Peak & LUFS)
  - Hochpassfilter zur Entfernung von Störgeräuschen
  - De-Essing zur Reduzierung von Zischlauten
  - Automatisches Trimmen von Stille am Anfang und Ende
- **Qualitätskontrolle**: Verwirft automatisch Segmente mit schlechter Qualität (z.B. niedriges Signal-Rausch-Verhältnis, Clipping).
- **Transkription**: Nutzt OpenAI Whisper (oder das schnellere `faster-whisper`), um die Audio-Segmente zu transkribieren.
- **Caching**: Speichert bereits getätigte Transkriptionen, um wiederholte Arbeit zu vermeiden.
- **Export**: Erstellt ein ZIP-Archiv mit den verarbeiteten `.wav`-Dateien und einer `metadata.tsv`-Datei, die für das TTS-Training bereit ist.
- **Gender-Kalibrierung**: Ermöglicht das Trainieren eines eigenen Modells zur Geschlechtererkennung für bessere Ergebnisse.

## Anforderungen

Das Skript benötigt die folgenden Python-Bibliotheken:

- `pydub`
- `librosa`
- `numpy`
- `openai-whisper`
- `PySimpleGUI`
- `num2words`
- `scikit-learn`
- `soundfile`
- `gTTS`
- `faster-whisper`
- `pytest` (für die Entwicklung)
- `pytest-mock` (für die Entwicklung)

## Installation

1.  Stellen Sie sicher, dass Sie Python 3.8 oder neuer installiert haben.
2.  Installieren Sie die erforderlichen Bibliotheken mit `pip`:
    ```bash
    pip install -r requirements.txt
    ```
3.  Stellen Sie außerdem sicher, dass `ffmpeg` auf Ihrem System installiert und im System-PATH verfügbar ist. `pydub` benötigt es für die Audioverarbeitung.

## Verwendung

1.  Starten Sie das Skript über die Kommandozeile:
    ```bash
    python main1.py
    ```
2.  **Dateien hinzufügen**: Fügen Sie Audiodateien (.wav, .mp3, etc.) oder ganze Ordner über den "Hinzufügen"-Button oder per Drag & Drop hinzu.
3.  **Einstellungen wählen**:
    - **Whisper Modell**: Wählen Sie die gewünschte Modellgröße (z.B. `small` für einen guten Kompromiss aus Geschwindigkeit und Genauigkeit).
    - **Engine**: `auto` wählt automatisch die schnellste verfügbare Whisper-Implementierung (`faster-whisper` wird bevorzugt).
    - **Gender-Filter**: Wählen Sie, ob Sie nur männliche, nur weibliche oder alle Stimmen verarbeiten möchten.
4.  **Ressourcen laden**: Klicken Sie auf "Ressourcen laden", um das gewählte Whisper-Modell in den Speicher zu laden. Der Start-Button wird danach aktiv.
5.  **Start**: Klicken Sie auf "Start", um die Verarbeitungspipeline zu beginnen.
6.  **Ergebnis**: Nach Abschluss des Prozesses finden Sie im `results`-Ordner ein ZIP-Archiv (`session_JJJJMMTT_HHMMSS.zip`), das Ihren fertigen Datensatz enthält.

## Die Pipeline-Schritte im Detail

1.  **Gender-Filterung**: Wenn ein Filter (männlich/weiblich) aktiv ist, wird jede Datei analysiert und nur die passenden werden für die weiteren Schritte ausgewählt.
2.  **Segmentierung**: Jede Audiodatei wird auf Basis von stillen Passagen und der Satzmelodie in kleinere Segmente zerlegt. Das Ziel ist es, einzelne Sätze oder Teilsätze zu isolieren.
3.  **Preprocessing**: Jedes einzelne Segment durchläuft eine Kette von Audio-Filtern, um die Qualität zu maximieren. Segmente, die die Qualitätsprüfung nicht bestehen, werden verworfen.
4.  **Transkription**: Die hochwertigen Segmente werden mit Whisper transkribiert. Der Text wird normalisiert (Zahlen in Wörter umgewandelt, Satzzeichen entfernt).
5.  **Export**: Alle erfolgreich verarbeiteten Segmente werden als `.wav`-Dateien zusammen mit einer `metadata.tsv`-Datei in einem ZIP-Archiv gespeichert.

## Gender-Kalibrierung

Wenn die automatische Geschlechtererkennung für Ihr Audiomaterial nicht optimal funktioniert, können Sie ein eigenes Modell trainieren:

1.  Klicken Sie auf "🎙️ Gender-Kalibrierung".
2.  Wählen Sie einige Beispiel-Audiodateien aus, die eindeutig männliche oder weibliche Sprecher enthalten.
3.  Markieren Sie die entsprechenden Dateien in der Tabelle und weisen Sie ihnen über die Buttons "Als Männlich" / "Als Weiblich" das korrekte Geschlecht zu.
4.  Klicken Sie auf "Training starten". Sie benötigen mindestens 2 Beispiele für jedes Geschlecht.
5.  Das neu trainierte Modell wird automatisch für zukünftige Filterungen verwendet.
