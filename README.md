# TTS-Datensatz-Formatter

Dieses Tool ist eine grafische Benutzeroberfläche (GUI) zur Verarbeitung von Audio-Dateien, um formatierte Datensätze für das Training von Text-to-Speech (TTS)-Modellen zu erstellen. Die Anwendung segmentiert lange Audio-Dateien, transkribiert die Segmente und erstellt eine Metadaten-Datei, die für TTS-Trainings-Frameworks wie Coqui-TTS oder Tacotron benötigt wird.

## Funktionen

- **Grafische Benutzeroberfläche:** Einfach zu bedienende Oberfläche, erstellt mit PySimpleGUI.
- **Audio-Segmentierung:** Automatisches Aufteilen von langen Audio-Dateien in kürzere Segmente basierend auf Stille.
- **Hochwertige Transkription:** Nutzt OpenAI's Whisper-Modelle (via `openai-whisper` oder `faster-whisper`) für präzise Transkriptionen.
- **Geschlechtererkennung:** Filtert Audio-Dateien nach Geschlecht (männlich/weiblich) mithilfe eines trainierbaren Klassifikators.
- **Kalibrierung der Geschlechtererkennung:** Ermöglicht das Trainieren des Geschlechter-Klassifikators mit eigenen Audio-Beispielen für höhere Genauigkeit.
- **Caching:** Speichert bereits transkribierte Segmente, um wiederholte Verarbeitung zu vermeiden.
- **Export:** Gibt einen sauberen Datensatz als ZIP-Archiv aus, das einen `wavs`-Ordner und eine `metadata.tsv`-Datei enthält.

## Installation

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/jeannetteetzold1980-sketch/mastermind_tts-
    cd mastermind_tts-
    ```

2.  **Abhängigkeiten installieren:**
    Es wird empfohlen, eine virtuelle Umgebung zu verwenden.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **FFmpeg installieren:**
    Für die Audioverarbeitung (Laden und Konvertieren verschiedener Formate) wird `ffmpeg` benötigt. Stellen Sie sicher, dass es in Ihrem System-PATH verfügbar ist.

    -   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    -   **macOS (mit Homebrew):** `brew install ffmpeg`
    -   **Windows:** Laden Sie eine Build von der [offiziellen FFmpeg-Website](https://ffmpeg.org/download.html) herunter und fügen Sie den `bin`-Ordner zu Ihrer `PATH`-Umgebungsvariable hinzu.

## Verwendung

1.  **Anwendung starten:**
    ```bash
    python main.py
    ```

2.  **Dateien hinzufügen:**
    -   Klicken Sie auf "Hinzufügen", um Audio-Dateien (z.B. `.wav`, `.mp3`) auszuwählen.
    -   Oder ziehen Sie Dateien und Ordner per Drag & Drop in die Dateiliste.

3.  **Einstellungen vornehmen:**
    -   **Whisper Modell:** Wählen Sie die gewünschte Modellgröße (z.B. `small`, `medium`). Größere Modelle sind genauer, aber langsamer.
    -   **Engine:** `auto` wählt automatisch die beste verfügbare Whisper-Implementierung (`faster-whisper` wird bevorzugt).
    -   **Geschlechter-Filter:** Wählen Sie, ob alle, nur männliche oder nur weibliche Stimmen verarbeitet werden sollen.

4.  **Ressourcen laden:**
    -   Klicken Sie auf "Ressourcen laden", um das ausgewählte Whisper-Modell in den Speicher zu laden. Der "Start"-Button wird danach aktiv.

5.  **Verarbeitung starten:**
    -   Klicken Sie auf "Start", um die Segmentierung und Transkription zu beginnen. Der Fortschritt wird in der GUI angezeigt.

6.  **Ergebnis:**
    -   Nach Abschluss des Prozesses wird eine ZIP-Datei im `results`-Verzeichnis erstellt. Diese Datei enthält die Audio-Segmente und die `metadata.tsv`.

### Gender-Kalibrierung

Wenn die automatische Geschlechtererkennung nicht zuverlässig funktioniert, können Sie das Modell mit Ihren eigenen Daten kalibrieren:

1.  Klicken Sie auf "🎙️ Gender-Kalibrierung".
2.  Laden Sie mehrere Audio-Dateien, die jeweils eine einzelne Person enthalten.
3.  Wählen Sie die Dateien in der Tabelle aus und weisen Sie ihnen das korrekte Geschlecht ("Als Männlich" / "Als Weiblich") zu.
4.  Klicken Sie auf "Training starten", wenn Sie genügend Beispiele für beide Geschlechter haben. Das trainierte Modell wird für zukünftige Filterungen verwendet.

## Tests

Um die Funktionalität des Projekts zu überprüfen, können Sie die Test-Suite ausführen:

```bash
pytest
```