"""Projektweite Exception-Hierarchie.

Anstatt überall eingebaute Exceptions (Exception, ValueError, ...) zu
fangen und stillschweigend zu behandeln, definiert dieses Modul eigene,
aussagekräftige Fehlertypen für die einzelnen Stufen der Pipeline
(Datenaufbereitung, Modell laden, Inferenz, Persistierung).

Aufrufende Funktionen können so gezielt auf bestimmte Fehlerklassen
reagieren, statt jeden Fehler gleich zu behandeln und zu verschlucken.
Alle Exceptions erben von ProjectError, sodass äußere Schichten (z. B.
ein CLI-Skript) bei Bedarf weiterhin alle projektspezifischen Fehler
mit einem einzigen except-Block abfangen können.
"""

from __future__ import annotations


class ProjectError(Exception):
    """Basisklasse für alle projektspezifischen Fehler.

    Dient als gemeinsamer Anker für except-Blöcke, die jegliche im
    Projekt definierte Fehlerklasse abfangen wollen, ohne dabei
    unerwartete Fremdfehler (z. B. Programmierfehler) zu verschlucken.
    """


class DataPreparationError(ProjectError):
    """Wird ausgelöst, wenn Rohdaten nicht geladen oder bereinigt werden können.

    Beispiele: fehlende oder leere CSV-Datei, fehlerhaftes Encoding,
    nach der Bereinigung verbleiben keine gültigen Zeilen.
    """


class ModelLoadError(ProjectError):
    """Wird ausgelöst, wenn ein trainiertes Modell nicht geladen werden kann.

    Beispiele: Checkpoint-Datei fehlt, beschädigte .pt/.pkl/.keras-Datei,
    inkompatible Modellarchitektur beim Laden der state_dict.
    """


class InferenceError(ProjectError):
    """Wird ausgelöst, wenn die Vorhersage eines geladenen Modells fehlschlägt.

    Diese Exception ersetzt das frühere Verhalten, den gesamten Prozess
    über sys.exit() zu beenden. Aufrufende Schichten (z. B. ein Dashboard
    oder ein Service) können den Fehler so abfangen und angemessen
    reagieren, anstatt dass der komplette Prozess abstürzt.
    """


class TrainingError(ProjectError):
    """Wird ausgelöst, wenn das Training eines Modells fehlschlägt.

    Beispiele: leerer Trainingsdatensatz, NaN-Verlust während des
    Trainings, Fehler beim Speichern eines Checkpoints.
    """


class PersistenceError(ProjectError):
    """Wird ausgelöst, wenn ein Datenbankzugriff fehlschlägt.

    Beispiele: Datenbankverbindung nicht verfügbar, Constraint-Verletzung
    beim Insert, Transaktion konnte nicht committet werden.
    """