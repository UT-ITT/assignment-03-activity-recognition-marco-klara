#### WICHTIG 
Zum Bewerten der accuracy auf einem unabhängigen set (Bewertung der Tutoren) bitte folgendes beachten:

    - Testdaten (als .csv) in den Ordner "test" ablegen
    - test_model(TEST_PATH)  innerhalb der load_model()-funktion entkommentieren

Dadurch wird beim Programmstart, nachdem das Modell fertig trainiert wurde, einmal die model-accuracy auf den Testdaten geprintet

#### APP Funktionsweise 
Der Fitness Trainer lässt sich starten, indem man die Datei fitness_trainer.py ausführt.
Dann wird das Model mit den gesammelten Daten trainiert.
Ist das Model fertig, wird die accuracy, sowie der F1 score im Terminal ausgegeben und man kommt zum eigentlichen Fitness Trainer.
Hierbei wird immer zufällig eine der vier Aktivitäten ausgewählt und angezeigt.
Im Fenster ist ein Counter mit 10 Sekunden, der nur runterzählt, wenn die richtige Aktivität ausgeführt wird.
Sind die 10 Sekunden um, wird die nächste Aktivität angezeigt.
Der Fitness Trainer lässt sich durch die esc-Taste schließen.

#### Modell training 
Die Trainingsdaten werden als .csv Dateien eingelesen.
Damit die Trainingsdaten vergleichbar zu den Aufgenommenen Daten während des Trainings sind, müssen sie einheitlich sein.

Wir haben uns für Zeitfenster von einer Sekunde entschieden, um den Fortschritt in der fitness_trainer app flüssig zu aktualisieren und gleichzeitig genug information für eine akkurate Vorhersage zu gewährleisten

Die csv_feature_extraction funktion erstellt mithilfe der create_windows() funktion aus beliebig langen Datensätzen
1-Sekunden-Fenster (mit 20% überlappung) und speichert jedes Fenster als einzelnes Trainingssample.
Dadurch wird die Menge an Trainingsdaten zusätzlich erhöht, was die Modellgenauigkeit verbessert
(aus einer 10s datei werden ~45 trainings samples)

Für die Auswertung der Accuracy des Train-test-splits wird ein Group-Shuffle-Split genutzt, das verhindert, dass Trainings- und Testdaten die Daten der selben Person beinhalten. 
Da jede Person eine eigene Art der Bewegung hat, kann bei Random train-test-splits Bias entstehen, der die Modellgenauigkeit beim auswerten nicht korrekt repräsentiert
