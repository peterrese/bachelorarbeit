# Bachelorarbeit "Modellierung von Internetblasen durch Markow-Ketten" von Peter Rese
Hier ist der Code für meine Bachelorarbeit an der FU Berlin zu finden. Um ihn auszuführen, muss Python3 auf dem lokalen System installiert sein. Zudem müssen die folgenden Libraries installiert sein :
```
numpy
pandas
seaborn
matplotlib
python3-pip
``` 

Der relevante Code befindet sich in `Markow_Modell2_3.py` 

## Ausführung
Um diesen auszuführen, muss auf den meisten Systemen folgende Befehlkette ausgeführt werden:

### Linux-basierte Systeme
```
python3 -m venv bachelorarbeit
source bachelorarbeit/bin/activate
pip install -r requirements.txt
python Markow_Modell_2_3.py
``` 
### Windows 
```
python3 -m venv bachelorarbeit
. bachelorarbeit\scripts\activate
pip install -r .\requirements.txt
python .\Markow_Modell2_3.py
``` 

## Tuning der Ausführung
Um die Größe der erstellten Markov-Ketten Blasen anzupassen, gibt es drei Variablen, die angepasst werden müssen n1 (A), n2 (C), und m (B).  
Diese können im Code in Zeile 6, 7, und 8 angepasst werden.

