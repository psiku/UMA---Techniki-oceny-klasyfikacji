# UMA---Techniki-oceny-klasyfikacji
Repozytorium w ramach projektu z przedmiotu "Uczenie Maszynowe"

## Zadanie do wykonania
Zaimplementuj techniki oceny klasyfikacji dla zestawów danych dotyczących raka piersi i jakiegoś innego, które dostępne są w http://archive.ics.uci.edu/ml/datasets

## Struktura projektu
```text
UMA/
├── datasets/                  # Zbiory danych (.csv) i ich organizacja
│   ├── breast_cancer/
│   └── students/
│
├── notebooks/                 # Notebooki do eksploracji danych i prezentacji wyników
│
├── reports/                   # Raporty, wyniki eksperymentów, podsumowania
│
├── src/                       # Główna logika projektu (implementacje i klasy)
│   ├── confusion_matrix/      # Własna implementacja i wizualizacja macierzy pomyłek
│   ├── data/                  # Ładowanie i wstępne przetwarzanie danych
│   │   ├── datasets_test.py
│   │   └── merge_datasets.py
│   ├── models/                # Trening i testowanie klasyfikatorów (SVM, Las losowy itp.)
│   ├── performance_metrics/   # Implementacja metryk: acc, prec, recall, f1
│   └── roc_curve/             # Własna implementacja i rysowanie krzywej ROC
│
├── LICENSE
└── README.md
'''

