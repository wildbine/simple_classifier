from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go
import copy

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']

# Пороги, которые будем изменять
thresholds = [0.2, 0.4, 0.6, 0.8]

# Пустые списки для метрик
accuracies = []
f1_scores = []

# Пустые списки для PR-кривой
precisions = []
recalls = []

# Переменные для хранения лучших метрик
best_accuracy = 0
best_f1_score = 0
best_threshold = 0

for threshold in thresholds:
    # Если уверенность >= порогу, то баскетболист
    predicted_classes = (predictions >= threshold).astype(int)

    # Расчет TP, FP, FN, TN
    tp = np.sum((predicted_classes == 1) & (gt == 1))
    fp = np.sum((predicted_classes == 1) & (gt == 0))
    fn = np.sum((predicted_classes == 0) & (gt == 1))
    tn = np.sum((predicted_classes == 0) & (gt == 0))

    # Расчет метрик
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Сохранение метрик
    accuracies.append(accuracy)
    f1_scores.append(f1)

    # Сохранение метрик
    accuracies.append(accuracy)
    f1_scores.append(f1)

    # Сохранение лучших метрик
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
    if f1 > best_f1_score:
        best_f1_score = f1

    # Сохранение precision и recall для PR-кривой
    precisions.append(precision)
    recalls.append(recall)

# Кривая
fig = go.Figure(data=[
    go.Scatter(x=recalls, y=precisions, mode='lines+markers', marker=dict(size=8),
               text=[f'Threshold: {threshold}, Accuracy: {accuracy}, F1 Score: {f1}'
                     for threshold, accuracy, f1 in zip(thresholds, accuracies, f1_scores)],
               hoverinfo='text'),
])
fig.update_layout(
    title='PR Curve',
    xaxis=dict(title='Recall'),
    yaxis=dict(title='Precision'),
)
fig.show()

# Вывод лучших метрик и порога
print(f"Best Accuracy: {best_accuracy} (Threshold: {best_threshold})")
print(f"Best F1 Score: {best_f1_score} (Threshold: {best_threshold})")