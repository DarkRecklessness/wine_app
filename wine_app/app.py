import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

folder = Path(__file__).parent
model = joblib.load(folder / 'model.joblib')
config = json.loads((folder / 'config.json').read_text())
data = pd.read_csv(folder / 'test.csv')
X = data.drop(columns='target')
y = data['target']
features = X.columns.tolist()

st.title('Оценка качества вина')
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.001, format='%.3f')
tab1, tab2, tab3, tab4 = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'
])

with tab1:
    values = {}
    for feature in features:
        if feature == 'is_red':
            values[feature] = int(st.checkbox('Красное вино'))
        else:
            values[feature] = st.number_input(
                feature,
                min_value=float(X[feature].min()),
                max_value=float(X[feature].max()),
                value=float(X[feature].median()),
                format='%.5f',
            )
    wine = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.2%}')
    st.write('Рекомендуем' if probability >= threshold else 'Не рекомендуем')

with tab2:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y, predictions, zero_division=0)
    recall = metrics.recall_score(y, predictions, zero_division=0)
    f1 = metrics.f1_score(y, predictions, zero_division=0)
    st.metric('Precision', f'{precision:.4f}')
    st.metric('Recall', f'{recall:.4f}')
    st.metric('F1', f'{f1:.4f}')

    matrix = metrics.confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = matrix.ravel()
    fpr, tpr, _ = metrics.roc_curve(y, probabilities)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.scatter(fp / (fp + tn), recall, color='red', label='Текущий порог')
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots()
    ax.step(pr_recall, pr_precision, where='post')
    ax.scatter(recall, precision, color='red', label='Текущий порог')
    ax.set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write('При повышении порога подборка становится меньше и обычно точнее. '
             'При снижении порога находим больше хороших вин, но можем добавить больше посредственных.')

with tab3:
    file = st.file_uploader('CSV с признаками вин без target', type='csv')
    if file is not None:
        try:
            table = pd.read_csv(file)
            missing = [name for name in features if name not in table.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            else:
                result = table.copy()
                result['probability'] = model.predict_proba(table[features])[:, 1]
                result['prediction'] = (result['probability'] >= threshold).astype(int)
                st.dataframe(result)
                st.download_button(
                    'Скачать результаты',
                    result.to_csv(index=False).encode('utf-8-sig'),
                    file_name='predictions.csv',
                    mime='text/csv',
                )
        except Exception as error:
            st.error('Не удалось обработать CSV: ' + str(error))

with tab4:
    weights = pd.Series(model[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    weights.plot.barh(ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = weights[weights > 0].sort_values(ascending=False).head(3)
    negative = weights[weights < 0].head(3)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(positive.index))
    st.write('Сильнее всего против: ' + ', '.join(negative.index))
