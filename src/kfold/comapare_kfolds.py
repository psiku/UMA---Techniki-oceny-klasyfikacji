import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from .kfold import ManualKFoldValidator


def compare_kfold_validators(model, X, y, stratified=False, n_splits=5, random_state=42, verbose=True):
    manual_validator = ManualKFoldValidator(
        model=model,
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        stratified=stratified
    )

    manual_df = manual_validator.cross_validate(
        X=X,
        y=y,
        metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        averages=['micro']
    )
    manual_df['source'] = 'manual'

    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='micro'),
        'recall': make_scorer(recall_score, average='micro'),
        'f1_score': make_scorer(f1_score, average='micro')
    }

    sk_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    records = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        for i, score in enumerate(sk_results[f'test_{metric}']):
            records.append({
                'fold': i + 1,
                'metric': metric,
                'average': 'micro',
                'score': score,
                'source': 'sklearn'
            })
    sk_df = pd.DataFrame.from_records(records)

    comparison_df = pd.concat([manual_df, sk_df], ignore_index=True)
    pivot_df = comparison_df.pivot_table(
        index=['fold', 'metric'],
        columns='source',
        values='score'
    ).reset_index()

    return pivot_df
