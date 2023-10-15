import click
import numpy as np
from sklearn.model_selection import cross_val_score

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model
import pickle

@click.group()
def cli():
    pass

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")

    model = make_model()
    model.fit(X, y)

    with open(model_dump_filename, 'wb') as f:
        pickle.dump(model, f)

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, model_dump_filename, input_filename, output_filename):
    df = make_dataset(input_filename)
    X = make_features(df, task, train_mode=False, vectorizer_path="vectorizer/vectorizer.pkl")[0]  # Only extract X

    with open(model_dump_filename, 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    df['predictions'] = predictions
    df.to_csv(output_filename, index=False)

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task, train_mode=True, vectorizer_path="vectorizer/vectorizer.pkl")

    model = make_model()

    # Run k-fold cross validation. Print results
    evaluate_model(model, X, y)

def evaluate_model(model, X, y):
    # Scikit-learn has function for cross-validation
    scores = cross_val_score(model, X, y, scoring="accuracy")
    mean_accuracy = 100 * np.mean(scores)
    print(f"Accuracy: {mean_accuracy:.2f}%")

    return scores

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
