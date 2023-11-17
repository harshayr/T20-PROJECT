from steps.data_ingestion import ingest_df
from steps.data_transformation import transfrom_df
from steps.model_train import train_model
from steps.model_evaluate import evaluate_df


def training_pipeline(data_path:str):
    df = ingest_df(data_path)
    transfrom_df(df)
    train_model(df)
    evaluate_df(df)
