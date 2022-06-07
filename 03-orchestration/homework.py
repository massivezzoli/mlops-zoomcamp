import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DATE_FORMAT = "%Y-%m-%d"
DATE_FILE_FORMAT = "%Y-%m"

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, date):
    logger =  get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    with open(f"models/dv-{date}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    with open(f"models/model-{date}.bin", "wb") as m_out:
            pickle.dump(lr, m_out)
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger =  get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    logger =  get_run_logger()
    if date is None:
        date = date.today()
    else:
        date = datetime.strptime(date, DATE_FORMAT)
    
    train_date = (date - relativedelta(months=2)).strftime(DATE_FILE_FORMAT)
    val_date = (date - relativedelta(months=1)).strftime(DATE_FILE_FORMAT)
    date = date.strftime(DATE_FORMAT)

    train_path = f'../data/fhv_tripdata_{train_date}.parquet'
    val_path = f'../data/fhv_tripdata_{val_date}.parquet'

    logger.info(f"train data path: {train_path}")
    logger.info(f"validation data path: {val_path}")

    return train_path, val_path, date

@flow
def main(date='2021-08-15'):

    categorical = ['PUlocationID', 'DOlocationID']
    train_path , val_path , date = get_paths(date).result()
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name="cron-schedule-model_train",
    flow=main,
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    tags=['dep_test']
)

