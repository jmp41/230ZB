import os
import pickle

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN, REG_US
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.riskmodel import StructuredCovEstimator
from qlib.utils import init_instance_by_config

provider_uri = "~/.qlib/qlib_data/us_data"
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# if not exists_qlib_data(provider_uri):
#     from qlib.tests.data import GetData
#     GetData().qlib_data(target_dir=provider_uri, region=REG_US)
qlib.init(provider_uri=provider_uri, region=REG_US)  # REG_US

region = "US"  # US
market = "sp500"  # sp500
label = 20
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "learn_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "DropnaLabel"},
        {"class": 'CSRankNorm', 'kwargs' : {'fields_group': 'label'}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": 'CSRankNorm', 'kwargs' : {'fields_group': 'label'}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "label": [f"Ref($close, -{label})/$close - 1"],
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2010-01-01", "2017-12-31"),
            "valid": ("2018-01-01", "2018-12-31"),
            "test": ("2019-01-01", "2020-08-01"),
        },
    },
}


def get_daily_code(df):
    return df.reset_index(level=0).index.values


def robust_z_score(df):
    return (df - np.nanmean(df)) / (0.0001 + np.nanstd(df))


def remove_ret_lim(df):
    loc = np.argwhere((df["label"].values.squeeze() < 0.099) & (df["label"].values.squeeze() > -0.099)).squeeze()

    return df.iloc[loc, :].fillna(method="ffill")


def prepare_risk_data(
    df_index, region="CN", suffix="Train", T=240, start_time="2007-01-01", riskdata_root="./riskdata"
):
    riskmodel = StructuredCovEstimator(scale_return = False)
    codes = df_index.groupby("datetime").apply(get_daily_code)
    ret_date = codes.index.values
    price_all = (
        D.features(D.instruments("all"), ["$close"], start_time=start_time, end_time=ret_date[-1])
        .squeeze()
        .unstack(level="instrument")
    )
    cur_idx = np.argwhere(price_all.index == ret_date[0])[0][0]
    assert cur_idx - T + 1 >= 0
    for i in range(len(ret_date)):
        date = pd.Timestamp(ret_date[i])
        print(date)
        ref_date = price_all.index[i + cur_idx - T + 1]
        code = codes[i]
        price = price_all.loc[ref_date:date, code]
        ret = price.pct_change().dropna(how='all')
        ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)

        ret = ret.groupby("datetime").apply(robust_z_score)
        try:
            cov_estimated = riskmodel.predict(ret, is_price=False, return_decomposed_components=False)
        except ValueError:
            print('Extreme situations: zero or one tradeable stock')
            cov_estimated = ret.cov()
        root = riskdata_root + region + suffix + "/" + date.strftime("%Y%m%d")
        os.makedirs(root, exist_ok=True)
        cov_estimated.to_pickle(root + "/factor_exp.pkl")

def prepare_data(riskdata_root="./riskdata", T=240, start_time="2010-12-01"):
    universe = D.features(D.instruments("sp500"), ["$close"], start_time=start_time).swaplevel().sort_index()

    price_all = (
        D.features(D.instruments("all"), ["$close"], start_time=start_time).squeeze().unstack(level="instrument")
    )

    # StructuredCovEstimator is a statistical risk model
    riskmodel = StructuredCovEstimator()

    for i in range(T - 1, len(price_all)):
        date = price_all.index[i]
        ref_date = price_all.index[i - T + 1]
        print(date)

        codes = universe.loc[date].index
        price = price_all.loc[ref_date:date, codes]

        # calculate return and remove extreme return
        ret = price.pct_change()
        ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)

        # run risk model
        F, cov_b, var_u = riskmodel.predict(ret, is_price=False, return_decomposed_components=True)

        # save risk data
        root = riskdata_root + "/" + date.strftime("%Y%m%d")
        os.makedirs(root, exist_ok=True)

        pd.DataFrame(F, index=codes).to_pickle(root + "/factor_exp.pkl")
        pd.DataFrame(cov_b).to_pickle(root + "/factor_cov.pkl")
        # for specific_risk we follow the convention to save volatility
        pd.Series(np.sqrt(var_u), index=codes).to_pickle(root + "/specific_risk.pkl")


dataset = init_instance_by_config(dataset_config)
df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

if region == "CN":
    df_train = remove_ret_lim(df_train)
    df_valid = remove_ret_lim(df_valid)
    df_test = remove_ret_lim(df_test)
elif region == "US":
    df_train["label"].clip(df_train["label"].quantile(0.025), df_train["label"].quantile(0.975), axis=1, inplace=True)
    df_train = df_train.fillna(method="ffill")
    df_valid = df_valid.fillna(method="ffill")
    df_test = df_test.fillna(method="ffill")
else:
    raise NotImplementedError

# train label cross-sectional z-score
# df_train["label"] = df_train["label"].groupby("datetime").apply(robust_z_score)
print(df_train.shape)
with open(
    "./data/{}_feature_dataset_market_{}_{}_start{}_end{}_label{}".format(region, market, "train", "2008-01-01", "2014-12-31",label), "wb"
) as f:
    pickle.dump(df_train, f)
with open(
    "./data/{}_feature_dataset_market_{}_{}_start{}_end{}_label{}".format(region, market, "valid", "2015-01-01", "2016-12-31",label), "wb"
) as f:
    pickle.dump(df_valid, f)
with open(
    "./data/{}_feature_dataset_market_{}_{}_start{}_end{}_label{}".format(region, market, "test", "2017-01-01", "2020-08-01",label), "wb"
) as f:
    pickle.dump(df_test, f)
print("Preparing features done!")

# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# qlib.init(provider_uri=provider_uri, region=REG_US)
# prepare_data()
# prepare_risk_data(df_train, region=region, suffix="Train", T=240, start_time="2007-01-01")
# prepare_risk_data(df_valid, region=region, suffix="Valid", T=240, start_time="2014-01-01")
# prepare_risk_data(df_test, region=region, suffix="Test", T=240, start_time="2015-01-01")
print("preparing risk data done!")
