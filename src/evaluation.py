import pandas as pd
import numpy as np
from portfolio import get_wml

#計算我自己複製的DM分組中每個Decile的年度化報酬、波動度、sharpe ratio、偏態以及和原始DM檔的相關性
#驗證有沒有高度一致及看風險調整後表現
def generate_dm_summary_stats(CRSP_Stocks_Momentum_returns: pd.DataFrame, DM_returns: pd.DataFrame):
    
    def get_summary_stats(x: pd.Series):
        # Annualized Mean
        ann_mean = x.DM_Ret_replica.mean()*12
        # Standard Deviation
        ann_std = x.DM_Ret_replica.std()*np.sqrt(12)
        # Sharpe Ratio
        sr = ann_mean/ann_std
        # Skewness
        skewness = np.log(x.DM_Ret_replica+x.Rf+1).skew()
        # correlation
        corr = x.loc[:, ["DM_Ret_replica", "DM_Ret_true"]].corr().loc["DM_Ret_replica", "DM_Ret_true"]
        return [round(ann_mean*100, 2), round(ann_std*100, 2), round(sr, 2), round(skewness, 2), round(corr, 4)]
    
    df = CRSP_Stocks_Momentum_returns.copy()
    rf = CRSP_Stocks_Momentum_returns.loc[:, ["Year", "Month", "Rf"]]
    
    rf = rf.groupby(["Year", "Month"])["Rf"].first().reset_index()
    df = get_wml(df, "DM_Ret")
    
    df = pd.merge(df, rf, how="left", on=["Year", "Month"])
    DM_returns = get_wml(DM_returns, "DM_Ret")
    
    df = pd.merge(df, DM_returns, how="left", left_on=["Year", "Month", "decile"], right_on=["Year", "Month", "decile"], suffixes=("_replica", "_true"))
    
    df.loc[df.decile!="WML", "DM_Ret_replica"] = df.loc[df.decile!="WML", "DM_Ret_replica"] - df.loc[df.decile!="WML", "Rf"]
    df.loc[df.decile!="WML", "DM_Ret_true"] = df.loc[df.decile!="WML", "DM_Ret_true"] - df.loc[df.decile!="WML", "Rf"]
    
    res_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "WML"]
    res = [get_summary_stats(df.loc[df.decile==decile, :]) for decile in res_index]
    res_col = ["Excess Return", "Volatility", "Sharpe Ratio", "Skewness", "corr w/ original"]
    res = pd.DataFrame(res, index=res_index, columns=res_col)
    return res.T

#整理KRF decile報酬
def clean_KRF_ret(df: pd.DataFrame):
    df_clean = df.copy()
    df_clean.columns = np.arange(1, 11, 1)
    df_clean = df_clean.assign(
        Date = lambda x: pd.to_datetime(x.index, format='%Y%m'),
        Year = lambda x: x.Date.dt.year,
        Month = lambda x: x.Date.dt.month,
    ).reset_index(drop=True).drop("Date", axis=1)
   
    # Melt the columns to row
    df_clean = df_clean.melt(["Year","Month"], np.arange(1, 11, 1), "decile", "KRF_Ret")
    df_clean.loc[:, "KRF_Ret"] = df_clean.loc[:, "KRF_Ret"]/100
    return df_clean

#計算我複製出來的KRF decile投資組合的指標
def generate_krf_summary_stats(CRSP_Stocks_Momentum_returns: pd.DataFrame, KRF_returns: pd.DataFrame):
    def get_summary_stats(x: pd.Series):
        # Annualized Mean
        ann_mean = x.KRF_Ret_replica.mean()*12
        # Standard Deviation
        ann_std = x.KRF_Ret_replica.std()*np.sqrt(12)
        # Sharpe Ratio
        sr = ann_mean/ann_std
        # Skewness
        skewness = np.log(x.KRF_Ret_replica+x.Rf+1).skew()
        # correlation
        corr = x.loc[:, ["KRF_Ret_replica", "KRF_Ret_true"]].corr().loc["KRF_Ret_replica", "KRF_Ret_true"]
        return [round(ann_mean,4)*100, round(ann_std,4)*100, round(sr, 2), round(skewness, 2), round(corr, 4)]
    
    df = CRSP_Stocks_Momentum_returns.copy()
    rf = CRSP_Stocks_Momentum_returns.loc[:, ["Year", "Month", "Rf"]]
    
    rf = rf.groupby(["Year", "Month"])["Rf"].first().reset_index()
    df = get_wml(df, "KRF_Ret")
    
    df = pd.merge(df, rf, how="left", on=["Year", "Month"])
    KRF_returns = get_wml(KRF_returns, "KRF_Ret")
    
    df = pd.merge(df, KRF_returns, how="left", left_on=["Year", "Month", "decile"], right_on=["Year", "Month", "decile"], suffixes=("_replica", "_true"))
    
    df.loc[df.decile!="WML", "KRF_Ret_replica"] = df.loc[df.decile!="WML", "KRF_Ret_replica"] - df.loc[df.decile!="WML", "Rf"]
    df.loc[df.decile!="WML", "KRF_Ret_true"] = df.loc[df.decile!="WML", "KRF_Ret_true"] - df.loc[df.decile!="WML", "Rf"]
    
    res_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "WML"]
    res = [get_summary_stats(df.loc[df.decile==decile, :]) for decile in res_index]
    res_col = ["Excess Return", "Volatility", "Sharpe Ratio", "Skewness", "corr w/ original"]
    res = pd.DataFrame(res, index=res_index, columns=res_col)
    return res.T

#將paper提供的報酬檔案做清理與標準化, to compare
def clean_DM_ret(df: pd.DataFrame):
    df_clean = df.copy()
    df_clean.columns = ["date", "decile", "DM_Ret", "_", "__"]
    df_clean = df_clean.assign(
        Date = lambda x: pd.to_datetime(x.iloc[:, 0], format='%Y%m%d'),
        Year = lambda x: x.Date.dt.year,
        Month = lambda x: x.Date.dt.month,
    )
    # Rearranging and dropping the columns
    return df_clean.loc[:, ["Year", "Month", "decile", "DM_Ret"]]