import pandas as pd
import numpy as np

# 對於原始的 CRSP 股票資料，進行初步的資料清理與格式標準化。
def clean_CRSP_Stocks(df: pd.DataFrame):
    # Casting date to datetime
    df_clean = df.copy()
    df_clean.date = pd.to_datetime(df_clean.date, format='%Y-%m-%d')
    df_clean.EXCHCD = df_clean.loc[:, "EXCHCD"].astype("int64")

    # Filterling only common share class (10, 11) and  major exchange(1, 2, 3)
    df_clean = df_clean[df_clean.EXCHCD.isin([1, 2, 3]) & df_clean.SHRCD.isin([10, 11])]
    
    # replace all non-numeric (including nan) value to zero
    df_clean.loc[:, "valid_ret"] = ~(df_clean.RET.isna() & df_clean.DLRET.isna())
    
    df_clean.DLRET = df_clean.DLRET.fillna(0)
    df_clean.DLRET = df_clean.DLRET.replace(["S", "T", "A", "P"], 0).astype(float)
    df_clean.RET = df_clean.RET.fillna(0)
    df_clean.RET = df_clean.RET.replace(["A", "B", "C", "D", "E", "."], 0).astype(float)
    
    # Merging Delisting and Holding Returns
    df_clean.RET = (1+df_clean.RET) * (1+df_clean.DLRET) - 1

    # Sorting the dataframe by date and premno
    df_clean.sort_values(["date", "PERMNO"], ignore_index=True, inplace=True)
    df_clean.drop(["SHRCD", "DLRET"], axis=1, inplace=True)
    
    # Change datatype for EXCHCD
    df_clean.loc[:, "EXCHCD"] = df_clean.loc[:, "EXCHCD"].astype(np.int8)
    
    # Remove Null Quote
    return df_clean.rename({"RET":"Ret"}, axis=1)

#計算市值與上一期市值以計算市值權重(避免look-ahead bias)
def get_mkt_cap(df: pd.DataFrame):
    df_mkt_cap = df.copy()
    # Calculate Market Cap for each stock 
    df_mkt_cap.loc[:, "mkt_cap"]= np.abs(df_mkt_cap.PRC) * df_mkt_cap.SHROUT / 1e3
    
    # Calculate Lagged Market Cap for weight calculations
    def get_lagged_mkt_cap(df):
        df = df.assign(
            month_diff = lambda x: (x.date.dt.year - x.date.shift(1).dt.year) * 12 + x.date.dt.month - x.date.shift(1).dt.month,
            lag_Mkt_Cap = lambda x: (x.mkt_cap.shift(1) * (x.month_diff==1))
        ).drop("month_diff", axis=1)
        return df
    
    df_mkt_cap = df_mkt_cap.groupby(["PERMNO"]).apply(get_lagged_mkt_cap, include_groups=False).reset_index().drop("level_1", axis=1)
    return df_mkt_cap

#計算動能排序來分組
def get_ranking_returns(df: pd.DataFrame):
    df.sort_values(["PERMNO", "date"], inplace=True)
    unique_date = df.sort_values("date").date.unique()
    date = df.date
    # Check constraints
    df = df.assign(
        valid_prc = lambda x: x.PRC.notna(),
        # Floor return to avoid errors when converting to log returns
        floor_Ret = lambda x: np.maximum(x.Ret, -0.99999),
        log_Ret = lambda x: np.log1p(x.floor_Ret),
        valid_shr = lambda x: x.SHROUT.notna(),
        valid_mktcap = lambda x: x.lag_Mkt_Cap.notna() & x.lag_Mkt_Cap > 0,
        valid_formation = lambda x: x.valid_prc & x.valid_shr & x.valid_mktcap
    )
    res = []
    
    for start, ret_start, ret_end, end in zip(unique_date, unique_date[1:], unique_date[11:], unique_date[13:]):
        df_filtered = df.loc[(date>=start)&(date<=end)]

        # Filter out all ineligible stocks based on the requirements
        m_1 = (df_filtered.date==start)&(df_filtered.valid_prc)
        eligible_permno_1 = set(df_filtered.loc[m_1, "PERMNO"])
        
        m_2 = (df_filtered.date==ret_end)&(df_filtered.valid_ret)
        eligible_permno_2 = set(df_filtered.loc[m_2, "PERMNO"])
        
        m_3 = (df_filtered.date==end)&(df_filtered.valid_formation)
        eligible_permno_3 = set(df_filtered.loc[m_3, "PERMNO"])
        
        eligible_permno_4 = df_filtered.loc[(df_filtered.date>=ret_start)&(df_filtered.date<=ret_end)].groupby(["PERMNO"])['valid_ret'].sum().ge(8).reset_index()
        eligible_permno_4 = set(eligible_permno_4.loc[eligible_permno_4.valid_ret, "PERMNO"])
        
        eligible_permno = eligible_permno_1 & eligible_permno_2 & eligible_permno_3 & eligible_permno_4
        ret_t = df_filtered[(df_filtered.date>=ret_start) & (df_filtered.PERMNO.isin(eligible_permno))]

        cum_rets = ret_t[ret_t.date<=ret_end].groupby(["PERMNO"])["log_Ret"].sum().rename("Ranking_Ret")

        ret_t  = pd.merge(ret_t.loc[ret_t.date==end, :], cum_rets, how="inner", left_on=["PERMNO"], right_index=True)
        res.append(ret_t)
        
    res = pd.concat(res) 
    res = res.assign(
            Year = res.date.dt.year,
            Month = res.date.dt.month
        ).drop("date", axis=1)
    
    
    return res.reset_index().loc[:, ["Year", "Month", "PERMNO", "EXCHCD", "lag_Mkt_Cap", "Ret", "Ranking_Ret"]]

def prepare_momentum_data(CRSP_Stocks: pd.DataFrame):
    CRSP_Stocks_clean = clean_CRSP_Stocks(CRSP_Stocks)
    CRSP_Stocks_mkt_cap = get_mkt_cap(CRSP_Stocks_clean)
    CRSP_Stocks_Momentum = get_ranking_returns(CRSP_Stocks_mkt_cap)
    
    return CRSP_Stocks_Momentum

#清理Fama-French資料
def clean_FF_mkt(df: pd.DataFrame):
    df_clean = df.copy()
    df_clean = df_clean.assign(
        Date = lambda x: pd.to_datetime(x.iloc[:, 0], format='%Y%m'),
        Year = lambda x: x.Date.dt.year,
        Month = lambda x: x.Date.dt.month,
    )
    df_clean.loc[:, ["Mkt-RF", "SMB", "HML", "RF"]] = df_clean.loc[:, ["Mkt-RF", "SMB", "HML", "RF"]]/100
    # Rearranging and dropping the columns
    df_clean = df_clean.loc[:, ["Year", "Month", "Mkt-RF", "SMB", "HML", "RF"]]
    df_clean.columns = ["Year", "Month", "Market_minus_Rf", "SMB", "HML", "Rf"]
    return df_clean