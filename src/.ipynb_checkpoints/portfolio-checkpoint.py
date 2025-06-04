import pandas as pd
import numpy as np

# 分組(paper vs French)
def assign_momentum_deciles(CRSP_Stocks_Momentum: pd.DataFrame):
    def get_decile(x):
        # Daniel & Mokskowitz
        # Get Breakpoints based on all Stocks
        x.loc[:, "DM_decile"] = pd.qcut(x.Ranking_Ret, 10, labels=np.arange(1, 11, 1), precision=5).astype(int)
        # Kenneth R. French (Below code are from Momentum_weekly_FULL_CODE.ipynb provided by the professor)
        # Get Breakpoints based on only NYSE Stocks (Exchange Code = 1)
        nyse_breakpoint = pd.qcut(x.loc[x.EXCHCD==1, "Ranking_Ret"], 10, retbins=True, labels=False)[1]
        nyse_breakpoint[0], nyse_breakpoint[10] = -np.inf, np.inf
        x.loc[:, "KRF_decile"] = pd.cut(x.Ranking_Ret, bins=nyse_breakpoint, labels=np.arange(1, 11, 1), precision=5).astype(int)
        return x
    
    CRSP_Stocks_Momentum_decile = CRSP_Stocks_Momentum.groupby(["Year", "Month"]).apply(get_decile, include_groups=False).reset_index()
    return CRSP_Stocks_Momentum_decile.loc[:, ["Year", "Month", "PERMNO", "lag_Mkt_Cap", "Ret", "DM_decile", "KRF_decile", "EXCHCD"]]

#計算每個月時每個decile投資組合的市值加權報酬率
def calculate_portfolio_returns(CRSP_Stocks_Momentum_decile: pd.DataFrame, FF_mkt: pd.DataFrame):
    def get_vw_rets(x):
        x.loc[:, "total_mv"] = x.lag_Mkt_Cap.sum()
        x.loc[:, "Port_Ret"] = np.sum(x.Ret * x.lag_Mkt_Cap) / x.total_mv
        return x[["Port_Ret"]].iloc[-1]
    
    # Get Value-weighted return for Daniel & Mokskowitz Portfolio
    DM_ret = CRSP_Stocks_Momentum_decile.groupby(["Year", "Month", "DM_decile"]).apply(get_vw_rets, include_groups=False).reset_index()
    DM_ret = DM_ret.loc[:, ["Year", "Month", "DM_decile", "Port_Ret"]].rename({"DM_decile":"decile", "Port_Ret":"DM_Ret"}, axis=1)
    
    # Get Value-weighted return for Kenneth R. French
    KRF_ret = CRSP_Stocks_Momentum_decile.groupby(["Year", "Month", "KRF_decile"]).apply(get_vw_rets, include_groups=False).reset_index()
    KRF_ret = KRF_ret.loc[:, ["Year", "Month", "KRF_decile", "Port_Ret"]].rename({"KRF_decile":"decile", "Port_Ret":"KRF_Ret"}, axis=1)
    
    # Extract Rf rate for each month
    Rf = FF_mkt.loc[:, ["Year", "Month", "Rf"]]
    Rf.loc[:, "Rf"] = Rf.loc[:, "Rf"]
    
    # Merge value returns for both portfolio and risk free rate
    CRSP_Stocks_Momentum_returns = pd.merge(DM_ret, KRF_ret, how="inner", on=["Year", "Month", "decile"])
    CRSP_Stocks_Momentum_returns = pd.merge(CRSP_Stocks_Momentum_returns, Rf, how="left", on=["Year", "Month"])
    
    return CRSP_Stocks_Momentum_returns

#計算每個月winner-loser的報酬率
def get_wml(df: pd.DataFrame, col: str):
        # Pivot the dataframe
        pivot = df.pivot(index=["Year","Month"], columns="decile", values=col)
        # Derive WML returns
        pivot.loc[:, "WML"] = pivot[10] - pivot[1]
        # Restore the original format
        pivot = pivot.reset_index().melt(["Year","Month"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "WML"], "decile", col)
        return pivot