#!/usr/bin/env python3
"""
全市场量化选股 - 高效版
1. SQL月频聚合（避免pandas groupby）
2. Top300活跃股
3. Python计算因子 + 月频回测
"""
import os, warnings, json
os.chdir("/Users/openclaw/.openclaw/workspace/vibe-trading")
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np

def to_exchange(tc):
    if "." in tc: return tc
    return f"sh.{tc}" if tc.startswith(("6","9","5")) else f"sz.{tc}"

conn = sqlite3.connect("/Users/openclaw/stock data/a_stock.db")

# 1. SQL月频聚合 + 选Top300活跃股
print("加载月频数据...")
monthly = pd.read_sql("""
    SELECT ts_code,
           substr(trade_date,1,7) as ym,
           MAX(trade_date) as last_date,
           MAX(close) as high,
           MIN(close) as low,
           AVG(vol) as avg_vol,
           AVG(close) as avg_close
    FROM daily
    WHERE trade_date >= "20180101"
    GROUP BY ts_code, substr(trade_date,1,7)
    ORDER BY ts_code, ym
""", conn)
conn.close()

monthly["code"] = monthly["ts_code"].apply(to_exchange)
monthly["avg_vol"] = pd.to_numeric(monthly["avg_vol"], errors="coerce").fillna(0)
monthly["high"] = pd.to_numeric(monthly["high"], errors="coerce")
monthly["low"] = pd.to_numeric(monthly["low"], errors="coerce")
monthly["avg_close"] = pd.to_numeric(monthly["avg_close"], errors="coerce")

# 选Top300活跃股
stock_avg_vol = monthly.groupby("code")["avg_vol"].mean().sort_values(ascending=False)
TOPN = 300
top_codes = set(stock_avg_vol.head(TOPN).index)
monthly = monthly[monthly["code"].isin(top_codes)].copy()
print(f"Top{TOPN}活跃股票  阈值: {stock_avg_vol.iloc[TOPN-1]:.0f}股/月")

# 2. 加载日线数据（只针对Top300）
print("加载日线数据...")
conn = sqlite3.connect("/Users/openclaw/stock data/a_stock.db")
# 构建in子句
placeholders = ",".join([f"'{c}'" for c in top_codes])
daily = pd.read_sql(
    f"SELECT ts_code, trade_date, close FROM daily WHERE ts_code IN ({placeholders}) AND trade_date >= '20180101' ORDER BY ts_code, trade_date",
    conn, parse_dates=["trade_date"]
)
conn.close()
daily["code"] = daily["ts_code"].apply(to_exchange)
daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
daily = daily.sort_values(["code","trade_date"]).reset_index(drop=True)
print(f"日线数据: {len(daily):,}行  {daily['code'].nunique()}只")

# 月频close用last_date
monthly = monthly.rename(columns={"last_date": "trade_date"})
monthly["trade_date"] = pd.to_datetime(monthly["trade_date"])
monthly = monthly.sort_values(["code","trade_date"]).reset_index(drop=True)
monthly["close"] = monthly["avg_close"]  # 用月均价作为收盘价代理

# 4. 计算截面因子（向量化）
print("计算因子...")
monthly["ret_1m"]  = monthly.groupby("code")["close"].pct_change(1)
monthly["ret_3m"]  = monthly.groupby("code")["close"].pct_change(3)
monthly["ret_6m"]  = monthly.groupby("code")["close"].pct_change(6)
monthly["ret_12m"] = monthly.groupby("code")["close"].pct_change(12)

monthly["high_12m"] = monthly.groupby("code")["close"].transform(lambda x: x.rolling(12, min_periods=6).max())
monthly["low_12m"]  = monthly.groupby("code")["close"].transform(lambda x: x.rolling(12, min_periods=6).min())
monthly["rs_12m"] = (monthly["close"] - monthly["low_12m"]) / (monthly["high_12m"] - monthly["low_12m"] + 1e-9)

# 波动率：用月收益率std*sqrt(12)
monthly["vol_12m"] = monthly.groupby("code")["ret_1m"].transform(lambda x: x.rolling(12, min_periods=6).std() * np.sqrt(12))

# 剔除上市<24月
monthly["listing_months"] = monthly.groupby("code")["trade_date"].transform(lambda x: range(len(x)))
monthly = monthly[monthly["listing_months"] >= 24].copy()
print(f"有效月频: {len(monthly):,}行  {monthly['code'].nunique()}只")

# 截面标准化
factor_cols = ["ret_1m","ret_3m","ret_6m","ret_12m","vol_12m","rs_12m"]
for col in factor_cols:
    mu = monthly.groupby("trade_date")[col].transform("mean")
    sd = monthly.groupby("trade_date")[col].transform("std")
    monthly[f"z_{col}"] = (monthly[col] - mu) / (sd + 1e-9)

# Alpha公式
monthly["alpha_A"] = 0.30*monthly["z_ret_3m"]+0.25*monthly["z_ret_6m"]+0.20*monthly["z_ret_12m"]-0.10*monthly["z_vol_12m"]
monthly["alpha_A"] = monthly.groupby("trade_date")["alpha_A"].rank(pct=True)

monthly["alpha_B"] = -0.30*monthly["z_ret_1m"]-0.20*monthly["z_rs_12m"]-0.15*monthly["z_vol_12m"]
monthly["alpha_B"] = monthly.groupby("trade_date")["alpha_B"].rank(pct=True)

monthly["alpha_C"] = -0.35*monthly["z_vol_12m"]+0.25*monthly["z_rs_12m"]-0.10*monthly["z_ret_1m"]
monthly["alpha_C"] = monthly.groupby("trade_date")["alpha_C"].rank(pct=True)

monthly["alpha_D"] = 0.15*monthly["z_ret_3m"]+0.15*monthly["z_ret_6m"]-0.15*monthly["z_vol_12m"]-0.10*monthly["z_ret_1m"]-0.10*monthly["z_rs_12m"]
monthly["alpha_D"] = monthly.groupby("trade_date")["alpha_D"].rank(pct=True)

# 5. 回测日期
reb_dates = sorted(monthly["trade_date"].unique())
first_date = reb_dates[0]
print(f"回测期: {first_date} ~ {reb_dates[-1]}  共{len(reb_dates)}月")

# 日线NAV
daily["date"] = daily["trade_date"].dt.date
all_daily = sorted(daily["date"].unique())
first_daily_idx = next((i for i,d in enumerate(all_daily) if d >= first_date.date()), 0)

def make_dict(alpha_col, top_n):
    d = {}
    for rd in reb_dates:
        dd = monthly[monthly["trade_date"] == rd]
        if len(dd) >= top_n:
            d[pd.Timestamp(rd)] = set(dd.nlargest(top_n, alpha_col)["code"].tolist())
    return d

def backtest(top_dict):
    cur = top_dict.get(pd.Timestamp(first_date), set())
    pv = 1.0
    pvs = {}
    prev_d = pd.Timestamp(first_date)
    prev_ts = None

    for i in range(first_daily_idx, len(all_daily)):
        d = all_daily[i]
        d_ts = pd.Timestamp(d)
        if d_ts in top_dict:
            cur = top_dict[d_ts]
            prev_ts = d_ts
        if prev_ts is None:
            prev_ts = d_ts
        if d == prev_ts:
            pvs[d] = pv
            continue
        pd_ = daily[daily["date"] == prev_ts.date()] if isinstance(prev_ts, pd.Timestamp) else daily[daily["date"] == prev_ts]
        cd_ = daily[daily["date"] == d]
        if len(cd_) == 0 or len(pd_) == 0:
            pvs[d] = pv
            prev_ts = d_ts
            continue
        rets = []
        for code in cur:
            pp = pd_[pd_["code"]==code]["close"]
            cp = cd_[cd_["code"]==code]["close"]
            if len(pp)>0 and len(cp)>0 and float(pp.iloc[0])>0:
                rets.append(float(cp.iloc[0])/float(pp.iloc[0])-1)
        if rets:
            pv = pv*(1+sum(rets)/len(rets))
        pvs[d] = pv
        prev_ts = d_ts

    rpv = pd.Series({ts: pvs[ts.date()] for ts in reb_dates if ts.date() in pvs})
    rpv.index = pd.to_datetime(rpv.index)
    if len(rpv) < 2:
        return {"total_ret": 0, "annual_ret": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "pl_ratio": 0, "monthly": pd.Series()}
    dr = rpv.pct_change().dropna()
    td = (rpv.index[-1]-rpv.index[0]).days
    nyr = td/365.25
    tr = float(rpv.iloc[-1]/rpv.iloc[0]-1)
    ar = (1+tr)**(1/nyr)-1 if nyr>0 else 0
    rf = 0.025
    sh = float((dr.mean()*252-rf)/(dr.std()*np.sqrt(252))) if dr.std()>0 else 0
    cm = rpv.cummax()
    md = float(((rpv-cm)/cm).min())
    wr = float((dr>0).mean())
    aw = float(dr[dr>0].mean()) if (dr>0).any() else 0
    al = abs(float(dr[dr<0].mean())) if (dr<0).any() else 1
    pl = aw/al if al>0 else 0
    monthly_ret = (1+dr).resample("ME").prod()-1
    return {
        "total_ret": round(tr*100,1), "annual_ret": round(ar*100,1),
        "sharpe": round(sh,2), "max_dd": round(md*100,1),
        "win_rate": round(wr*100,1), "pl_ratio": round(pl,2),
        "monthly": monthly_ret,
    }

# 6. 回测
print("\n>>> 全市场月频量化选股回测")
print("="*85)
print("  {:32s}  {:>7s}  {:>6s}  {:>5s}  {:>8s}  {:>5s}".format(
    "策略","总收益","年化","夏普","最大回撤","胜率"))
print("-"*85)

results = []
for top_n in [5, 10, 20, 30]:
    for alpha_col, label in [
        ("alpha_A","A.趋势动量"),("alpha_B","B.超跌反弹"),
        ("alpha_C","C.低波动"),("alpha_D","D.多因子")]:
        r = backtest(make_dict(alpha_col, top_n))
        r["label"] = f"{label} Top{top_n}"
        results.append(r)
        print("  {:35s}  {:+6.1f}%  {:+5.1f}%  {:5.2f}  {:+7.1f}%  {:5.1f}%".format(
            r["label"], r["total_ret"], r["annual_ret"], r["sharpe"], r["max_dd"], r["win_rate"]))
    print()

# 基准：等权
bench_dict = {}
for rd in reb_dates:
    dd = monthly[monthly["trade_date"] == pd.Timestamp(rd)]
    if len(dd) >= 50:
        bench_dict[pd.Timestamp(rd)] = set(dd.sample(min(50,len(dd)), random_state=42)["code"].tolist())
rbench = backtest(bench_dict)
rbench["label"] = "E.等权基准Top50"
results.append(rbench)
print("-"*85)
print("  {:35s}  {:+6.1f}%  {:+5.1f}%  {:5.2f}  {:+7.1f}%  {:5.1f}%".format(
    rbench["label"], rbench["total_ret"], rbench["annual_ret"],
    rbench["sharpe"], rbench["max_dd"], rbench["win_rate"]))
print("="*85)

best = max(results, key=lambda x: x["sharpe"])
print(f"\n★ 最优策略: {best['label']}  夏普={best['sharpe']}  总收益={best['total_ret']}%")

m = best["monthly"].sort_index()
if len(m) > 0:
    print("\n月度收益:")
    print("{:8s}  {:8s}".format("月份","收益"))
    print("-"*28)
    for i in range(len(m)):
        a = m.index[i]
        val = float(m.iloc[i])*100
        bar = "█"*int(max(0,val)) if val >= 0 else "░"*int(max(0,-val))
        print("  {}-{:02d}   {:+6.1f}%  {}".format(a.year, a.month, val, bar))

out = {}
for r in results:
    out[r["label"]] = {
        "total_ret": r["total_ret"], "annual_ret": r["annual_ret"],
        "sharpe": r["sharpe"], "max_dd": r["max_dd"],
        "win_rate": r["win_rate"], "pl_ratio": r["pl_ratio"],
        "monthly": {str(k): round(float(v)*100,1) for k,v in r["monthly"].items()},
    }
out["_best"] = best["label"]
with open("/Users/openclaw/tmp_fullmarket_result.json","w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n已保存 {len(out)} 个策略到 /Users/openclaw/tmp_fullmarket_result.json")
