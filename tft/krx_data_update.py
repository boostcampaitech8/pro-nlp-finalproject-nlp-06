import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import argparse
import os

import mojito
import FinanceDataReader as fdr
import ta

from dotenv import load_dotenv

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
TFT_PATH = os.path.join(PROJECT_ROOT, 'tft')
DATA_PATH = os.path.join(PROJECT_ROOT, 'tft/data')

# -------------------------------- #
# OHLCV 수집: mojito(한국투자증권 API) #
# -------------------------------- #
def fetch_adjusted_prices(
        broker,
        symbol,
        name,
        start_date: str,
        end_date: str
):
    """
    start_date/end_date: 'YYYYMMDD'
    mojito에서 adj_price=True로 일봉(수정주가) OHLCV 수집
    """
    symbol_norm = str(symbol).strip()
    if symbol_norm.isdigit():
        symbol_norm = symbol_norm.zfill(6)

    all_parsed_data = []
    current_end_date = end_date

    while True:
        try:
            resp = broker.fetch_ohlcv(
                symbol=symbol_norm,
                timeframe='D',
                start_day=start_date,
                end_day=current_end_date,
                adj_price=True
            )

            if resp.get('msg1') == '초당 거래건수를 초과하였습니다.':
                print(f"Rate limit hit for {name} ({symbol_norm}). Retrying...")
                time.sleep(1)
                continue

            if 'output2' not in resp or not resp['output2']:
                break

            daily_data = resp['output2']
            batch_data = []
            for item in daily_data:
                batch_data.append({
                    'Date': item['stck_bsop_date'],
                    'Code': symbol_norm,
                    'Name': name,
                    'Open': int(item['stck_oprc']),
                    'High': int(item['stck_hgpr']),
                    'Low': int(item['stck_lwpr']),
                    'Close': int(item['stck_clpr']),
                    'Volume': int(item['acml_vol'])
                })

            all_parsed_data.extend(batch_data)

            earliest_date_in_batch = min(item['stck_bsop_date'] for item in daily_data)
            if earliest_date_in_batch <= start_date:
                break

            dt_earliest = datetime.strptime(earliest_date_in_batch, "%Y%m%d")
            prev_day = dt_earliest - timedelta(days=1)
            current_end_date = prev_day.strftime("%Y%m%d")
            time.sleep(0.5)

        except Exception as e:
            print(f"Exception occurred for {name} ({symbol}): {e}")
            break

    if not all_parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_parsed_data)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= start_date]
    return df


# -------------- #
# (2) 기술지표 계산 #
# -------------- #
def calculate_features(df_group: pd.DataFrame):
    df = df_group.sort_values('Date').copy()

    # 0. 기본 데이터
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # 1. Log Returns
    df['Log_Return'] = np.log(close / close.shift(1))

    # 2. RSI (Relative Strength Index) - 9일, 14일
    df['RSI_9'] = ta.momentum.RSIIndicator(close=close, window=9).rsi()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # 3. ROC (Rate of Change) - 5일, 10일, 20일
    df['ROC_5'] = ta.momentum.ROCIndicator(close=close, window=5).roc()
    df['ROC_10'] = ta.momentum.ROCIndicator(close=close, window=10).roc()
    df['ROC_20'] = ta.momentum.ROCIndicator(close=close, window=20).roc()

    # 4. NATR (Normalized ATR) - 10일, 20일
    df['NATR_10'] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=10
    ).average_true_range() / close * 100
    df['NATR_20'] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=20
    ).average_true_range() / close * 100

    # 5. HV (Historical Volatility) - 10일, 20일
    df['HV_10'] = df['Log_Return'].rolling(window=10).std() * np.sqrt(252)
    df['HV_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)

    # 6. VR (Volume Ratio) - 14일, 20일
    diff = close.diff()
    vol_up = volume.where(diff > 0, 0)
    vol_down = volume.where(diff < 0, 0)
    vol_flat = volume.where(diff == 0, 0)

    for w in [14, 20]:
        sum_up = vol_up.rolling(window=w).sum()
        sum_down = vol_down.rolling(window=w).sum()
        sum_flat = vol_flat.rolling(window=w).sum()

        denominator = sum_down + (sum_flat / 2) + 1e-9
        numerator = sum_up + (sum_flat / 2)
        df[f'VR_{w}'] = (numerator / denominator) * 100

    # 7. Bollinger Band Width - 20일
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()

    # 8. MFI (Money Flow Index) - 14일
    df['MFI_14'] = ta.volume.MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14
    ).money_flow_index()

    # 9. OBV (On Balance Volume)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # 10. Disparity - 5, 20, 60일
    for w in [5, 20, 60]:
        ma = close.rolling(window=w).mean()
        df[f'Disparity_{w}'] = (close / ma) * 100

    # 11. MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()

    # 12. ADX (Average Directional Movement Index) - 14일
    df['ADX'] = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14
    ).adx()

    # 13. Amihud Illiquidity (유동성 지표)
    df['Dollar_Volume'] = (close * volume).replace(0, np.nan)
    illiq_daily = df['Log_Return'].abs() / (df['Dollar_Volume'] + 1e-12)

    df['Amihud_5'] = illiq_daily.rolling(window=5).mean()
    df['Amihud_20'] = illiq_daily.rolling(window=20).mean()
    df['Log_Amihud_5'] = np.log(df['Amihud_5'] + 1e-12)
    df['Log_Amihud_20'] = np.log(df['Amihud_20'] + 1e-12)

    return df


# ------------- #
# (3) Macro 수집 #
# ------------- #
def fetch_macro(start_date, end_date, macro_sources_dict):
    df_macro = pd.DataFrame()

    # 넉넉하게 한 달 전부터 수집
    fetch_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=30)
    ).strftime('%Y-%m-%d')

    for name, ticker in tqdm(macro_sources_dict.items()):
        try:
            df = fdr.DataReader(ticker, fetch_start, end_date)
            df = df[['Close']].rename(columns={'Close': name})

            if df_macro.empty:
                df_macro = df
            else:
                df_macro = df_macro.join(df, how='outer')

        except Exception as e:
            print(f"[ERROR] {name} ({ticker}) 처리 실패: {e}")

    df_macro = df_macro.sort_index().ffill()

    target_change_cols = ["KOSPI", "SP500", "USD_KRW", "WTI_Oil"]
    for col in target_change_cols:
        if col in df_macro.columns:
            df_macro[f"{col}_Change"] = df_macro[col].pct_change() * 100
            df_macro.drop(columns=[col], inplace=True)

    return df_macro


# ------------------------- #
# 최근 구간만 재계산하여 업데이트 #
# ------------------------- #
def update_features_recent_window(df: pd.DataFrame, lookback_rows: int = 200) -> pd.DataFrame:
    """
    각 Code별로 최근 lookback_rows 만큼만 calculate_features()로 재계산하고
    계산된 값(NA 아닌 값)만 원본에 update 합니다.
    """
    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('Date').copy()

        tail = g.tail(lookback_rows).copy()
        tail_calc = calculate_features(tail)

        g_idx = g.set_index('Date')
        tail_idx = tail_calc.set_index('Date')

        # update: other(tail_idx)의 NA가 아닌 값만 g_idx에 덮어씀
        g_idx.update(tail_idx)
        return g_idx.reset_index()

    return df.groupby('Code', group_keys=False).apply(_per_code).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default=os.path.join(DATA_PATH, "kospi200_merged_2021_2025_v2.csv"))
    parser.add_argument("--output_csv", type=str, default=os.path.join(DATA_PATH, "kospi200_merged_2021_2025_updated.csv"))
    parser.add_argument("--env_path", type=str, default=os.path.join(TFT_PATH, "api.env"))
    parser.add_argument("--lookback_rows", type=int, default=200) # 기술지표 재계산 구간(종목별 최근 N행)
    parser.add_argument("--sleep", type=float, default=0.3) # API 호출 간 sleep
    parser.add_argument("--no_bfill", action="store_true") # 데이터 누수 우려 시 bfill 끄기
    args = parser.parse_args()

    # 0) 기존 merged CSV 로드 (1) 대체
    if not os.path.exists(args.master_csv):
        raise FileNotFoundError(f"master_csv not found: {args.master_csv}")

    df_master = pd.read_csv(args.master_csv, dtype={"Code": str})
    df_master["Date"] = pd.to_datetime(df_master["Date"])
    df_master["Code"] = df_master["Code"].astype(str).str.zfill(6)

    # Code/Name/Sector를 기존 DF에서 확보
    base_info = df_master[["Code", "Name", "Sector"]].drop_duplicates("Code").copy()
    base_info["Code"] = base_info["Code"].astype(str).str.zfill(6)

    last_dt = df_master["Date"].max().date()

    # 서비스: "하루 이전" (어제)까지 업데이트
    today = datetime.now().date()
    end_dt = today - timedelta(days=1)

    if last_dt >= end_dt:
        print(f"[INFO] 이미 최신입니다. last_dt={last_dt}, end_dt={end_dt}")
        # 그래도 output_csv가 다르면 저장만 할 수도 있지만, 기본은 종료
        if args.output_csv != args.master_csv:
            df_master.to_csv(args.output_csv, index=False)
        return

    start_dt = last_dt + timedelta(days=1)
    start_ymd = start_dt.strftime("%Y%m%d")
    end_ymd = end_dt.strftime("%Y%m%d")

    print(f"[INFO] 업데이트 범위: {start_dt} ~ {end_dt}")

    # 1) env 로드 및 broker 생성
    is_loaded = load_dotenv(dotenv_path=args.env_path, verbose=True, override=True)
    print(f"[INFO] api.env file loaded: {is_loaded}")

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    acc_no = os.getenv("ACC_NO")

    # 증권사 객체 생성
    broker = mojito.KoreaInvestment(
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no
    )

    # 2) OHLCV 신규분만 수집
    all_new = []
    target_stocks = base_info[["Code", "Name"]].to_dict("records")
    print(f"[INFO] 대상 종목 수: {len(target_stocks)}")

    for stock in tqdm(target_stocks, desc="OHLCV 업데이트 중"):
        code = stock["Code"]
        name = stock["Name"]

        df_new = fetch_adjusted_prices(
            broker=broker,
            symbol=code,
            name=name,
            start_date=start_ymd,
            end_date=end_ymd
        )

        if not df_new.empty:
            all_new.append(df_new)

        time.sleep(args.sleep)

    if not all_new:
        print("[WARNING] 신규 수집 데이터가 없습니다.")
        return

    df_new_price = pd.concat(all_new, ignore_index=True)
    df_new_price["Date"] = pd.to_datetime(df_new_price["Date"], format="%Y%m%d")
    df_new_price["Code"] = df_new_price["Code"].astype(str)

    # Sector 붙이기(기존 DataFrame에서 가져옴)
    df_new_price = df_new_price.merge(base_info[["Code", "Sector"]], on="Code", how="left")

    # 중복 방지: 이미 존재하는 (Code, Date)는 제거
    existing_keys = set(zip(df_master["Code"].astype(str), df_master["Date"]))
    new_mask = ~df_new_price.apply(lambda r: (r["Code"], r["Date"]) in existing_keys, axis=1)
    df_new_price = df_new_price.loc[new_mask].copy()

    if df_new_price.empty:
        print("[INFO] (Code, Date) 기준으로 신규 행이 없습니다. (중복 업데이트 방지)")
        return

    # 3) 기존 DataFrame 컬럼에 맞춰서 신규 행 확장(지표/매크로 컬럼은 NaN으로 시작)
    df_new_price = df_new_price.reindex(columns=df_master.columns, fill_value=np.nan)

    # 4) append
    df_updated = pd.concat([df_master, df_new_price], ignore_index=True)
    df_updated = df_updated.sort_values(["Code", "Date"]).reset_index(drop=True)

    # 5) 기술지표 업데이트: 종목별 최근 N행만 재계산해서 반영
    print(f"[INFO] 기술지표 재계산(종목별 최근 {args.lookback_rows}행)")
    df_updated = update_features_recent_window(df_updated, lookback_rows=args.lookback_rows)

    # 6) macro 업데이트: 필요한 구간만 다시 받아서 덮어쓰기
    macro_sources = {
        "KOSPI": "KS11",
        "SP500": "US500",
        "USD_KRW": "USD/KRW",
        "WTI_Oil": "CL",
        "VIX": "VIX",
        "US_Treasury_10Y": "US10YT",
        "Dollar_Index": "USDX"
    }

    macro_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")  # 약간 여유(연속성/변동률 계산)
    macro_end = end_dt.strftime("%Y-%m-%d")

    print(f"[INFO] Macro 업데이트: {macro_start} ~ {macro_end}")
    df_macro = fetch_macro(macro_start, macro_end, macro_sources)
    df_macro = df_macro.reset_index().rename(columns={"index": "Date"})
    df_macro["Date"] = pd.to_datetime(df_macro["Date"])

    # merge 후 _new로 들어온 값을 기존 컬럼에 덮어쓰기
    df_updated = df_updated.merge(df_macro, on="Date", how="left", suffixes=("", "_new"))

    for col in df_macro.columns:
        if col == "Date":
            continue
        new_col = f"{col}_new"
        if new_col in df_updated.columns:
            mask = df_updated[new_col].notna()
            if col in df_updated.columns:
                df_updated.loc[mask, col] = df_updated.loc[mask, new_col]
            else:
                df_updated[col] = df_updated[new_col]
            df_updated.drop(columns=[new_col], inplace=True)

    # 7) 결측치 처리 + 캘린더 컬럼 갱신
    df_updated["Date"] = pd.to_datetime(df_updated["Date"])
    df_updated = df_updated.sort_values(["Code", "Date"]).reset_index(drop=True)

    if args.no_bfill:
        df_updated = df_updated.groupby("Code", group_keys=False).apply(lambda g: g.ffill())
    else:
        df_updated = df_updated.groupby("Code", group_keys=False).apply(lambda g: g.ffill().bfill())

    df_updated["Month"] = df_updated["Date"].dt.month.astype(str)
    df_updated["Day_of_Week"] = df_updated["Date"].dt.dayofweek.astype(str)
    df_updated["Day_of_Month"] = df_updated["Date"].dt.day.astype(str)

    df_updated = df_updated.sort_values(["Code", "Date"]).reset_index(drop=True)

    # 8) 저장
    df_updated.to_csv(args.output_csv, index=False)
    print(f"[INFO] 저장 완료: {args.output_csv}")
    print(f"[INFO] 최종 날짜: {df_updated['Date'].max().date()}, row={len(df_updated)}")

if __name__ == "__main__":
    main()
