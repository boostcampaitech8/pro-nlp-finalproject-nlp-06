import os
import torch
import numpy as np
import pandas as pd
import json
import argparse

from pandas.tseries.offsets import CustomBusinessDay
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

DATA_PATH = './data'
MODEL_PATH = './models'

def restore_price(base_price, log_returns):
    """
    base_price: Encoder의 마지막 Close
    log_returns: 예측일의 로그 수익률

    base_price를 기준으로 log_returns를 가격으로 다시 변환하여 반환
    """
    lr = np.asarray(log_returns, dtype=np.float64)
    cumulative_log_returns = np.cumsum(lr)
    restored_prices = float(base_price) * np.exp(cumulative_log_returns)
    restored_prices = np.nan_to_num(restored_prices, nan=0.0, posinf=0.0, neginf=0.0)
    
    return restored_prices.tolist()

def _to_2d_var_weights(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(0)
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        return t.mean(dim=1)
    return t.reshape(t.shape[0], -1)

def _to_3d_attention(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3:
        return t
    if t.ndim == 4:
        return t.mean(dim=1)
    raise ValueError(f"Unsupported Attention tensor shape: {tuple(t.shape)}")

def _get_variable_name_lists(
        model: TemporalFusionTransformer,
        params: dict
) -> dict[str, list[str]]:
    """
    model에서 변수 목록 뽑아오기
    """
    out: dict[str, list[str]] = {}

    for key in ["static_variables", "encoder_variables", "decoder_variables"]:
        names = getattr(model, key, None)
        if isinstance(names, (list, tuple)) and len(names) > 0:
            out[key] = list(names)
    
    if "static_variables" not in out:
        out["static_variables"] = list(params.get("static_categoricals", [])) \
        + list(params.get("static_reals", []))

    if "encoder_variables" not in out:
        out["encoder_variables"] = list(params.get("time_varying_categoricals_encoder", [])) \
        + list(params.get("time_varying_reals_encoder", []))

    if "decoder_variables" not in out:
        out["decoder_variables"] = list(params.get("time_varying_categoricals_decoder", [])) \
        + list(params.get("time_varying_reals_decoder", []))
    
    return out

def _top_k_items(scores: dict[str, float], k: int) -> list[dict]:
    """
    a
    """
    for key in list(scores.keys()):
        if scores[key] < 0:
            scores[key] = 0.0
    
    s = sum(scores.values())
    if s > 0:
        for key in list(scores.keys()):
            scores[key] /= s
    
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"name": n, "weight": round(float(w), 6)} for n, w in items]

def safe_interpret_output(model, out_dict):
    """
    interpret_output() 호출하기 전에 모델을 CPU로 옮기기 (CUDA 에러 피하기 위함)
    """
    out_safe = {}
    for k, v in out_dict.items():
        if torch.is_tensor(v):
            out_safe[k] = v.detach().cpu()
        else:
            out_safe[k] = v
    
    max_enc = int(model.hparams.max_encoder_length)
    max_dec = int(out_safe["decoder_variables"].size(1))

    out_safe["encoder_lengths"] = out_safe["encoder_lengths"].to(torch.long).clamp(min=0, max=max_enc)
    out_safe["decoder_lengths"] = out_safe["decoder_lengths"].to(torch.long).clamp(min=1, max=max_dec)

    out_safe = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out_safe.items()}
    model_cpu = model.to("cpu")

    with torch.no_grad():
        interpretation = model_cpu.interpret_output(
            out_safe,
            reduction="none",
            # attention_prediction_horizon=0,
            # attention_as_autocorrelation=False
        )
    
    return interpretation, max_enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=os.path.join(DATA_PATH, "kospi200_merged_2021_2025_updated.csv"))
    parser.add_argument("--holiday_csv", type=str, default=os.path.join(DATA_PATH, "krx_close.csv"))
    args = parser.parse_args()

    # 데이터 불러오기
    #df = pd.read_csv(os.path.join(DATA_PATH, "kospi200_merged_2021_2025.csv"), dtype={'Code': str})
    #df_holiday = pd.read_csv(os.path.join(DATA_PATH, "krx_close.csv"))
    df = pd.read_csv(args.data_csv, dtype={'Code': str})
    df_holiday = pd.read_csv(args.holiday_csv)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)

    # 카테고리 타입으로 변환
    categorical_cols = ['Code', 'Sector', 'Month', 'Day_of_Week', 'Day_of_Month']
    for col in categorical_cols:
        df[col] = df[col].astype(str).astype("category")

    krx_holidays = pd.to_datetime(df_holiday['일자 및 요일']).dt.date.tolist()
    krx_business_day = CustomBusinessDay(holidays=krx_holidays)

    as_of_date = df['Date'].max()
    future_dates = pd.date_range(
        start=as_of_date + krx_business_day,
        periods=5, # 5일
        freq=krx_business_day
    ).normalize()

    print(f"[INFO] as_of_date: {as_of_date.strftime('%Y-%m-%d')}")
    print(f"[INFO] 생성된 미래 5일: {future_dates.tolist()}")

    all_dates = pd.DatetimeIndex(df['Date'].unique()).append(pd.DatetimeIndex(future_dates)).unique().sort_values()
    date2idx = pd.Series(range(len(all_dates)), index=all_dates)
    df['time_idx'] = df['Date'].map(date2idx).astype("int64")

    # 마지막 60 거래일 Slicing
    df = df.sort_values(by=['Code', 'Date'], ascending=[True, True])
    df_60days = df.groupby('Code').tail(60).reset_index(drop=True)
    print(df_60days.tail())

    # 로그 수익률 -> 실제 주가로 변환하기 위함
    base_price_map = (
        df_60days.sort_values(['Code', 'Date'])
        .groupby('Code', observed=True)
        .tail(1)
        .set_index('Code')['Close']
        .to_dict()
    )

    static_info = df_60days.drop_duplicates('Code').set_index('Code')[['Name', 'Sector']].to_dict('index')

    print(f"생성된 미래 5일: {future_dates.tolist()}")

    # Future Data 생성
    codes = df_60days['Code'].unique()
    new_rows = []
    for code in codes:
        info = static_info.get(code)
        name_val = info['Name'] if info else np.nan
        sector_val = info['Sector'] if info else np.nan
        last_time_idx = df_60days[df_60days['Code'] == code].iloc[-1]['time_idx']

        for i, date in enumerate(future_dates):
            new_rows.append({
                'Date': date,
                'Code': code,
                # 'time_idx': int(last_time_idx + i + 1),
                'time_idx': int(date2idx.loc[date]),
                'Name': name_val,
                'Sector': sector_val,
                'Month': str(date.month),
                'Day_of_Week': str(date.dayofweek),
                'Day_of_Month': str(date.day)
            })

    df_future = pd.DataFrame(new_rows)

    df_final = pd.concat([df_60days, df_future], ignore_index=True)

    for col in categorical_cols:
        df_final[col] = df_final[col].astype(str).astype("category")

    df_final = df_final.sort_values(by=['Code', 'Date']).reset_index(drop=True)

    print(df_final.tail())

    # Device 설정
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # CPU로 강제
    print(f"Inference Device: {device}")

    # Model 불러오기
    best_tft = TemporalFusionTransformer.load_from_checkpoint(
        os.path.join(MODEL_PATH, "pu9snzwg/checkpoints/epoch=13-step=24668.ckpt")
    )
    best_tft.to(device)
    best_tft.eval()

    # Nan -> ffill 혹은 0으로 채우기
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_cols] = df_final[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_final[numeric_cols] = df_final.groupby('Code', observed=True)[numeric_cols].ffill()
    df_final[numeric_cols] = df_final[numeric_cols].fillna(0.0)

    print("[INFO] Future NaN values filled with 0.")
    print(df_final.tail())

    inference_ds = TimeSeriesDataSet.from_parameters(
        best_tft.dataset_parameters,
        df_final,
        predict=True,
        stop_randomization=True
    )

    print(f"[INFO] Inference Data Loaded: {len(inference_ds)}")

    # Data Loader 생성
    batch_size = 256
    num_workers = 8

    inference_dataloader = inference_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    # Prediction
    raw_predictions, x, index, decoder_length, y = best_tft.predict(
        inference_dataloader,
        mode="raw",
        return_x=True,
        return_y=True,
        return_index=True,
        return_decoder_lengths=True
    )

    # Top Variables / Attention
    interpretation, max_enc = safe_interpret_output(best_tft, raw_predictions)
    static_w = None
    enc_w = None
    dec_w = None
    att_w = None

    # Variable Selection Network
    if interpretation.get("static_variables", None) is not None:
        static_w = _to_2d_var_weights(interpretation["static_variables"]).detach().cpu().numpy()
    if interpretation.get("encoder_variables", None) is not None:
        enc_w = _to_2d_var_weights(interpretation["encoder_variables"]).detach().cpu().numpy()
    if interpretation.get("decoder_variables", None) is not None:
        dec_w = _to_2d_var_weights(interpretation["decoder_variables"]).detach().cpu().numpy()

    if interpretation.get("attention", None) is not None:
        att_t = interpretation["attention"]
        if att_t.ndim == 2:
            # (B, E) -> (B, 1, E)
            att_w = att_t.unsqueeze(1).detach().cpu().numpy()
        else:
            # (B, D, E)
            att_w = _to_3d_attention(att_t).detach().cpu().numpy()

    var_names = _get_variable_name_lists(best_tft, best_tft.dataset_parameters)
    static_names = var_names.get("static_variables", [])
    enc_names = var_names.get("encoder_variables", [])
    dec_names = var_names.get("decoder_variables", [])

    enc_time_idx_np = None
    enc_len_np = None
    if isinstance(x, dict) and "encoder_time_idx" in x:
        enc_time_idx_np = x["encoder_time_idx"].detach().cpu().numpy().astype(int)
    if isinstance(x, dict) and "encoder_lengths" in x:
        enc_len_np = x["encoder_lengths"].detach().cpu().numpy().astype(int)

    encoder_dates_map = (
        df_60days.sort_values(["Code", "Date"])
        .groupby("Code", observed=True)["Date"]
        .apply(lambda s: [pd.Timestamp(d).normalize() for d in s.tail(max_enc).tolist()])
        .to_dict()
    )

    idx2date = list(pd.DatetimeIndex(all_dates))
    top_vars_map: dict[int, list[dict]] = {}
    top_attn_map: dict[int, list[dict]] = {}

    n_samples = len(index)

    for i in range(n_samples):
        # Prediction에 대하여 Top-5 Variables 뽑기
        scores: dict[str, float] = {}
        if static_w is not None:
            w = static_w[i]
            for j in range(min(len(static_names), len(w))):
                scores[str(static_names[j])] = scores.get(str(static_names[j]), 0.0) + float(w[j])
        
        if enc_w is not None:
            w = enc_w[i]
            for j in range(min(len(enc_names), len(w))):
                scores[str(enc_names[j])] = scores.get(str(enc_names[j]), 0.0) + float(w[j])
        
        if dec_w is not None:
            w = dec_w[i]
            for j in range(min(len(dec_names), len(w))):
                scores[str(dec_names[j])] = scores.get(str(dec_names[j]), 0.0) + float(w[j])
        
        top_vars_map[i] = _top_k_items(scores, k=5)

        # Prediction에 대하여 Top-3 Attention Points 뽑기
        att_points: list[dict] = []
        if att_w is not None:
            att_vec = att_w[i].mean(axis=0).astype(float) # (D, E) -> Decoder 평균 -> (E,)
            E = int(att_vec.shape[0])
            s = float(att_vec.sum())

            if s > 0:
                att_vec = att_vec /s
            
            code_i = str(index.iloc[i]["Code"])
            dates = encoder_dates_map.get(code_i, [])

            if len(dates) < E:
                dates = [None] * (E - len(dates)) + dates
            else:
                dates = dates[-E:]
            
            top_pos = np.argsort(att_vec)[-3:][::-1]
            for p in top_pos:
                d = dates[int(p)]
                lag = int(p - E)
                att_points.append({
                    "date": d.strftime("%Y-%m-%d") if d is not None else None,
                    "lag": lag,
                    "weight": round(float(att_vec[int(p)]), 6)
                })
        
        top_attn_map[i] = att_points

    y_pred_lower = raw_predictions["prediction"][..., 0]
    y_pred_median = raw_predictions["prediction"][..., 1]
    y_pred_upper = raw_predictions["prediction"][..., 2]

    # Final Result 저장 (199개 종목)
    final_results = []
    preds_median_np = y_pred_median.cpu().numpy()
    preds_lower_np = y_pred_lower.cpu().numpy()
    preds_upper_np = y_pred_upper.cpu().numpy()

    date_strings = [d.strftime("%Y-%m-%d") for d in future_dates]

    print("[INFO] Formatting results to JSON...")

    # 종목 추천
    # (1) 5일 누적 예상 수익률이 최대
    best_up = None
    # (2) 5일 누적 예상 수익률 > 0인 동시에 불확실성이 가장 작은 종목
    best_stable = None

    for idx, row in index.iterrows():
        code = row['Code']
        stock_name = static_info.get(code, {}).get('Name', 'Unknown')

        base_close = float(base_price_map.get(code, 0.0))

        lr_median = preds_median_np[idx]
        lr_lower = preds_lower_np[idx]
        lr_upper = preds_upper_np[idx]

        # 가격 복원
        price_median = restore_price(base_close, lr_median)
        price_lower = restore_price(base_close, lr_lower)
        price_upper = restore_price(base_close, lr_upper)

        # 수익률 (%)
        pct_median = (np.expm1(lr_median) * 100.0).astype(np.float64)
        pct_lower = (np.expm1(lr_lower) * 100.0).astype(np.float64)
        pct_upper = (np.expm1(lr_upper) * 100.0).astype(np.float64)

        # 5일 누적 예상 수익률 (median, lower, upper)
        cum_lr_median = float(np.sum(lr_median))
        cum_ret_median = float(np.expm1(cum_lr_median))

        cum_lr_lower = float(np.sum(lr_lower))
        cum_ret_lower = float(np.expm1(cum_lr_lower))

        cum_lr_upper = float(np.sum(lr_upper))
        cum_ret_upper = float(np.expm1(cum_lr_upper))

        risk_spread = float(cum_ret_upper - cum_ret_lower)
        if best_up is None or cum_ret_median > best_up[0]:
            best_up = (cum_ret_median, str(code).zfill(6), stock_name)

        if cum_ret_median > 0:
            if best_stable is None:
                best_stable = (risk_spread, cum_ret_median, str(code).zfill(6), stock_name)
            else:
                if (risk_spread < best_stable[0]) or (risk_spread == best_stable[0] and cum_ret_median > best_stable[1]):
                    best_stable = (risk_spread, cum_ret_median, str(code).zfill(6), stock_name)

        daily_forecasts = []
        for i, date_str in enumerate(date_strings):
            daily_forecasts.append({
                "date": date_str,
                # 로그 수익률
                "log_return": round(float(lr_median[i]), 6),
                "log_return_lower": round(float(lr_lower[i]), 6),
                "log_return_upper": round(float(lr_upper[i]), 6),
                # 전일 대비 변화율 (%)
                "pct_change": round(float(pct_median[i]), 4),
                "pct_change_lower": round(float(pct_lower[i]), 4),
                "pct_change_upper": round(float(pct_upper[i]), 4),
                # 가격
                "price": int(round(price_median[i])),
                "price_lower": int(round(price_lower[i])),
                "price_upper": int(round(price_upper[i]))
            })
        
        stock_result = {
            "code": str(code).zfill(6),
            "name": stock_name,
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "base_close": base_close,
            "forecasts": daily_forecasts,
            "top_variables": top_vars_map.get(int(idx), []),
            "top_attention": top_attn_map.get(int(idx), [])
        }
        final_results.append(stock_result)
    
    recommendations = {}
    if best_up is not None:
        recommendations["highest_upside"] = {
            "code": best_up[1],
            "name": best_up[2],
            "horizon_days": 5,
            "expected_return": round(float(best_up[0]) * 100.0, 4),
            "metric": "5d cumulative median return"
        }
    
    if best_stable is not None:
        recommendations["stable_positive"] = {
            "code": best_stable[2],
            "name": best_stable[3],
            "horizon_days": 5,
            "expected_return": round(float(best_stable[1]) * 100.0, 4),
            "risk_spread": round(float(best_stable[0]) * 100.0, 4),
            "metric": "min(5d cumulative upper-lower spread) among positive-return"
        }
    
    payload = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "horizon_days": 5,
        "recommendations": recommendations,
        "results": final_results
    }

    output_file = "inference_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    
    print(f"[INFO] Prediction Complete")
    print(json.dumps(payload["recommendations"], ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()