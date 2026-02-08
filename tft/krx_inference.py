import os
import torch
import numpy as np
import pandas as pd
import json
import argparse

from tft_loss import HorizonWeightedQuantileLoss
from pandas.tseries.offsets import CustomBusinessDay
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from pathlib import Path
from dotenv import load_dotenv
from langchain_naver import ChatClovaX

from src.Agent import app as agent_app, AgentState

load_dotenv()

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
DATA_PATH = os.path.join(PROJECT_ROOT, 'tft/data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'tft/model')
CLOVA_STUDIO_API_KEY = os.getenv("CLOVA_STUDIO_API_KEY", "")

answer_llm = ChatClovaX(
    model="HCX-007",
    api_key=CLOVA_STUDIO_API_KEY,
    max_tokens=32000,
    temperature=0.1,
    seed=42,
    timeout=30,
    max_retries=50
)

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

def _format_top_vars(top_vars: list[dict], k: int = 3) -> str:
    items = []
    for v in (top_vars or [])[:k]:
        name = str(v.get("name") or "").strip()
        if not name:
            continue
        try:
            w = float(v.get("weight", 0.0)) * 100.0
        except (TypeError, ValueError):
            w = 0.0
        items.append(f"{name}({w:.1f}%)")
    return ", ".join(items) if items else "없음"

def _format_top_attn(top_attn: list[dict], k: int = 3) -> str:
    items = []
    for a in (top_attn or [])[:k]:
        d = str(a.get("date") or "").strip()
        if not d:
            continue
        try:
            w = float(a.get("weight", 0.0)) * 100.0
        except (TypeError, ValueError):
            w = 0.0
        items.append(f"{d}({w:.1f}%)")
    return ", ".join(items) if items else "없음"

def _generate_reason_with_hcx(
        strategy_key: str,
        rec_data: dict,
        result_data: dict
) -> str:
    metric = rec_data.get("metric")
    expected = rec_data.get("expected_return")
    top_vars_text = _format_top_vars(result_data.get("top_variables"), k=3)
    top_attn_text = _format_top_attn(result_data.get("top_attention"), k=3)

    if strategy_key == "highest_upside":
        strategy_name = "공격적 수익 추구 (Highest Upside)"
        strategy_desc = "향후 1거래일 기준, 중앙값(q50) 상승폭이 가장 높게 예측된 종목입니다."
        focus_point = "상승 여력(Upside Potential)과 단기 변동성"
    elif strategy_key == "stable_positive":
        strategy_name = "안정적 우상향(Stable Positive)"
        strategy_desc = "향후 3거래일 동안 보수적 시나리오(q10)에서도 누적 수익이 높을 것으로 예측된 종목입니다."
        focus_point = "하방 경직성(Risk Defense)과 추세의 안정성"
    else:
        strategy_name = strategy_key
        strategy_desc = "기타 전략"
        focus_point = "일반적인 예측 추이"

    # strategy_label_map = {
    #     "highest_upside": "다음 1거래일 상승 여지가 가장 큰 종목",
    #     "stable_positive": "보수적 시나리오(q10) 기준 3거래일 누적 기대가 큰 종목",
    # }
    # strategy_label = strategy_label_map.get(strategy_key, strategy_key)

    fc_rows = []
    for r in (result_data.get("forecasts") or [])[:3]:
        fc_rows.append(
            f"{r.get('date')}: [하방(q10)] {r.get('pct_change_lower')}%, "
            f"[기준(q50)] {r.get('pct_change')}%, [상방(q90)] {r.get('pct_change_upper')}%"
        )
    
    fc_text = " | ".join(fc_rows) if fc_rows else "예측 데이터 없음"

    prompt = (
        f"당신은 딥러닝 퀀트 모델(TFT)의 예측 결과를 해석하여 투자자에게 브리핑하는 'AI 금융 분석가'입니다.\n"
        f"상대는 '초보 투자자'로, 주식 초보도 이해하기 쉽게 이야기해야 합니다.\n"
        f"아래의 [입력 데이터]를 바탕으로 [작성 포맷]에 맞춰 보고서를 작성해 주세요.\n\n"
        f"[입력 데이터]\n"
        f"1. 추천 전략: {strategy_name}\n"
        f"   - 전략 정의: {strategy_desc}\n"
        f"   - 해석 초점: {focus_point}\n"
        f"2. 선정 기준값(Metric): {metric}\n"
        f"3. 예상 수익률(1거래일 이후): {expected}\n"
        f"4. 모델이 중요하게 본 변수(Top Variables): {top_vars_text}\n"
        f"   (의미: 예측을 수행할 때 모델이 가장 민감하게 반응한 변수)\n"
        f"5. 모델이 주목한 과거 시점(Attention): {top_attn_text}\n"
        f"   (의미: 현재의 가격 변동을 예측하기 위해 참고한 과거의 유사 패턴 발생 시점)\n"
        f"6. 향후 3일 시나리오 예측값:\n{fc_text}\n\n"
        
        f"[작성 포맷(엄수)]\n"
        f"다음 3가지 항목으로 구분하여 작성하세요. 각 항목에서 '###' 헤더는 사용하지 않으며, 포맷을 엄수하세요.\n"
        f"**1. 선정 배경** \n"
        f"(이 종목이 왜 '{strategy_name}' 전략에 포착되었는지 전략 정의와 연결하여 1문장으로 설명)\n\n"
        f"**2. AI 모델의 분석 근거** \n"
        f"('중요 변수'와 '주목한 과거 시점'을 연결하여 서술형으로 작성. 단, 인과관게로 단정 짓지 말고 '모델은 ~변수의 움직임과 ~시점의 패턴을 중요하게 참고했습니다'와 같은 형식을 사용할 것.)\n\n"
        f"**3. 투자 리스크** \n"
        f"(q10과 q90의 격차(불확실성) 또는 예측의 한계를 언급하며 보수적 접근을 권고하는 1문장 경고문)\n\n"

        f"[제약 사항]\n"
        f"- 말투: 전문적이나 친절하게, '해요'체를 사용 (예: 분석됩니다, 참고했습니다).\n"
        f"- 환각 방지: 입력된 데이터 외의 외부 뉴스나 사실을 절대 지어내지 말 것.\n"
        f"- 수치 인용: 구체적인 예측 수치(%)를 포함하여 신뢰도를 높일 것.\n"
        f"- 마크 다운: 모든 마크다운 강조(`**`) 기호 안팎에는 적절한 공백을 유지할 것."
    )

    try:
        ai_msg = answer_llm.invoke([
            (
                "system",
                "당신은 퀀트 모델이 추천한 종목의 선정 이유를 설명하는 '금융 전문가'입니다."
                "사용자가 제공한 데이터만을 기반으로 TFT 모델의 내부 동작을 설명해야 합니다."
            ),
            ("human", prompt),
        ])
        text = (getattr(ai_msg, "content", "") or "").strip()
        if text:
            return text
    except Exception as e:
        print(f"Exception: {e}")

    return (
        f"이 종목은 '{metric}' 기준으로 선정되었습니다.\n"
        f"모델이 중요하게 본 변수는 {top_vars_text}, 주요 참고 시점은 {top_attn_text}입니다.\n"
        "예측값은 오차가 있을 수 있으므로 보수적으로 해석해야 합니다.\n"
    )

def _get_recent_news(name: str) -> str | None:
    query = f"{name}에 대한 최근 뉴스를 보여주세요."
    # state: AgentState = {
    #     "query": query,           # ✅ LLM용 (대화 이력 포함)
    #     "user_input": query, # ✅ 벡터 검색용 (순수 입력만)
    #     "category": "rag",
    #     "rag_categories": ["news"],          # ✅ 추가 필요
    #     "results": [],                 # ✅ 추가 필요
    #     "debate_history": [],          # ✅ 빈 리스트로 초기화
    #     "debate_count": 0,
    #     "response": "",
    #     "target_companies": [],        # ✅ 추가 필요
    #     "tft_data": [],                # ✅ 추가 필요
    # }

    state: AgentState = {
        "query": query,           
        "user_input": query,
        "history": "", 
        "category": "rag",
        "rag_categories": ["news"], 
        "final_query" : "",
        "final_user_input" : "",
        "relevance": "",         
        "results": [],                 
        "debate_history": [],          
        "debate_count": 0,
        "response": "",
        "target_companies": [],        
        "tft_data": [],                
    }
    try:
        result = agent_app.invoke(state)

        answer = result.get("response", "응답을 생성할 수 없습니다.")
        category = result.get("category", "unknown")
        sub_category = result.get("sub_category", "")

        print(f"Answer: {answer}\n")
        print(f"Category: {category}\n")
        print(f"Sub Category: {sub_category}\n")

        return answer
    except Exception as e:
        print(f"Exception: {e}")
        return None

def _final_reason_with_hcx(
        tft_text: str,
        news_text: str | None
) -> str:
    if news_text is None:
        return tft_text
    
    news = (news_text or "").strip()
    if (not news) or ("2023년" in news):
        return tft_text.strip()
    
    if len(news) > 3000:
        news = news[:3000] + "\n...(중략)"
    
    concat_text = (
        f"[TFT 기반 선정 이유]\n{tft_text.strip()}\n\n"
        f"[최근 뉴스 정보]\n{news}"
    )

    prompt = (
        "당신은 초보 투자자를 위한 금융 브리핑 도우미입니다.\n"
        "아래 [결합 입력]은 'TFT 모델 설명'과 '최근 뉴스'를 합친 텍스트입니다.\n"
        "두 정보를 함께 반영해 최종 설명을 작성하세요.\n\n"
        "[출력 규칙]\n"
        "1) 2개 섹션으로 작성:\n"
        "   **1. 선정 배경과 AI 모델의 분석 근거**\n"
        "   **2. 최근 뉴스 반영 관찰 포인트**\n"
        "2) 모델(TFT) 기반 설명과 뉴스 기반 설명을 명확하게 구분해서 서술하세요.\n"
        "3) 입력에 없는 사실은 절대 추가하지 마세요.\n"
        "4) 초보자를 대상으로 작성해야 합니다. 특히 '최근 뉴스 반영 관찰 포인트'에서는 어려운 용어를 쉽게 풀어서 설명하세요."
        "5) 과장/확정 표현 금지. 마지막 문장은 리스크/불확실성 경고로 마무리하세요.\n\n"
        f"[결합 입력]\n{concat_text}"
    )

    try:
        ai_msg = answer_llm.invoke([
            (
                "system",
                "당신은 퀀트 모델이 추천한 종목의 선정 이유를 설명하는 '금융 전문가'입니다."
                "사용자가 제공한 데이터만을 기반으로 TFT 모델의 내부 동작을 설명해야 합니다."
            ),
            ("human", prompt),
        ])
        text = (getattr(ai_msg, "content", "") or "").strip()
        if text:
            return text
    except Exception as e:
        print(f"Exception: {e}")

    return tft_text

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
        )
    
    return interpretation, max_enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=os.path.join(DATA_PATH, "kospi200_merged_2021_2025_updated.csv"))
    parser.add_argument("--holiday_csv", type=str, default=os.path.join(DATA_PATH, "krx_close.csv"))
    args = parser.parse_args()

    # 데이터 불러오기
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
        periods=3, # 3일
        freq=krx_business_day
    ).normalize()

    print(f"[INFO] as_of_date: {as_of_date.strftime('%Y-%m-%d')}")
    print(f"[INFO] 생성된 미래 3일: {future_dates.tolist()}")

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

    print(f"생성된 미래 3일: {future_dates.tolist()}")

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
        os.path.join(MODEL_PATH, "epoch=16-step=12580.ckpt")
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
    batch_size = 64
    num_workers = 0

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
    # (1) 다음 1거래일 상승률(median)이 가장 높은 종목
    best_up = None
    # (2) 3일 누적 q10(lower) 수익률이 가장 높은 종목
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

        # (1) 1일 뒤 가장 상승률이 높은 종목
        code_str = str(code).zfill(6)
        h = 3
        lr_m_h = np.asarray(lr_median[:h], dtype=np.float64)
        lr_l_h = np.asarray(lr_lower[:h], dtype=np.float64)
        lr_u_h = np.asarray(lr_upper[:h], dtype=np.float64)

        next_day_ret = float(np.expm1(float(lr_m_h[0])))
        next_day_spread = float(np.expm1(float(lr_u_h[0])) - np.expm1(float(lr_l_h[0])))

        if (best_up is None) or (next_day_ret > best_up[0]):
            best_up = (next_day_ret, next_day_spread, code_str, stock_name)
        
        # (2) 3일 누적 q10이 가장 높은 종목
        q10_cum_ret_3d = float(np.expm1(float(np.sum(lr_l_h))))
        med_cum_ret_3d = float(np.expm1(float(np.sum(lr_m_h))))
        up_cum_ret_3d = float(np.expm1(float(np.sum(lr_u_h))))
        spread_3d = float(up_cum_ret_3d - q10_cum_ret_3d)

        if (best_stable is None) or (q10_cum_ret_3d > best_stable[0]) \
            or (q10_cum_ret_3d == best_stable[0] and med_cum_ret_3d > best_stable[1]):
            best_stable = (q10_cum_ret_3d, med_cum_ret_3d, next_day_ret, spread_3d, code_str, stock_name)

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
            "code": best_up[2],
            "name": best_up[3],
            "horizon_days": 3,
            "expected_return": round(float(best_up[0]) * 100.0, 4),
            "risk_spread": round(float(best_up[1]) * 100.0, 4),
            "metric": "highest next-day median return",
            "reason": None
        }
    else:
        recommendations["highest_upside"] = {
            "code": None,
            "name": None,
            "horizon_days": 3,
            "expected_return": None,
            "metric": "highest next-day median return",
            "reason": None
        }

    
    if best_stable is not None:
        recommendations["stable_positive"] = {
            "code": best_stable[4],
            "name": best_stable[5],
            "horizon_days": 3,
            "expected_return": round(float(best_stable[2]) * 100.0, 4),
            "risk_spread": round(float(best_stable[3]) * 100.0, 4),
            "metric": "maximize 3d cumulative q10 return",
            "reason": None
        }
    else:
        recommendations["stable_positive"] = {
            "code": None,
            "name": None,
            "horizon_days": 3,
            "expected_return": None,
            "risk_spread": None,
            "metric": "maximize 3d cumulative q10 return",
            "reason": None
        }
    
    # 추천 이유 채우기 (HyperClovaX)
    results_by_code = {
        str(item.get("code") or "").strip(): item
        for item in final_results
    }

    for rec_key in ("highest_upside", "stable_positive"):
        rec = recommendations.get(rec_key)
        if not rec:
            continue
            
        code = str(rec.get("code") or "").strip()
        if not code:
            rec["reason"] = None
            continue
            
        result_data = results_by_code.get(code, {})
        tft_reason = _generate_reason_with_hcx(
            strategy_key=rec_key,
            rec_data=rec,
            result_data=result_data
        )

        news_reason = _get_recent_news(rec.get("name") or result_data.get("name") or "")
        rec["reason"] = _final_reason_with_hcx(tft_reason, news_reason)

    payload = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "horizon_days": 3,
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