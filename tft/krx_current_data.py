import os
import re
import pandas as pd
import FinanceDataReader as fdr
import mojito
from dotenv import load_dotenv

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
TFT_PATH = os.path.join(PROJECT_ROOT, 'tft')

def _normalize_name(name: str) -> str:
    """
    종목명 정규화
    """
    x = name.strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"\(.*?\)", "", x)
    return x

def _resolve_krx_code(
        code_or_name: str,
        krx_df: pd.DataFrame
) -> tuple[str, str | None]:
    """
    1. 입력이 6자리 숫자 -> 종목 코드
    2. else -> 종목명
    """
    x = code_or_name.strip()

    if re.fullmatch(r"\d{6}", x):
        return x, None
    
    nx = _normalize_name(x)
    candidates = krx_df.copy()
    candidates["_n"] = candidates["Name"].map(_normalize_name)

    exact = candidates[candidates["_n"] == nx]
    if len(exact) == 1:
        row = exact.iloc[0]
        return row["Code"], row["Name"]
    
    partial = candidates[candidates["_n"].str.contains(re.escape(nx), na=False)]
    if len(partial) == 1:
        row = partial.iloc[0]
        return row["Code"], row["Name"]
    
    if len(partial) > 1:
        # 애매할 때
        sample = partial[["Code", "Name"]].head(10).to_dict("records")
        raise ValueError(f"[ERROR] 종목명이 여러 개로 매칭됩니다: {x} -> {sample}")

    raise ValueError(f"종목명을 KRX 리스트에서 찾지 못했습니다: {x}")

def _make_kis_broker() -> mojito.KoreaInvestment:
    is_loaded = load_dotenv(dotenv_path=os.path.join(TFT_PATH, "api.env"), verbose=True, override=True)
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

    return broker

def krx_current_price(
        code_or_name: str
):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "tft", "data", "krx_list.csv"), dtype={"Code": str})
    df['Code'] = df['Code'].astype(str).str.zfill(6)

    code, _name = _resolve_krx_code(code_or_name, df)
    broker = _make_kis_broker()
    resp = broker.fetch_price(code)

    if str(resp.get("rt_cd")) != "0":
        raise RuntimeError(f"KIS 현재가 조회 실패: {resp.get('msg_cd')} {resp.get('msg1')}")

    price_str = resp["output"]["stck_prpr"]
    return int(price_str)

if __name__ == '__main__':
    print(krx_current_price("삼성전자"))