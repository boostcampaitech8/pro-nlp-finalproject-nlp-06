FEATURE_META: dict[str, dict[str, str]] = {
    # 식별/시간
    "Date": {"label": "날짜", "desc": "관측 시점(시간적 맥락)"},
    "Name": {"label": "종목명", "desc": "종목 식별 정보"},
    "Code": {"label": "종목코드", "desc": "종목 식별 정보"},
    "Month": {"label": "월", "desc": "월별 계절성/수급 패턴"},
    "Day_of_Week": {"label": "요일", "desc": "요일 효과(주간 패턴)"},
    "Day_of_Month": {"label": "월중 위치", "desc": "월초/월말 패턴"},
    "time_idx": {"label": "시간 인덱스", "desc": "시계열 추세의 위치 정보"},

    # 가격/거래
    "Open": {"label": "시가", "desc": "당일 시작 가격"},
    "High": {"label": "고가", "desc": "당일 최고 가격"},
    "Low": {"label": "저가", "desc": "당일 최저 가격"},
    "Close": {"label": "수정종가", "desc": "배당/분할 반영 종가"},
    "Volume": {"label": "거래량", "desc": "거래 활발도(수급 강도)"},
    "Sector": {"label": "업종", "desc": "업종 공통 움직임/테마 영향"},

    # 수익률/모멘텀
    "Log_Return": {"label": "로그수익률", "desc": "가격 변화율(로그)"},
    "Log_Return_scale": {"label": "로그수익률(스케일)", "desc": "정규화된 로그수익률"},
    "RSI9": {"label": "RSI(9)", "desc": "단기 과매수/과매도 강도"},
    "RSI14": {"label": "RSI(14)", "desc": "중기 과매수/과매도 강도"},
    "ROC_5": {"label": "ROC(5)", "desc": "5일 가격 변화 속도"},
    "ROC_10": {"label": "ROC(10)", "desc": "10일 가격 변화 속도"},
    "ROC_20": {"label": "ROC(20)", "desc": "20일 가격 변화 속도"},
    "MACD": {"label": "MACD", "desc": "추세 전환/모멘텀 신호"},
    "ADX": {"label": "ADX", "desc": "추세 강도(방향 아님)"},

    # 변동성
    "NATR_10": {"label": "NATR(10)", "desc": "10일 정규화 변동성"},
    "NATR_20": {"label": "NATR(20)", "desc": "20일 정규화 변동성"},
    "HV_10": {"label": "역사적 변동성(10)", "desc": "10일 실현 변동성"},
    "HV_20": {"label": "역사적 변동성(20)", "desc": "20일 실현 변동성"},
    "BB_Width": {"label": "볼린저 밴드 폭", "desc": "변동성 확대/축소 신호"},
    "VIX": {"label": "VIX", "desc": "미국 시장 공포/변동성 지표"},

    # 수급/유동성
    "VR_14": {"label": "VR(14)", "desc": "14일 거래량 비율(수급)"},
    "VR_20": {"label": "VR(20)", "desc": "20일 거래량 비율(수급)"},
    "MFI_14": {"label": "MFI(14)", "desc": "가격+거래량 기반 자금흐름"},
    "OBV": {"label": "OBV", "desc": "누적 거래량 기반 매집/분산"},
    "Disparity_5": {"label": "이격도(5일)", "desc": "5일 평균 대비 괴리"},
    "Disparity_20": {"label": "이격도(20일)", "desc": "20일 평균 대비 괴리"},
    "Disparity_60": {"label": "이격도(60일)", "desc": "60일 평균 대비 괴리"},
    "Log_Amihud_5": {"label": "Amihud 로그(5)", "desc": "5일 유동성(가격충격)"},
    "Log_Amihud_20": {"label": "Amihud 로그(20)", "desc": "20일 유동성(가격충격)"},

    # 거시/외부
    "US_Treasury_10Y": {"label": "미국 10년물 금리", "desc": "할인율/밸류에이션 환경"},
    "Dollar_Index": {"label": "달러 인덱스", "desc": "글로벌 달러 강세/약세"},
    "KOSPI_Change": {"label": "코스피 변화율", "desc": "국내 시장 전반 흐름"},
    "SP500_Change": {"label": "S&P500 변화율", "desc": "미국 시장 위험선호"},
    "USD_KRW_Change": {"label": "원달러 환율 변화율", "desc": "환율 민감도"},
    "WTI_Oil_Change": {"label": "WTI 유가 변화율", "desc": "원자재/인플레/비용 영향"},
}

STRATEGY_TEXT_MAP: dict[str, str] = {
    "highest_upside": "다음 거래일(1일) 기준으로 상승 여력이 가장 크다고 예측된 종목입니다.",
    "stable_positive": "향후 3거래일 동안 보수적 구간(q10)의 누적값이 가장 큰 종목으로, 상대적으로 안정적인 종목입니다."
}