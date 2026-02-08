"""
통합 네이버 금융 리포트 수집 및 ChromaDB 저장 파이프라인
- 시황/종목/경제/산업 리포트 자동 수집
- vLLM 요약
- ChromaDB 저장
"""

import requests
from bs4 import BeautifulSoup
import fitz
from openai import OpenAI
import time
import json
from datetime import datetime
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Optional


class NaverReportPipeline:
    """네이버 금융 리포트 통합 파이프라인"""
    
    # 카테고리별 URL 및 DB 경로 매핑
    CATEGORIES = {
        "market": {
            "name": "시황",
            "url": "https://finance.naver.com/research/market_info_list.naver",
            "db_name": "MarketConditions_report_chroma_db",
            "is_industry": False,
        },
        "company": {
            "name": "종목",
            "url": "https://finance.naver.com/research/company_list.naver",
            "db_name": "Company_report_chroma_db",
            "is_industry": False,
        },
        "economy": {
            "name": "경제",
            "url": "https://finance.naver.com/research/economy_list.naver",
            "db_name": "Economy_report_chroma_db",
            "is_industry": False,
        },
        "industry": {
            "name": "산업",
            "url": "https://finance.naver.com/research/industry_list.naver",
            "db_name": "Industry_report_chroma_db",
            "is_industry": True,  # 산업 리포트는 레이아웃이 다름
        },
    }
    
    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8001/v1",
        vllm_api_key: str = "vllm-key",
        vllm_model: str = "skt/A.X-4.0-Light",
        embedding_model: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        chroma_base_dir: str = "./Chroma_db",
        max_text_length: int = 8000,
        summary_max_tokens: int = 1024,
        temperature: float = 0.3,
        debug: bool = True,
    ):
        """
        Args:
            vllm_base_url: vLLM 서버 URL
            vllm_api_key: API 키
            vllm_model: 요약에 사용할 모델
            embedding_model: 임베딩 모델
            chroma_base_dir: ChromaDB 저장 루트 경로
            max_text_length: PDF 텍스트 자르기 길이 (기본 8000자)
            summary_max_tokens: 요약 최대 토큰 수
            temperature: LLM temperature
            debug: 디버그 로그 출력 여부
        """
        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.embedding_model = embedding_model
        self.chroma_base_dir = chroma_base_dir
        self.max_text_length = max_text_length
        self.summary_max_tokens = summary_max_tokens
        self.temperature = temperature
        self.debug = debug
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key
        )
        
        # 임베딩 모델 초기화 (lazy loading)
        self._embeddings = None
        
        # HTTP 헤더
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        if self.debug:
            print(f"[NaverReportPipeline] 초기화 완료")
            print(f"  - vLLM URL: {self.vllm_base_url}")
            print(f"  - 요약 모델: {self.vllm_model}")
            print(f"  - 임베딩: {self.embedding_model}")
            print(f"  - 텍스트 자르기: {self.max_text_length}자")
            print(f"  - ChromaDB: {self.chroma_base_dir}")
    
    def get_embeddings(self):
        """임베딩 모델 로드 (싱글톤)"""
        if self._embeddings is None:
            if self.debug:
                print(f"[INFO] 임베딩 모델 로드: {self.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model
            )
        return self._embeddings
    
    def get_report_list(self, category: str) -> List[Dict]:
        """
        특정 카테고리의 오늘 날짜 리포트 목록 가져오기
        
        Args:
            category: "market", "company", "economy", "industry"
        """
        if category not in self.CATEGORIES:
            print(f"[ERROR] 잘못된 카테고리: {category}")
            return []
        
        cat_info = self.CATEGORIES[category]
        url = cat_info["url"]
        is_industry = cat_info["is_industry"]
        
        today_str = datetime.now().strftime("%y.%m.%d")
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📅 [{cat_info['name']}] {today_str} 리포트 수집 시작")
            print(f"{'='*60}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.encoding = 'euc-kr'
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.select('table.type_1 tr')
            reports = []
            
            for row in rows:
                tds = row.select('td')
                if len(tds) < 4:
                    continue
                
                # 날짜 찾기 (YY.MM.DD 형식)
                report_date = ""
                for td in tds:
                    text = td.get_text(strip=True)
                    if len(text) == 8 and text.count('.') == 2:
                        report_date = text
                        break
                
                if report_date != today_str:
                    continue
                
                # 제목 및 PDF 링크 추출
                title_idx = 1 if is_industry else 0
                title_tag = tds[title_idx].select_one('a')
                pdf_link_tag = row.select_one('a[href*=".pdf"]')
                
                if pdf_link_tag and title_tag:
                    reports.append({
                        'title': title_tag.get_text(strip=True),
                        'pdf_url': pdf_link_tag['href'],
                        'date': report_date,
                        'category': category
                    })
            
            if self.debug:
                print(f"✅ {len(reports)}개 리포트 발견")
            
            return reports
            
        except Exception as e:
            print(f"[ERROR] {cat_info['name']} 리포트 수집 실패: {e}")
            return []
    
    def extract_text_from_pdf(self, pdf_url: str) -> Optional[str]:
        """PDF URL에서 텍스트 추출"""
        try:
            response = requests.get(pdf_url, headers=self.headers, timeout=30)
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = "".join([page.get_text() for page in doc])
                return text[:self.max_text_length]  # 설정된 길이로 자르기
        except Exception as e:
            if self.debug:
                print(f"[ERROR] PDF 추출 실패: {e}")
            return None
    
    def summarize_text(self, text: str, title: str = "", date: str = "") -> str:
        """
        텍스트 요약 (날짜와 제목 포함)
        
        Args:
            text: 요약할 텍스트
            title: 리포트 제목
            date: 리포트 날짜 (YY.MM.DD)
        
        Returns:
            "[날짜] 제목\n요약내용" 형식
        """
        if not text or len(text) < 100:
            return "본문 내용이 너무 적어 요약할 수 없습니다."
        
        try:
            response = self.client.chat.completions.create(
                model=self.vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 금융 분석 전문가입니다. 다음 리포트 내용을 핵심 위주로 요약하세요."
                    },
                    {
                        "role": "user",
                        "content": f"리포트 내용:\n{text}"
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.summary_max_tokens
            )
            
            summary_content = response.choices[0].message.content
            
            # 날짜와 제목을 요약 앞에 추가
            header = ""
            if date:
                # YY.MM.DD -> 20YY-MM-DD 형식으로 변환 (더 읽기 쉽게)
                try:
                    year, month, day = date.split('.')
                    full_date = f"20{year}-{month}-{day}"
                    header += f"[{full_date}]"
                except:
                    header += f"[{date}]"
            
            if title:
                header += f" {title}"
            
            if header:
                return f"{header}\n\n{summary_content}"
            else:
                return summary_content
            
        except Exception as e:
            return f"요약 중 오류: {e}"
    
    def save_to_chromadb(
        self,
        summaries: List[Dict],
        category: str
    ) -> Optional[Chroma]:
        """요약 결과를 ChromaDB에 저장"""
        if not summaries:
            print(f"[WARN] {category}: 저장할 요약본 없음")
            return None
        
        cat_info = self.CATEGORIES[category]
        db_path = os.path.join(self.chroma_base_dir, cat_info["db_name"])
        
        # Document 생성
        docs = []
        for item in summaries:
            content = item.get("summary", "")
            if not content or "오류" in content:
                continue
            
            metadata = {
                "title": item.get("title", "제목 없음"),
                "date": item.get("date", ""),
                "source": item.get("pdf_url", ""),
                "category": category
            }
            docs.append(Document(page_content=content, metadata=metadata))
        
        if not docs:
            print(f"[WARN] {category}: 유효한 요약본 없음")
            return None
        
        if self.debug:
            print(f"\n💾 [{cat_info['name']}] ChromaDB 저장 중...")
            print(f"   경로: {db_path}")
            print(f"   문서 수: {len(docs)}")
        
        try:
            embeddings = self.get_embeddings()
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=db_path
            )
            
            if self.debug:
                print(f"✅ [{cat_info['name']}] 저장 완료")
            
            return vectorstore
            
        except Exception as e:
            print(f"[ERROR] {cat_info['name']} DB 저장 실패: {e}")
            return None
    
    def process_category(
        self,
        category: str,
        save_json: bool = True
    ) -> Optional[Chroma]:
        """
        특정 카테고리 전체 파이프라인 실행
        
        Args:
            category: "market", "company", "economy", "industry"
            save_json: JSON 파일로도 저장할지 여부
        
        Returns:
            ChromaDB vectorstore 또는 None
        """
        cat_info = self.CATEGORIES[category]
        
        # 1. 리포트 목록 가져오기
        report_list = self.get_report_list(category)
        
        if not report_list:
            print(f"[INFO] {cat_info['name']}: 오늘 날짜 리포트 없음")
            return None
        
        # 2. PDF 추출 및 요약
        summaries = []
        for i, report in enumerate(report_list, 1):
            if self.debug:
                print(f"\n[{i}/{len(report_list)}] {report['title'][:50]}...")
            
            # PDF 텍스트 추출
            full_text = self.extract_text_from_pdf(report['pdf_url'])
            
            if not full_text:
                continue
            
            # 요약 (제목과 날짜 포함)
            summary = self.summarize_text(
                full_text,
                title=report['title'],
                date=report['date']
            )
            
            summaries.append({
                "title": report['title'],
                "pdf_url": report['pdf_url'],
                "date": report['date'],
                "summary": summary,
                "category": category
            })
            
            if self.debug:
                print(f"   ✅ 요약 완료: {summary[:80]}...")
            
            time.sleep(0.5)  # 서버 부하 방지
        
        # 3. JSON 저장 (선택)
        if save_json and summaries:
            json_filename = f"{category}_summaries_{datetime.now().strftime('%Y%m%d')}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
            
            if self.debug:
                print(f"\n📄 JSON 저장: {json_filename}")
        
        # 4. ChromaDB 저장
        vectorstore = self.save_to_chromadb(summaries, category)
        
        return vectorstore
    
    def process_all_categories(self, save_json: bool = True):
        """모든 카테고리 일괄 처리"""
        print(f"\n{'='*70}")
        print(f"🚀 네이버 금융 리포트 통합 수집 시작")
        print(f"{'='*70}")
        
        results = {}
        
        for category in self.CATEGORIES.keys():
            try:
                vectorstore = self.process_category(category, save_json)
                results[category] = vectorstore
            except Exception as e:
                print(f"\n[ERROR] {category} 처리 중 오류: {e}")
                results[category] = None
        
        # 결과 요약
        print(f"\n{'='*70}")
        print(f"📊 처리 결과 요약")
        print(f"{'='*70}")
        
        for category, vs in results.items():
            cat_name = self.CATEGORIES[category]['name']
            status = "✅ 완료" if vs else "❌ 실패 또는 데이터 없음"
            print(f"  {cat_name:8s}: {status}")
        
        print(f"\n✨ 전체 파이프라인 완료!\n")
        
        return results
    
    def test_search(self, category: str, query: str, k: int = 3):
        """저장된 DB 검색 테스트"""
        cat_info = self.CATEGORIES[category]
        db_path = os.path.join(self.chroma_base_dir, cat_info["db_name"])
        
        if not os.path.exists(db_path):
            print(f"[ERROR] DB 없음: {db_path}")
            return
        
        print(f"\n🔍 [{cat_info['name']}] 검색 테스트")
        print(f"   질문: {query}")
        print(f"   DB: {db_path}")
        
        try:
            embeddings = self.get_embeddings()
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            results = vectorstore.similarity_search(query, k=k)
            
            print(f"\n📌 검색 결과 ({len(results)}개):")
            for i, doc in enumerate(results, 1):
                print(f"\n[{i}] 제목: {doc.metadata['title']}")
                print(f"    날짜: {doc.metadata.get('date', 'N/A')}")
                print(f"    내용: {doc.page_content[:150]}...")
        
        except Exception as e:
            print(f"[ERROR] 검색 실패: {e}")


# ============================================================
# 실행 예시
# ============================================================

if __name__ == "__main__":
    # 파이프라인 초기화
    pipeline = NaverReportPipeline(
        vllm_base_url="http://localhost:8001/v1",
        vllm_model="skt/A.X-4.0-Light",
        vllm_api_key = "vllm-key",  # 수정 
        embedding_model="dragonkue/snowflake-arctic-embed-l-v2.0-ko", # 수정
        chroma_base_dir="./Chroma_db",
        max_text_length=8000,      # PDF 8000자까지만 읽기
        summary_max_tokens=1024,   # 요약 최대 토큰
        temperature=0.3,
        debug=True,
    )
    
    # 방법 1: 모든 카테고리 일괄 처리
    results = pipeline.process_all_categories(save_json=True)
    
    # 방법 2: 특정 카테고리만 처리
    # pipeline.process_category("market")
    # pipeline.process_category("company")
    # pipeline.process_category("economy")
    # pipeline.process_category("industry")
    
    # 검색 테스트
    # pipeline.test_search("market", "엔화에 대한 분석", k=2)
    # pipeline.test_search("company", "삼성전자", k=2)
    # pipeline.test_search("economy", "중국 GDP", k=2)
    # pipeline.test_search("industry", "원전 산업", k=2)