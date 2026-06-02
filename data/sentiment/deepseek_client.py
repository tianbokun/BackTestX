"""DeepSeek API 客户端 — 情感分析 + 基本面认知提取.

API: https://api.deepseek.com/v1/chat/completions
模型: deepseek-chat
定价: ~¥1/1M 输入, ¥2/1M 输出 tokens
"""

import time
from typing import Optional

import requests

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = DEEPSEEK_BASE):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._last_request = 0.0
        self._min_interval = 0.12  # ~500 RPM

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _call(self, messages: list[dict], max_tokens: int = 256,
              temperature: float = 0.1) -> Optional[str]:
        self._rate_limit()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None

    def analyze_sentiment(self, texts: list[str], symbol: str = "") -> list[dict]:
        """批量分析文本情感.

        Returns:
            list[dict]: 每项 {sentiment, score, confidence}
        """
        results = []
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            combined = "\n---\n".join(f"[{j}] {t}" for j, t in enumerate(batch))
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一位A股市场情绪分析师。分析用户输入的各条内容的情感倾向。"
                        "请以JSON数组格式返回，每项包含: "
                        '{"sentiment": "positive/negative/neutral", "score": -1~1, "confidence": 0~1}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"分析以下关于{symbol}的讨论内容的情感倾向:\n{combined}",
                },
            ]
            resp = self._call(messages)
            if resp:
                try:
                    import json
                    parsed = json.loads(resp)
                    if isinstance(parsed, list):
                        for item in parsed:
                            results.append({
                                "sentiment": item.get("sentiment", "neutral"),
                                "score": float(item.get("score", 0)),
                                "confidence": float(item.get("confidence", 0)),
                            })
                        continue
                except json.JSONDecodeError:
                    pass
            # Fallback: return neutral for each text in batch
            for _ in batch:
                results.append({"sentiment": "neutral", "score": 0.0, "confidence": 0.0})
        return results

    def extract_fundamentals(self, articles: list[dict], symbol: str = "") -> list[dict]:
        """从新闻/研报中提取基本面信号.

        Args:
            articles: [{"title": str, "content": str, "date": str}, ...]

        Returns:
            list[dict]: 每项 {sentiment, score, industry_outlook,
                             fundamental_signals, management_tone}
        """
        results = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "") or article.get("summary", "")
            text = f"{title}\n{content}"[:2000]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一位基本面分析师。从财经新闻中提取结构化信息，"
                        "输出JSON:\n"
                        '{"sentiment": "positive/negative/neutral",\n'
                        ' "sentiment_score": -1~1,\n'
                        ' "industry_outlook": "improving/stable/deteriorating",\n'
                        ' "fundamental_signals": {\n'
                        '   "revenue_outlook": "positive/negative/neutral",\n'
                        '   "margin_trend": "expanding/contracting/stable",\n'
                        '   "market_share": "gaining/losing/stable"\n'
                        ' },\n'
                        ' "management_tone": "confident/cautious/neutral"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"分析以下关于{symbol}的新闻:\n{text}",
                },
            ]
            resp = self._call(messages, max_tokens=512)
            if resp:
                try:
                    import json
                    parsed = json.loads(resp)
                    results.append(parsed)
                    continue
                except json.JSONDecodeError:
                    pass
            results.append({
                "sentiment": "neutral", "sentiment_score": 0.0,
                "industry_outlook": "stable",
                "fundamental_signals": {
                    "revenue_outlook": "neutral",
                    "margin_trend": "stable",
                    "market_share": "stable",
                },
                "management_tone": "neutral",
            })
        return results
