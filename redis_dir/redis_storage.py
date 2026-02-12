from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import redis as redislib


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24)))  # 24h
    keep_messages: int = int(os.getenv("SESSION_KEEP_MESSAGES", "200"))  # 세션당 보관 메시지 수


class RedisSessionStore:
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self.r = redislib.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            decode_responses=True,
        )

    # ----------------------------
    # Key helpers
    # ----------------------------
    def _messages_key(self, session_id: str) -> str:
        return f"chat:{session_id}:messages"

    def _meta_key(self, session_id: str) -> str:
        return f"chat:{session_id}:meta"

    # ----------------------------
    # Session lifecycle
    # ----------------------------
    def create_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()

        self.r.hset(self._meta_key(session_id), mapping={"created_at": now, "updated_at": now})
        self.r.expire(self._meta_key(session_id), self.config.ttl_seconds)

        self.r.expire(self._messages_key(session_id), self.config.ttl_seconds)

    def touch(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.r.hset(self._meta_key(session_id), "updated_at", now)
        self.r.expire(self._meta_key(session_id), self.config.ttl_seconds)
        self.r.expire(self._messages_key(session_id), self.config.ttl_seconds)

    # 세션 삭제 (메타 + 메시지 전부 삭제)
    def delete_session(self, session_id: str) -> bool:
        meta_k = self._meta_key(session_id)
        msg_k = self._messages_key(session_id)

        exists = bool(self.r.exists(meta_k) or self.r.exists(msg_k))
        if not exists:
            return False

        self.r.delete(meta_k, msg_k)
        return True

    # ----------------------------
    # Messages
    # ----------------------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        msg = {
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        k = self._messages_key(session_id)

        self.r.lpush(k, json.dumps(msg, ensure_ascii=False))
        self.r.ltrim(k, 0, self.config.keep_messages - 1)
        self.touch(session_id)

    def get_last_n(self, session_id: str, n: int = 10, chronological: bool = True) -> List[Dict[str, Any]]:
        k = self._messages_key(session_id)
        raw = self.r.lrange(k, 0, max(0, n - 1))
        msgs = [json.loads(x) for x in raw]
        if chronological:
            msgs.reverse()
        return msgs

    def ping(self) -> bool:
        return bool(self.r.ping())
