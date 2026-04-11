"""
OPAC Wikipedia Engine  (Phase 3.5)
====================================
Searches Wikipedia and returns article summaries to enrich LLM responses.

Mode selection (automatic):
  1. Online API  -- fetches from Wikipedia, caches results in local SQLite
  2. Offline DB  -- queries a pre-built SQLite database (advanced, optional)

The online mode works out of the box with:
    pip install wikipedia-api

Results are cached locally so repeated searches are instant and offline.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger
from config.settings import (
    WIKI_DB_PATH, WIKI_MAX_RESULTS, WIKI_SNIPPET_CHARS, WIKI_ENABLED
)

logger = get_logger("opac.wiki")


@dataclass
class WikiResult:
    title:   str
    summary: str
    url:     str = ""

    def snippet(self, max_chars: int = WIKI_SNIPPET_CHARS) -> str:
        text = self.summary.strip()
        if len(text) <= max_chars:
            return text
        cut = text.rfind(". ", 0, max_chars)
        return text[:cut + 1] if cut > 0 else text[:max_chars]


class WikiEngine:
    def __init__(self):
        self._db_conn       = None
        self._online_wiki   = None
        self._mode          = "none"

    def setup(self) -> None:
        """Initialise Wikipedia engine. Online API is the default."""
        if not WIKI_ENABLED:
            return

        # ── Try online API first (most useful, works immediately) ─────────────
        try:
            import wikipediaapi
            self._online_wiki = wikipediaapi.Wikipedia(
                user_agent="OPAC-local-agent/1.0",
                language="en",
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )
            self._mode = "online"
            logger.info("Wikipedia: online mode active (results cached locally)")
            self._init_cache_db()
            return
        except ImportError:
            logger.info("Wikipedia: pip install wikipedia-api to enable")
        except Exception as e:
            logger.debug(f"Wikipedia online init error: {e}")

        # ── Try offline DB (only if it has actual content) ────────────────────
        if Path(WIKI_DB_PATH).exists():
            try:
                conn  = sqlite3.connect(str(WIKI_DB_PATH), check_same_thread=False)
                count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
                if count > 0:
                    self._db_conn = conn
                    self._mode    = "offline"
                    logger.info(f"Wikipedia: offline DB loaded ({count:,} articles)")
                    return
                else:
                    conn.close()
                    logger.info("Wikipedia: offline DB is empty, skipping")
            except Exception as e:
                logger.debug(f"Wikipedia offline DB error: {e}")

        self._mode = "none"
        logger.info("Wikipedia: not available")

    @property
    def available(self) -> bool:
        return self._mode != "none" and WIKI_ENABLED

    def search(self, query: str) -> List[WikiResult]:
        """Search Wikipedia and return up to WIKI_MAX_RESULTS results."""
        if not self.available or not query.strip():
            return []

        # Check local cache first (fast, works offline after first search)
        cached = self._search_cache(query)
        if cached:
            logger.info(f"Wikipedia cache hit: '{query}'")
            return cached

        if self._mode == "online":
            return self._search_online(query)
        elif self._mode == "offline":
            return self._search_offline(query)
        return []

    def format_context(self, results: List[WikiResult]) -> str:
        if not results:
            return ""
        parts = [f"[Wikipedia: {r.title}]\n{r.snippet()}" for r in results]
        return "\n\n".join(parts)

    def is_factual_query(self, text: str) -> bool:
        """Return True if the query likely benefits from Wikipedia context."""
        text_lower = text.lower().strip()

        # Skip commands and operations
        if any(text_lower.startswith(p) for p in
               ("summarize", "summarise", "open", "launch", "http", "search",
                "voice", "clear", "help", "quit", "exit", "be casual", "be formal")):
            return False

        # Skip very short inputs
        if len(text.split()) < 3:
            return False

        # Skip casual conversation
        casual_re = re.compile(
            r"^(hi|hello|hey|how are you|what.?s up|thanks?|thank you|"
            r"good (morning|afternoon|evening)|ok|okay|cool|nice|great|"
            r"can you help|what can you do|who are you|are you there).*$",
            re.I
        )
        if casual_re.match(text_lower):
            return False

        # Trigger on factual questions
        factual_re = re.compile(
            r"\b(what is|what are|who is|who was|where is|when did|when was|"
            r"how does|how do|why does|why did|explain|define|tell me about|"
            r"what do you know about|history of|origin of|invented|discovered|"
            r"capital of|founded|created|built|wrote|difference between|"
            r"compare|versus|describe|about)\b",
            re.I
        )
        return bool(factual_re.search(text_lower))

    # ── online search ─────────────────────────────────────────────────────────

    def _search_online(self, query: str) -> List[WikiResult]:
        """Fetch article from Wikipedia API."""
        if not self._online_wiki:
            return []
        try:
            page = self._online_wiki.page(query)
            results = []
            if page.exists() and page.summary:
                results.append(WikiResult(
                    title   = page.title,
                    summary = page.summary,
                    url     = page.fullurl,
                ))
                logger.info(f"Wikipedia: fetched '{page.title}'")
            else:
                logger.info(f"Wikipedia: no article found for '{query}'")
            self._save_to_cache(query, results)
            return results
        except Exception as e:
            logger.error(f"Wikipedia online search error: {e}")
            return []

    # ── offline search ────────────────────────────────────────────────────────

    def _search_offline(self, query: str) -> List[WikiResult]:
        if not self._db_conn:
            return []
        try:
            like = f"%{query}%"
            rows = self._db_conn.execute(
                "SELECT title, summary, url FROM articles "
                "WHERE title LIKE ? OR summary LIKE ? LIMIT ?",
                (like, like, WIKI_MAX_RESULTS)
            ).fetchall()
            return [WikiResult(title=r[0], summary=r[1], url=r[2] or "") for r in rows]
        except Exception as e:
            logger.error(f"Wikipedia offline search error: {e}")
            return []

    # ── local cache ───────────────────────────────────────────────────────────

    def _init_cache_db(self) -> None:
        """Set up local SQLite cache for online results."""
        try:
            conn = sqlite3.connect(str(WIKI_DB_PATH), check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wiki_cache (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    query   TEXT NOT NULL,
                    title   TEXT,
                    summary TEXT,
                    url     TEXT,
                    ts      INTEGER
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wiki_query ON wiki_cache(query)"
            )
            conn.commit()
            self._db_conn = conn
            logger.info("Wikipedia: local cache DB ready")
        except Exception as e:
            logger.error(f"Wikipedia cache DB init error: {e}")

    def _search_cache(self, query: str) -> List[WikiResult]:
        if not self._db_conn:
            return []
        try:
            # Cache valid for 7 days
            cutoff = int(time.time()) - 86400 * 7
            rows   = self._db_conn.execute(
                "SELECT title, summary, url FROM wiki_cache "
                "WHERE query = ? AND ts > ? LIMIT ?",
                (query.lower().strip(), cutoff, WIKI_MAX_RESULTS)
            ).fetchall()
            return [WikiResult(title=r[0], summary=r[1], url=r[2] or "")
                    for r in rows if r[0] and r[1]]
        except Exception:
            return []

    def _save_to_cache(self, query: str, results: List[WikiResult]) -> None:
        if not self._db_conn or not results:
            return
        try:
            ts = int(time.time())
            self._db_conn.executemany(
                "INSERT INTO wiki_cache (query, title, summary, url, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                [(query.lower().strip(), r.title, r.summary, r.url, ts)
                 for r in results]
            )
            self._db_conn.commit()
        except Exception as e:
            logger.debug(f"Wikipedia cache save error: {e}")
