"""
OPAC Wikipedia Engine  (Phase 3.5)
====================================
Offline Wikipedia search using wikipedia-api (online) with a local SQLite
cache, or a pre-built SQLite database from a Wikipedia dump.

Two modes:
  1. Online mode  — fetches from Wikipedia API, caches locally in wiki.db
  2. Offline mode — queries a pre-built SQLite database (from dump import)

The Wikipedia context is injected into the LLM prompt when the agent
detects that a factual question would benefit from reference material.

Install:
    pip install wikipedia-api

Build offline database from Wikipedia dump (optional, for true offline):
    python -m opac.tools.build_wiki_db --dump enwiki-latest-abstract.xml.gz

Usage (automatic — called by the agent):
    wiki = WikiEngine()
    results = wiki.search("machine learning")
    context = wiki.format_context(results)
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
        """Return a truncated snippet suitable for LLM context."""
        text = self.summary.strip()
        if len(text) <= max_chars:
            return text
        # Cut at sentence boundary
        cut = text.rfind(". ", 0, max_chars)
        return text[:cut + 1] if cut > 0 else text[:max_chars]


class WikiEngine:
    def __init__(self):
        self._db_conn: Optional[sqlite3.Connection] = None
        self._online_wiki = None
        self._mode = "none"

    def setup(self) -> None:
        """Initialise the wiki engine — try DB first, then online API."""
        if not WIKI_ENABLED:
            return

        # Try local SQLite DB
        if Path(WIKI_DB_PATH).exists():
            try:
                self._db_conn = sqlite3.connect(str(WIKI_DB_PATH), check_same_thread=False)
                self._db_conn.execute("SELECT COUNT(*) FROM articles")
                self._mode = "offline"
                count = self._db_conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
                logger.info(f"Wikipedia: offline DB loaded ({count:,} articles)")
                return
            except Exception as e:
                logger.debug(f"Wikipedia DB error: {e}")

        # Try online API
        try:
            import wikipediaapi
            self._online_wiki = wikipediaapi.Wikipedia(
                user_agent="OPAC-local-agent/1.0",
                language="en",
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )
            self._mode = "online"
            logger.info("Wikipedia: online API mode (results cached locally)")
            self._init_cache_db()
        except ImportError:
            logger.info("Wikipedia: not available (pip install wikipedia-api)")
            self._mode = "none"

    @property
    def available(self) -> bool:
        return self._mode != "none" and WIKI_ENABLED

    def search(self, query: str) -> List[WikiResult]:
        """Search Wikipedia and return up to WIKI_MAX_RESULTS results."""
        if not self.available:
            return []

        query = query.strip()
        if not query:
            return []

        # Try cache first (both modes)
        cached = self._search_cache(query)
        if cached:
            logger.info(f"Wikipedia cache hit: '{query}' ({len(cached)} results)")
            return cached

        if self._mode == "offline":
            return self._search_offline(query)
        elif self._mode == "online":
            return self._search_online(query)
        return []

    def format_context(self, results: List[WikiResult]) -> str:
        """Format search results into a context block for the LLM prompt."""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Wikipedia: {r.title}]\n{r.snippet()}")
        return "\n\n".join(parts)

    def is_factual_query(self, text: str) -> bool:
        """
        Heuristic: should we look this up on Wikipedia?
        Returns True for questions that likely need reference knowledge.
        """
        text_lower = text.lower().strip()

        # Skip if it's a file/URL operation
        if any(text_lower.startswith(p) for p in
               ("summarize", "summarise", "open", "launch", "http")):
            return False

        # Skip very short inputs (greetings, commands)
        if len(text.split()) < 4:
            return False

        # Skip casual conversation patterns
        casual = re.compile(
            r"^(hi|hello|hey|how are you|what('s| is) up|"
            r"good (morning|afternoon|evening)|thanks|thank you|ok|okay|"
            r"can you help|what can you do|who are you)", re.I
        )
        if casual.match(text_lower):
            return False

        # Trigger on factual question patterns
        factual = re.compile(
            r"\b(what is|what are|who is|who was|where is|when did|when was|"
            r"how does|how do|why does|why did|explain|define|tell me about|"
            r"what do you know about|history of|origin of|invented|discovered|"
            r"capital of|population of|founded|created|built|wrote|"
            r"difference between|compare|versus|vs)\b", re.I
        )
        return bool(factual.search(text))

    # ── offline search ────────────────────────────────────────────────────────

    def _search_offline(self, query: str) -> List[WikiResult]:
        """Full-text search in local SQLite database."""
        if not self._db_conn:
            return []
        try:
            # Try FTS5 first, then LIKE fallback
            try:
                rows = self._db_conn.execute(
                    "SELECT title, summary, url FROM articles "
                    "WHERE articles MATCH ? LIMIT ?",
                    (query, WIKI_MAX_RESULTS)
                ).fetchall()
            except sqlite3.OperationalError:
                # Non-FTS table — use LIKE
                like = f"%{query}%"
                rows = self._db_conn.execute(
                    "SELECT title, summary, url FROM articles "
                    "WHERE title LIKE ? OR summary LIKE ? LIMIT ?",
                    (like, like, WIKI_MAX_RESULTS)
                ).fetchall()

            results = [WikiResult(title=r[0], summary=r[1], url=r[2] or "") for r in rows]
            self._save_to_cache(query, results)
            return results
        except Exception as e:
            logger.error(f"Wikipedia offline search error: {e}")
            return []

    # ── online search ─────────────────────────────────────────────────────────

    def _search_online(self, query: str) -> List[WikiResult]:
        """Fetch from Wikipedia API and cache."""
        if not self._online_wiki:
            return []
        try:
            import wikipediaapi
            page = self._online_wiki.page(query)
            results = []
            if page.exists():
                results.append(WikiResult(
                    title   = page.title,
                    summary = page.summary,
                    url     = page.fullurl,
                ))
            self._save_to_cache(query, results)
            return results
        except Exception as e:
            logger.error(f"Wikipedia online search error: {e}")
            return []

    # ── cache database ────────────────────────────────────────────────────────

    def _init_cache_db(self) -> None:
        """Create local cache DB for online results."""
        try:
            conn = sqlite3.connect(str(WIKI_DB_PATH), check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id      INTEGER PRIMARY KEY,
                    query   TEXT,
                    title   TEXT,
                    summary TEXT,
                    url     TEXT,
                    ts      INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON articles(query)")
            conn.commit()
            self._db_conn = conn
        except Exception as e:
            logger.error(f"Cache DB init error: {e}")

    def _search_cache(self, query: str) -> List[WikiResult]:
        if not self._db_conn:
            return []
        try:
            cutoff = int(time.time()) - 86400 * 7   # 7-day cache
            rows   = self._db_conn.execute(
                "SELECT title, summary, url FROM articles "
                "WHERE query = ? AND ts > ? LIMIT ?",
                (query.lower(), cutoff, WIKI_MAX_RESULTS)
            ).fetchall()
            return [WikiResult(title=r[0], summary=r[1], url=r[2] or "") for r in rows]
        except Exception:
            return []

    def _save_to_cache(self, query: str, results: List[WikiResult]) -> None:
        if not self._db_conn or not results:
            return
        try:
            ts = int(time.time())
            self._db_conn.executemany(
                "INSERT OR REPLACE INTO articles (query, title, summary, url, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                [(query.lower(), r.title, r.summary, r.url, ts) for r in results]
            )
            self._db_conn.commit()
        except Exception as e:
            logger.debug(f"Cache save error: {e}")
