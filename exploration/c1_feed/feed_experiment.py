# Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
"""Part C1: how an index and a cache change the speed of a social-media feed query.

Builds a synthetic SQLite database (users, follows, posts), then times
  1. the feed query  - newest 20 posts from people a user follows
  2. a tiny ranker   - top 10 posts by likes and recency (bonus)
first WITHOUT and then WITH an index on posts(author_id, created_at), and
finally with an in-memory cache. Writes EXPLAIN QUERY PLAN output and timings
to exploration/c1_feed/results.md.

    python exploration/c1_feed/feed_experiment.py            # default sizes
    python exploration/c1_feed/feed_experiment.py --posts 50000 --users 500   # quick run
"""

import argparse
import random
import sqlite3
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOW = 1_700_000_000          # fixed "current time" (unix seconds) -> reproducible

FEED_SQL = """
SELECT p.post_id, p.author_id, p.created_at, p.likes
FROM posts AS p
JOIN follows AS f ON f.followee_id = p.author_id
WHERE f.follower_id = ?
ORDER BY p.created_at DESC
LIMIT 20;
"""

# Candidate retrieval: only the last 3 days of posts from followed users,
# then rank in SQL by likes / (age in hours + 2)^1.5  (a "Hacker News" style decay).
RANK_SQL = """
SELECT p.post_id, p.author_id, p.likes,
       p.likes / ((( ? - p.created_at) / 3600.0 + 2) * sqrt(( ? - p.created_at) / 3600.0 + 2)) AS score
FROM posts AS p
JOIN follows AS f ON f.followee_id = p.author_id
WHERE f.follower_id = ? AND p.created_at >= ?
ORDER BY score DESC
LIMIT 10;
"""


def build_db(con, users, posts, follows_per_user, seed=42):
    """Create tables and fill them with reproducible random data."""
    rng = random.Random(seed)
    con.executescript("""
        DROP TABLE IF EXISTS posts; DROP TABLE IF EXISTS follows;
        CREATE TABLE follows (follower_id INTEGER, followee_id INTEGER,
                              PRIMARY KEY (follower_id, followee_id));
        CREATE TABLE posts (post_id INTEGER PRIMARY KEY, author_id INTEGER,
                            created_at INTEGER, likes INTEGER, body TEXT);
    """)
    follow_rows = {(u, rng.randrange(users)) for u in range(users) for _ in range(follows_per_user)}
    con.executemany("INSERT INTO follows VALUES (?, ?)", [r for r in follow_rows if r[0] != r[1]])
    # Posts spread over 60 days; a few popular authors post more (skewed like real life).
    post_rows = [(i, int(rng.paretovariate(1.2)) % users, NOW - rng.randrange(60 * 86400),
                  int(rng.expovariate(1 / 20)), "x" * 80) for i in range(posts)]
    con.executemany("INSERT INTO posts VALUES (?, ?, ?, ?, ?)", post_rows)
    con.commit()


def plan(con, sql, params):
    """Return EXPLAIN QUERY PLAN lines as plain text."""
    rows = con.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return "\n".join(f"{r[0]:>3} {r[1]:>3}  {r[3]}" for r in rows)


def time_query(con, sql, param_sets, repeats=3):
    """Median milliseconds per query over all parameter sets and repeats."""
    times = []
    for _ in range(repeats):
        for params in param_sets:
            start = time.perf_counter()
            con.execute(sql, params).fetchall()
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


class FeedCache:
    """Very small read-through cache: user_id -> (computed_at, ranked rows).

    Real systems use Redis/Memcached with a TTL; the idea is the same. A
    cached feed is served without touching the database until it expires.
    """

    def __init__(self, con, ttl_seconds=60):
        self.con, self.ttl, self.store, self.hits, self.misses = con, ttl_seconds, {}, 0, 0

    def top10(self, user_id, clock):
        entry = self.store.get(user_id)
        if entry and clock - entry[0] < self.ttl:
            self.hits += 1
            return entry[1]
        self.misses += 1
        rows = self.con.execute(RANK_SQL, (NOW, NOW, user_id, NOW - 3 * 86400)).fetchall()
        self.store[user_id] = (clock, rows)
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--posts", type=int, default=300_000)
    parser.add_argument("--follows", type=int, default=150, help="follows per user")
    parser.add_argument("--out", type=Path, default=HERE / "results.md")
    args = parser.parse_args()

    con = sqlite3.connect(":memory:")
    con.create_function("sqrt", 1, lambda x: x ** 0.5)   # older SQLite builds lack sqrt()
    t0 = time.perf_counter()
    build_db(con, args.users, args.posts, args.follows)
    build_s = time.perf_counter() - t0
    sample_users = random.Random(7).sample(range(args.users), 30)
    feed_params = [(u,) for u in sample_users]
    rank_params = [(NOW, NOW, u, NOW - 3 * 86400) for u in sample_users]

    report = {"plan_feed_before": plan(con, FEED_SQL, feed_params[0]),
              "plan_rank_before": plan(con, RANK_SQL, rank_params[0]),
              "feed_ms_before": time_query(con, FEED_SQL, feed_params),
              "rank_ms_before": time_query(con, RANK_SQL, rank_params)}

    con.execute("CREATE INDEX idx_posts_author_time ON posts(author_id, created_at)")
    con.execute("ANALYZE")
    report.update({"plan_feed_after": plan(con, FEED_SQL, feed_params[0]),
                   "plan_rank_after": plan(con, RANK_SQL, rank_params[0]),
                   "feed_ms_after": time_query(con, FEED_SQL, feed_params),
                   "rank_ms_after": time_query(con, RANK_SQL, rank_params)})

    # Cache: simulate 3,000 feed opens by 30 users within one minute -> most are hits.
    cache, rng = FeedCache(con), random.Random(1)
    start = time.perf_counter()
    for i in range(3000):
        cache.top10(rng.choice(sample_users), clock=i * 0.02)
    report["rank_ms_cached"] = (time.perf_counter() - start) * 1000 / 3000
    report["cache_hit_rate"] = cache.hits / (cache.hits + cache.misses)
    example = cache.top10(sample_users[0], clock=0)

    n_follows = con.execute("SELECT COUNT(*) FROM follows").fetchone()[0]
    lines = [
        "# C1 results - feed query with and without an index",
        "",
        f"Data: {args.users:,} users, {n_follows:,} follow edges, {args.posts:,} posts over 60 days "
        f"(built in {build_s:.1f} s). Timings = median of 30 users × 3 repeats, in-memory SQLite.",
        "",
        "| Query | No index (ms) | Index on posts(author_id, created_at) (ms) | Speed-up |",
        "|---|---|---|---|",
        f"| Feed: newest 20 from followed users | {report['feed_ms_before']:.2f} | {report['feed_ms_after']:.2f} | "
        f"{report['feed_ms_before'] / report['feed_ms_after']:.0f}× |",
        f"| Ranker: top 10 by likes + recency (last 3 days) | {report['rank_ms_before']:.2f} | "
        f"{report['rank_ms_after']:.2f} | {report['rank_ms_before'] / report['rank_ms_after']:.0f}× |",
        f"| Ranker served from cache (TTL 60 s, hit rate {report['cache_hit_rate']:.0%}) | — | "
        f"{report['rank_ms_cached']:.3f} | {report['rank_ms_after'] / report['rank_ms_cached']:.0f}× vs index |",
        "",
        "## EXPLAIN QUERY PLAN - feed query, no index",
        "```", report["plan_feed_before"], "```",
        "## EXPLAIN QUERY PLAN - feed query, with index",
        "```", report["plan_feed_after"], "```",
        "## EXPLAIN QUERY PLAN - ranker, no index",
        "```", report["plan_rank_before"], "```",
        "## EXPLAIN QUERY PLAN - ranker, with index",
        "```", report["plan_rank_after"], "```",
        "## Example: top 10 for one user (post_id, author_id, likes, score)",
        "```", *[f"{r[0]:>7} {r[1]:>5} {r[2]:>5} {r[3]:.3f}" for r in example], "```",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
