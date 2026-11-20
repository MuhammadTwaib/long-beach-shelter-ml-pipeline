# C1 results - feed query with and without an index

Data: 2,000 users, 288,905 follow edges, 300,000 posts over 60 days (built in 1.0 s). Timings = median of 30 users × 3 repeats, in-memory SQLite.

| Query | No index (ms) | Index on posts(author_id, created_at) (ms) | Speed-up |
|---|---|---|---|
| Feed: newest 20 from followed users | 47.59 | 0.07 | 722× |
| Ranker: top 10 by likes + recency (last 3 days) | 12.45 | 0.29 | 42× |
| Ranker served from cache (TTL 60 s, hit rate 99%) | — | 0.011 | 27× vs index |

## EXPLAIN QUERY PLAN - feed query, no index
```
  5   0  SCAN p
  7   0  SEARCH f USING COVERING INDEX sqlite_autoindex_follows_1 (follower_id=? AND followee_id=?)
 28   0  USE TEMP B-TREE FOR ORDER BY
```
## EXPLAIN QUERY PLAN - feed query, with index
```
  6   0  SEARCH f USING COVERING INDEX sqlite_autoindex_follows_1 (follower_id=?)
 12   0  SEARCH p USING INDEX idx_posts_author_time (author_id=?)
 32   0  USE TEMP B-TREE FOR ORDER BY
```
## EXPLAIN QUERY PLAN - ranker, no index
```
  5   0  SCAN p
  9   0  SEARCH f USING COVERING INDEX sqlite_autoindex_follows_1 (follower_id=? AND followee_id=?)
 41   0  USE TEMP B-TREE FOR ORDER BY
```
## EXPLAIN QUERY PLAN - ranker, with index
```
  6   0  SEARCH f USING COVERING INDEX sqlite_autoindex_follows_1 (follower_id=?)
 12   0  SEARCH p USING INDEX idx_posts_author_time (author_id=? AND created_at>?)
 45   0  USE TEMP B-TREE FOR ORDER BY
```
## Example: top 10 for one user (post_id, author_id, likes, score)
```
 110852    20    41 10.690
 152196     6    53 7.348
 297542     6    38 6.759
 200916     6    47 5.642
 112806     6    26 5.358
  39526     6    20 4.601
  13535     6    10 3.297
  97410     6    33 3.202
  48568     6    15 3.190
 175649     6    32 2.902
```
