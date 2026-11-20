# Part C - Exploration homework

Four short write-ups, each with the artifact it asks for. Written in my own words; sources are listed at the end of each section.

---

## C1. Optimised query system behind a social-media feed

**Artifact:** `exploration/c1_feed/feed_experiment.py` → results in `exploration/c1_feed/results.md`
(includes the bonus ranker and a cache).

### How a feed is served fast
A feed is answered in two stages. **Candidate retrieval** collects a few hundred posts that *could* be shown (posts from people I follow, from the last few days). **Ranking** then scores only those candidates (likes, recency, how often I interact with the author) and returns one page. Retrieval must be cheap because it runs on every app open; ranking can be expensive because it only sees a small set.

There are two ways to do retrieval:

| | Fan-out on **read** (pull) | Fan-out on **write** (push) |
|---|---|---|
| When work happens | When I open the app: join my follow list to posts | When someone posts: copy the post ID into every follower's precomputed feed list |
| Read speed | Slower; grows with how many people I follow | Very fast; just read my list |
| Write cost | One insert | One insert *per follower* - a celebrity with 50 M followers creates 50 M writes |
| Good for | Celebrities, inactive users | Normal users who read much more than they post |

Large services use a **hybrid**: push for normal accounts, pull for celebrity accounts at read time, and merge.

**Pagination** uses a cursor ("give me posts older than `created_at = X`") rather than `OFFSET`, because `OFFSET 1000` still has to walk past 1,000 rows and gives duplicates when new posts arrive. **Caching** keeps a user's ranked feed in memory (e.g. Redis) for a short TTL, so ten refreshes in a minute hit the database once.

### Guiding questions
* **Why is the naive "join every followed user's posts at read time" query slow?** Without an index on `posts.author_id`, SQLite must `SCAN` the whole posts table for every request (see the plan: `SCAN p`), then sort all matches in a temporary B-tree. Cost grows with *total* posts in the system, not with what I need.
* **What does an index on `(author_id, created_at)` change?** The plan switches to `SEARCH p USING INDEX idx_posts_author_time (author_id=?)` - for each followed author SQLite jumps straight to that author's posts, already in time order. In my run the feed query went from **47.6 ms to 0.07 ms** (≈700×) and the ranker, which also filters `created_at > now − 3 days`, from **12.5 ms to 0.29 ms** (≈40×), using the second column of the index for the range (`author_id=? AND created_at>?`).
* **When would you precompute feeds instead?** When reads vastly outnumber writes and most authors have a modest number of followers - the usual case. You would *not* precompute for accounts with millions of followers (write storm) or for users who rarely log in (wasted work).
* **How does the experiment support this?** Even with a perfect index, every request still does one index search per followed author and a sort (`USE TEMP B-TREE FOR ORDER BY`). That work is repeated on each refresh. The cache in my experiment answered 99% of 3,000 simulated opens from memory in **0.011 ms** each - 27× faster than the indexed query - which is exactly the saving a precomputed/cached feed gives. The price is staleness (up to 60 s) and memory.

### Bonus - tiny feed ranker
`RANK_SQL` scores each candidate post as `likes / (age_hours + 2)^1.5` (a Hacker-News-style decay: a new post with few likes can beat an old popular one) and returns the top 10 for a user. The `FeedCache` class is a read-through cache with a 60-second TTL.

**Sources:** SQLite, *Query Planning* - https://www.sqlite.org/queryplanner.html ; SQLite, *EXPLAIN QUERY PLAN* - https://www.sqlite.org/eqp.html ; R. Krikorian, *Timelines at Scale* (Twitter, QCon 2012) - https://www.infoq.com/presentations/Twitter-Timeline-Scalability/ ; M. Kleppmann, *Designing Data-Intensive Applications*, ch. 1 (Twitter home-timeline example).

---

## C2. GitHub Actions, PR rules and code reviews

**Artifacts:** `.github/workflows/ci.yml`, `.github/pull_request_template.md`, `.github/CODEOWNERS`, `docs/review_checklist.md`.

### What I learned
* A **workflow** is a YAML file in `.github/workflows/`. It has **triggers** (`on:`), one or more **jobs** (each runs on a fresh virtual machine), and **steps** inside a job (run a command or use a reusable *action* such as `actions/checkout`). Jobs run in parallel unless `needs:` makes one wait for another.
* **Secrets** (API keys, deploy tokens) are stored in repository settings and read as `${{ secrets.NAME }}`; they are masked in logs and are *not* given to workflows triggered by pull requests from forks. My CI needs none, so it also sets `permissions: contents: read` (least privilege).
* **Caching:** `actions/setup-python` with `cache: pip` stores downloaded packages keyed on `requirements.txt`, so later runs install much faster.
* **Branch protection rules** on `main`: require a pull request (no direct pushes), require the `lint` and `test` status checks to pass, require at least one approving review, require review from **CODEOWNERS**, and dismiss old approvals when new commits are pushed.

### Guiding questions
* **Which events should trigger the workflow?** `pull_request` to `main` (catch problems *before* merge), `push` to `main` (prove `main` is still green after merging), and `workflow_dispatch` for a manual re-run. I did not add a schedule; it would make sense if the raw data were re-downloaded nightly.
* **What should block a merge?** Lint errors, any failing unit test, the Part A pipeline failing its quality gate on real data, the gate *not* failing on corrupted data (that would mean the gate is broken), and the saved model failing to score new rows. Plus: no approving review from a code owner.
* **Who should review which files, and why?** Cleaning/validation code changes what "trusted data" means → a data-quality reviewer. Feature and training code can introduce leakage → an ML reviewer. The `.github/` folder controls what CI enforces, so the TA must approve changes there (otherwise anyone could weaken the checks). Raw data should only change with the TA's approval.

### What a good review looks like
Small PRs (under ~400 lines), reviewed quickly, correctness and design before style, questions instead of orders, and approval when the change improves the code base even if not perfect. Details are in `docs/review_checklist.md`.

**Sources:** GitHub Docs, *Workflow syntax for GitHub Actions* - https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions ; GitHub Docs, *About protected branches* - https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches ; GitHub Docs, *About code owners* - https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners ; Google, *Engineering Practices: Code Review* - https://google.github.io/eng-practices/review/

---

## C3. Theory of system design

### Glossary (10 terms, one concrete example each)

| # | Term | Meaning in one line | Concrete example |
|---|---|---|---|
| 1 | **Functional vs non-functional requirements** | *What* the system does vs *how well* (speed, uptime, cost) | "Staff can score an intake" (functional) vs "score returns in < 300 ms, 99.9% uptime" (non-functional) |
| 2 | **Capacity estimation** | Back-of-envelope maths for traffic and storage before choosing tech | 30,000 intakes/year ≈ 82/day ≈ 0.001 req/s - one small server is plenty |
| 3 | **Load balancer** | Spreads requests over several identical servers and skips unhealthy ones | Nginx sends each `/predict` call to whichever of 3 API servers is least busy |
| 4 | **Cache** | Fast copy of data that is expensive to compute or fetch, with an expiry (TTL) | C1: a user's ranked feed kept in memory for 60 s → 99% of opens skip the DB |
| 5 | **SQL (relational) database** | Tables with a fixed schema, joins and ACID transactions | The cleaned shelter table + lookup table joined in SQLite (Part A) |
| 6 | **NoSQL / key-value store** | Store looked up by a key, flexible schema, scales out easily | Redis key `feed:user:42` → list of 500 post IDs |
| 7 | **Message queue** | Buffer where producers drop jobs and workers process them later | On a new post, a "fan-out" job is queued; workers write it into followers' feeds |
| 8 | **Replication** | Keeping copies of the same data on several machines | One primary DB for writes + two read replicas for dashboards |
| 9 | **Sharding (partitioning)** | Splitting data across machines by a key | Posts table split by `author_id % 16` across 16 database servers |
| 10 | **CAP / consistency vs availability** | During a network split you must choose: refuse requests (consistent) or answer with possibly stale data (available) | A like-counter may show 1,203 on one replica and 1,205 on another for a few seconds - acceptable for availability |

### One trade-off: consistency vs availability for a feed
For a social feed I would choose **availability with eventual consistency**. If a replica is cut off from the primary, it is better to keep showing a feed that is a few seconds old than to show an error page. Nobody is harmed if a new post appears 5 seconds late or a like count is briefly off by two. The opposite is true for something like a bank balance or the shelter's "animal released to owner" record: showing stale data there could release an animal twice or lose a payment, so those writes should go to a strongly consistent primary database and fail loudly if it is unreachable. The same company can make different choices for different data.

### Guiding question
* **Relational over key-value when…** the data has relationships I need to query in new ways and must stay correct together - e.g. intakes, outcomes and owners joined for a monthly report, with transactions so an adoption and the payment are saved together or not at all.
* **Key-value over relational when…** I always look data up by one key, need very high read/write throughput and can live without joins - e.g. a session store or a precomputed feed list per user.

**Sources:** A. Xu, *System Design Interview - An Insider's Guide*, ch. 1-2, 11 ; M. Kleppmann, *Designing Data-Intensive Applications*, ch. 2, 5, 6, 9 ; donnemartin, *The System Design Primer* - https://github.com/donnemartin/system-design-primer

---

## C4. A workflow for system design

### My workflow (repeatable steps)
1. **Clarify** - who uses it, for what decision, what is out of scope. Write the questions down *before* drawing anything.
2. **Requirements** - list functional and non-functional ones with numbers (latency, uptime, data size).
3. **Estimate** - traffic, storage, growth; record every assumption next to the number.
4. **API & data model** - the endpoints and the tables, before boxes.
5. **High-level design** - the smallest set of boxes that meets the requirements.
6. **Deep dive** - the riskiest part (here: model correctness and leakage).
7. **Trade-offs & failure modes** - what breaks, what we chose not to do, and why.
8. **Review** - share the doc, collect comments, record decisions in the doc itself.

**What do I ask before drawing boxes?** Who calls the system and how often; what happens if it is wrong or down; what data exists already; what "done" looks like. **Where do I record assumptions and numbers?** In the *Assumptions & estimates* section of the design doc, so a reviewer can challenge them in one place.

### Design document - "Intake Risk Score" service for an animal shelter
*(written by following the workflow above; it turns the Part B model into a small product)*

**1. Clarify.** Users: intake staff and the shelter vet. Decision: which newly arrived animals need a vet check or a rescue-partner call **today**. Out of scope: deciding euthanasia (the score never makes that decision), adoptions, billing.

**2. Requirements.** *Functional:* score one animal at intake; show the score and top reasons on the intake screen; nightly list of high-risk animals still in the shelter; retrain monthly. *Non-functional:* < 1 s per score; available during opening hours (99.5%); every score logged for audit; no personal owner data stored in the scoring service.

**3. Assumptions & estimates.** ~30,000 intakes/year ≈ 82/day, peak maybe 20/hour → < 0.01 requests/s. Each logged score ≈ 1 KB → 30 MB/year. Model file 150 KB. Conclusion: one small container and one SQLite/Postgres database are enough; no load balancer, queue or sharding needed (and adding them would only add failure points).

**4. API & data model.**
`POST /score` - body: the 13 intake fields in `INPUT_COLUMNS` → `{risk_probability, high_risk_flag, model_version, top_factors}`.
`GET /high-risk?date=` - nightly list.
Table `scores(animal_id, intake_date, risk_probability, flag, model_version, created_at)`.

**5. High-level design.**
```
Shelter intake app ──HTTP──▶ Score API (FastAPI + saved sklearn pipeline)
                                   │ writes
                                   ▼
                             scores table (Postgres) ◀── nightly report job
Monthly: export new outcomes ▶ Part A pipeline (quality gate) ▶ train.py ▶ new model file
                                                     └─ CI checks + human review before deploy
```

**6. Deep dive - correctness.** The model must only see intake-time fields (leak checks in Part B found two leaking fields: names and neutered status). The monthly retrain runs the Part A quality gate first; if the pass rate drops below 95% the retrain stops. A new model is deployed only if its PR-AUC on the most recent month is not worse than the current model's.

**7. Trade-offs & failure modes.** If the API is down, intake continues normally (the score is advice, not a gate). Chosen *synchronous* scoring over a queue because volume is tiny and staff want the answer on screen. Risk: dogs are under-flagged (recall 0.31 in 2024 test) - shown on the screen as "low confidence for dogs". Risk of self-fulfilling prophecy (high score → less effort → bad outcome): the score is framed as "needs help first", and outcomes are monitored by score band.

**8. Review.** Doc shared with the shelter vet and the TA; comments and decisions are added below this line.

**Sources:** Google, *Design Docs at Google* (Industrial Empathy blog) - https://www.industrialempathy.com/posts/design-docs-at-google/ ; A. Xu, *System Design Interview*, ch. 3 "A framework for system design interviews".
