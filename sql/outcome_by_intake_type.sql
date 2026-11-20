-- Name: Muhammad Twaib | Roll No: TRN-2026-000353 | Date: 2026-11-20
-- Question: For each broad intake group, how many animals came in, what share
-- did not leave alive, and how long did they stay on average?
-- Joins the cleaned intakes table to a small lookup table, then groups.
SELECT
    l.intake_group,
    COUNT(*)                                        AS intakes,
    ROUND(100.0 * AVG(i.bad_outcome), 1)            AS pct_died_or_euthanised,
    ROUND(AVG(i.length_of_stay_days), 1)            AS avg_stay_days
FROM intakes AS i
JOIN intake_type_lookup AS l
    ON i.intake_type = l.intake_type
WHERE i.bad_outcome IS NOT NULL          -- only rows with a known outcome
GROUP BY l.intake_group
ORDER BY pct_died_or_euthanised DESC;
