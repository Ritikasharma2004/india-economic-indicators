-- ============================================================================
-- India Economic Indicators - analysis queries
--
-- Written for SQLite (data/india_economy.db) but standard SQL throughout:
-- these run unchanged on PostgreSQL and MySQL 8+.
--
-- Table: indicators(year, indicator, value)
--        indicator_meta(indicator, display_name, unit, category)
-- ============================================================================


-- Q1. Average GDP growth by decade
-- Which decades did India actually grow fastest in?
SELECT
    (year / 10) * 10                       AS decade,
    COUNT(*)                               AS years_observed,
    ROUND(AVG(value), 2)                   AS avg_growth_pct,
    ROUND(MIN(value), 2)                   AS worst_year_pct,
    ROUND(MAX(value), 2)                   AS best_year_pct
FROM indicators
WHERE indicator = 'gdp_growth_pct'
GROUP BY decade
ORDER BY decade;


-- Q2. Growth volatility by decade
-- Average growth alone hides risk. SQLite has no STDDEV, so it is computed
-- from the definition: sqrt(E[x^2] - E[x]^2).
SELECT
    (year / 10) * 10                                              AS decade,
    ROUND(AVG(value), 2)                                          AS avg_growth_pct,
    ROUND(SQRT(AVG(value * value) - AVG(value) * AVG(value)), 2)  AS volatility,
    ROUND(
        AVG(value) / NULLIF(SQRT(AVG(value * value) - AVG(value) * AVG(value)), 0),
        2
    )                                                             AS growth_per_unit_risk
FROM indicators
WHERE indicator = 'gdp_growth_pct'
GROUP BY decade
ORDER BY decade;


-- Q3. Year-on-year change in growth, using a window function
-- Isolates the years where the growth rate itself moved sharply.
WITH growth AS (
    SELECT
        year,
        value                                        AS growth_pct,
        LAG(value) OVER (ORDER BY year)              AS prev_growth_pct
    FROM indicators
    WHERE indicator = 'gdp_growth_pct'
)
SELECT
    year,
    ROUND(growth_pct, 2)                             AS growth_pct,
    ROUND(growth_pct - prev_growth_pct, 2)           AS change_vs_prev_year
FROM growth
WHERE prev_growth_pct IS NOT NULL
ORDER BY ABS(growth_pct - prev_growth_pct) DESC
LIMIT 10;


-- Q4. Three-year moving average of growth
-- Smooths single-year shocks to show the underlying trend.
SELECT
    year,
    ROUND(value, 2)                                  AS growth_pct,
    ROUND(
        AVG(value) OVER (
            ORDER BY year
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    )                                                AS moving_avg_3yr
FROM indicators
WHERE indicator = 'gdp_growth_pct'
  AND year >= 1991
ORDER BY year;


-- Q5. Structural shift of the economy
-- Agriculture, industry and services as a share of GDP, one row per decade.
SELECT
    (year / 10) * 10 AS decade,
    ROUND(AVG(CASE WHEN indicator = 'agriculture_pct_gdp' THEN value END), 1) AS agriculture,
    ROUND(AVG(CASE WHEN indicator = 'industry_pct_gdp'    THEN value END), 1) AS industry,
    ROUND(AVG(CASE WHEN indicator = 'services_pct_gdp'    THEN value END), 1) AS services
FROM indicators
WHERE indicator IN ('agriculture_pct_gdp', 'industry_pct_gdp', 'services_pct_gdp')
GROUP BY decade
ORDER BY decade;


-- Q6. Does high inflation coincide with weak growth?
-- Joins the two series on year and buckets by inflation level.
WITH paired AS (
    SELECT
        g.year,
        g.value AS growth_pct,
        i.value AS inflation_pct
    FROM indicators g
    JOIN indicators i
      ON i.year = g.year
     AND i.indicator = 'inflation_cpi_pct'
    WHERE g.indicator = 'gdp_growth_pct'
)
SELECT
    CASE
        WHEN inflation_pct <  5  THEN 'a. Low (<5%)'
        WHEN inflation_pct < 10  THEN 'b. Moderate (5-10%)'
        ELSE                          'c. High (10%+)'
    END                        AS inflation_band,
    COUNT(*)                   AS years,
    ROUND(AVG(growth_pct), 2)  AS avg_growth_pct,
    ROUND(MIN(growth_pct), 2)  AS min_growth_pct,
    ROUND(MAX(growth_pct), 2)  AS max_growth_pct
FROM paired
GROUP BY inflation_band
ORDER BY inflation_band;


-- Q7. Best and worst growth years, ranked
-- DENSE_RANK from both ends in a single pass.
WITH ranked AS (
    SELECT
        year,
        value,
        DENSE_RANK() OVER (ORDER BY value DESC) AS rank_best,
        DENSE_RANK() OVER (ORDER BY value ASC)  AS rank_worst
    FROM indicators
    WHERE indicator = 'gdp_growth_pct'
)
SELECT
    year,
    ROUND(value, 2) AS growth_pct,
    CASE WHEN rank_best <= 5 THEN 'top 5' ELSE 'bottom 5' END AS category
FROM ranked
WHERE rank_best <= 5 OR rank_worst <= 5
ORDER BY value DESC;


-- Q8. Trade openness against growth, post-liberalisation
-- Exports plus imports as a share of GDP is the standard openness measure.
WITH openness AS (
    SELECT
        e.year,
        e.value + im.value AS openness_pct_gdp,
        g.value            AS growth_pct
    FROM indicators e
    JOIN indicators im ON im.year = e.year AND im.indicator = 'imports_pct_gdp'
    JOIN indicators g  ON g.year  = e.year AND g.indicator  = 'gdp_growth_pct'
    WHERE e.indicator = 'exports_pct_gdp'
      AND e.year >= 1991
)
SELECT
    CASE
        WHEN openness_pct_gdp < 30 THEN 'a. Under 30%'
        WHEN openness_pct_gdp < 45 THEN 'b. 30-45%'
        ELSE                            'c. Over 45%'
    END                             AS openness_band,
    COUNT(*)                        AS years,
    ROUND(AVG(openness_pct_gdp), 1) AS avg_openness,
    ROUND(AVG(growth_pct), 2)       AS avg_growth_pct
FROM openness
GROUP BY openness_band
ORDER BY openness_band;
