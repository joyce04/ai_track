190117

### Remove duplicates in PostgreSQL

컬럼에 값을 업데이트 할 때, 중복적으로 string이 들어가게 되었는데, 이를 중복을 제거하고 싶다.

현재 컬럼값 상태 / current colume value example : "**tachyarrhythmias**,cardiac disorders,supraventricular tachyarrhythmias,**tachyarrhythmias**"

원하는 상태 / target colume value example : **tachyarrhythmias**,cardiac disorders,supraventricular tachyarrhythmias



1. STRING_TO_ARRAY(col, sep)

   스트링 값을 배열로 변환한다.

   ```sql
   SELECT id, STRING_TO_ARRAY(lower(check_ad), ' , ') from article_table_sentences_m WHERE (check_ad='') is false ORDER BY check_drug limit 50;
   ```

2. 중복제거를 위해 Distinct 키워드를 사용하고 싶지만, 컬럼에만 적용할 수 있음. 따라서 UNNEST를 적용해서 배열을 ROWS로 확장.
   To able to utilize 'DISTINCT' keyword, I need to convert array into rows with UNNEST.

   ```sql
   SELECT
     id,
     drug_list
   FROM article_table_sentences_m m,
         UNNEST(STRING_TO_ARRAY(lower(m.check_ad), ' , ')) AS drug_list
   WHERE (check_ad>='u')
   ORDER BY id DESC
   LIMIT 50;
   ```

3. DISTINCT

   ```sql
   SELECT
     drugs.id,
     STRING_AGG(DISTINCT drugs.drug_list, ' , ') AS drug_2
   FROM
     (SELECT
        id,
        drug_list
      FROM article_table_sentences_m M,
            UNNEST(STRING_TO_ARRAY(lower(m.check_ad), ' , ')) AS drug_list
      WHERE (check_ad >= 'u')
      ORDER BY id DESC
      LIMIT 50) AS drugs
   GROUP BY drugs.id;
   ```



4. Compare with Original Column

   ```sql
   SELECT *
   FROM
     (
       SELECT
         drugs.id,
         STRING_AGG(DISTINCT drugs.drug_list, ' , ') AS drug_2
       FROM
         (SELECT
            id,
            drug_list
          FROM article_table_sentences_m M,
                UNNEST(STRING_TO_ARRAY(lower(m.check_ad), ' , ')) AS drug_list
         ) AS drugs
       GROUP BY drugs.id) AS agg_drugs,
     (SELECT
        id,
        check_ad
      FROM article_table_sentences_m
      WHERE (check_ad = '') IS FALSE) AS origin_drugs
   WHERE agg_drugs.id = origin_drugs.id
         AND agg_drugs.drug_2 != origin_drugs.check_ad;
   ```


5. Update

   ```sql
   UPDATE article_table_sentences_m
   SET check_drug = new_drug.drug_2
   FROM
     (SELECT
        origin_drugs.id  AS id,
        agg_drugs.drug_2 AS drug_2
      FROM
        (
          SELECT
            drugs.id,
            STRING_AGG(DISTINCT drugs.drug_list, ' , ') AS drug_2
          FROM
            (SELECT
               id,
               drug_list
             FROM article_table_sentences_m M,
                   UNNEST(STRING_TO_ARRAY(lower(m.check_drug), ' , ')) AS drug_list
            ) AS drugs
          GROUP BY drugs.id) AS agg_drugs,
        (SELECT
           id,
           check_drug
         FROM article_table_sentences_m
         WHERE (check_drug = '') IS FALSE) AS origin_drugs
      WHERE agg_drugs.id = origin_drugs.id
            AND agg_drugs.drug_2 != origin_drugs.check_drug) AS new_drug
   WHERE new_drug.id = article_table_sentences_m.id;
   ```


