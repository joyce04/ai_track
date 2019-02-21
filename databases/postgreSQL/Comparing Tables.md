### Comparing Tables

I have two seprate tables for drug names collected from different sources. And I want to compare duplicates, and non-overlapping records comparing two tables.

1. Get records only exist in table A but not table B. = A-B

   But B contains list of drug names in one column, so I have UNNEST to retrieve appropriately.

   ```sql
   SELECT C.only_in_drug
   FROM
     (
       SELECT lower(str) as only_in_drug
       FROM drug_180622
       EXCEPT
       (SELECT DISTINCT (TRIM(LOWER(p.drug))) AS t_drug
        FROM
          (SELECT drug
           FROM pubtator_mesh_chem,
                 UNNEST(STRING_TO_ARRAY(mentions, ' , ')) AS drug
          ) AS p
       )
     ) C;
   ```
