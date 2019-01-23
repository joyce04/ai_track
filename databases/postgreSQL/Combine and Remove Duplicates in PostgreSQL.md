190123

## Combine Columns and Remove duplicates in PostgreSQL

초기에 서로 다른 추처에서 얻은 데이터를 활용할 방안이 다를 수 있다는 계획이 있어서,
다른 출처에서 얻은 같은 종류의 값들을 두 개의 컬럼으로 나눠서 저장한 상태였지만,
추출된 내용을 보니 하나의 컬럼으로 합쳐도 무리가 없을 것이라고 판단 되었다.
<br>
따라서 두 컬럼을 합쳐서 중복값을 제거한 상태로 하나의 컬럼으로 업데이트 하기로하였다.
<br>

### 1. CONCAT VS CONCAT_WS
CONCAT : Sepeartor가 필요 없다면,
하지만 각 컬럼은 스트링에서 리스트로 변환해야하는 상태이기 때문에, 이 경우, Separator필요.
[concat_ws(',', 'abcde', 2, NULL, 22)	=> abcde,2,22](https://www.postgresql.org/docs/current/functions-string.html#FUNCTIONS-STRING-OTHER)

` SELECT id, CONCAT_WS(' , ', drug, check_drug) as t_drug
from article_table_sentences_m
WHERE (check_drug='') is false or (drug='') is false
ORDER BY check_drug
limit 50; `

### 2. 중복제거를 위해 DISTINCT, STRING_AGG로 다시 스트링으로 변환

### 3. LENGTH
empty string 때문에 ','가 추가되는 문제가 생겨서 length비교를 통해 1보다 커야지만 다시 스트링으로 합쳐지도록 변경

` select d.id,
STRING_AGG(DISTINCT(TRIM(d.t_drug)), ' , ') AS t_d
from (
SELECT id, t_drug, length(trim(t_drug)) as len
FROM article_table_sentences_m M,
	UNNEST(STRING_TO_ARRAY(CONCAT_WS(' , ', drug, check_drug), ',')) AS t_drug
WHERE (check_drug='') is false or (drug='') is false
ORDER BY check_drug
LIMIT 500) as d
where d.len >1
group by d.id; `


정돈된 쿼리로 UPDATE

` update article_table_sentences_m am
set t_ad = com_d.tt_ad
from
(select d.id,
	STRING_AGG(DISTINCT(TRIM(d.tt_ad)), ' , ') AS tt_ad
	from (
		SELECT id, tt_ad, length(trim(tt_ad)) as len
		FROM article_table_sentences_m M,
			UNNEST(STRING_TO_ARRAY(CONCAT_WS(' , ', adverse_effect, check_ad), ',')) AS tt_ad
		WHERE (check_ad='') is false or (adverse_effect='') is false) as d
	where d.len >1
	group by d.id
) as com_d
where am.id = com_d.id; `


### 4. Retrieve Unique(Distinct) values in PostgreSQL
I want to retrieve unique values from a certain column regardless of record ids.
<br>

` SELECT DISTINCT(TRIM(d.tt_drug)) AS t_d
FROM
(SELECT id, tt_drug
FROM article_table_sentences_m M,
	UNNEST(STRING_TO_ARRAY(t_drug, ' , ')) AS tt_drug
WHERE (tt_drug='') is false) as d; `
