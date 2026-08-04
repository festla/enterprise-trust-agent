# financial_fact_retrieval_dev_v1 audit

## 1. Audit Scope

- Evaluation set: `financial_fact_retrieval_dev_v1`
- Current case count: 16
- Reports covered:
  - `midea_group_2024`: 8 cases
  - `hisense_home_2024`: 8 cases
- Audit purpose:
  - confirm Gold PDF page;
  - confirm printed page;
  - confirm fiscal period and consolidated/company scope;
  - confirm disclosed value and unit;
  - record the source table and a short evidence excerpt.

## 2. Verification Convention

- `CNY_thousand`: amounts disclosed in RMB thousands.
- `CNY`: amounts disclosed in RMB yuan.
- Midea 2024 has a stable page mapping of `pdf_page = printed_page + 1`.
- This audit was completed by checking the source annual-report PDF and existing project records.
- Midea 2024 cases have been manually spot-checked and marked `completed` by the project owner.
- Hisense 2024 cases have been manually spot-checked and marked `completed` by the project owner.

---

## 3. Midea Group 2024

### fact_001

- question: 美的集团2024年营业收入是多少？
- metric_name: 营业收入
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [158]
- gold_printed_pages: [157]
- verified_value: 407,149,600
- unit: CNY_thousand
- source_table: 2024年度合并及公司利润表
- column_label: 2024年度·合并
- evidence_excerpt: 其中：营业收入……407,149,600
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 取合并口径，不取公司口径946,607，也不取2023年度372,037,280。

### fact_003

- question: 美的集团2024年合并报表口径的营业总收入是多少？
- metric_name: 营业总收入
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [158]
- gold_printed_pages: [157]
- verified_value: 409,084,266
- unit: CNY_thousand
- source_table: 2024年度合并及公司利润表
- column_label: 2024年度·合并
- evidence_excerpt: 一、营业总收入……409,084,266
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 营业总收入包含营业收入、利息收入和手续费及佣金收入。

### smoke_midea_2024_ocf

- question: 美的集团2024年经营活动产生的现金流量净额是多少？
- metric_name: 经营活动产生的现金流量净额
- statement_type: cash_flow_statement
- statement_scope: consolidated
- gold_pdf_pages: [160]
- gold_printed_pages: [159]
- verified_value: 60,511,572
- unit: CNY_thousand
- source_table: 2024年度合并及公司现金流量表
- column_label: 2024年度·合并
- evidence_excerpt: 经营活动产生的现金流量净额……60,511,572
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取经营活动现金流入小计412,775,133，也不取公司口径4,645,875。

### smoke_midea_2024_total_assets

- question: 美的集团2024年末合并口径的资产总计是多少？
- metric_name: 资产总计
- statement_type: balance_sheet
- statement_scope: consolidated
- gold_pdf_pages: [156]
- gold_printed_pages: [155]
- verified_value: 604,351,853
- unit: CNY_thousand
- source_table: 合并及公司资产负债表
- column_label: 2024年12月31日·合并
- evidence_excerpt: 资产总计……604,351,853
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取2023年末486,038,184，也不取公司口径300,151,455。

### smoke_midea_2024_mainland_external_revenue

- question: 美的集团2024年度中国内地对外交易收入是多少？
- metric_name: 中国内地对外交易收入
- statement_type: other
- statement_scope: consolidated
- gold_pdf_pages: [271]
- gold_printed_pages: [270]
- verified_value: 240,049,883
- unit: CNY_thousand
- source_table: 财务报表附注八、分部报告（续）—地区信息
- column_label: 对外交易收入·2024年度·中国内地
- evidence_excerpt: 中国内地……240,049,883
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 其他国家或地区为169,034,383；两者合计409,084,266。

### smoke_midea_2024_operating_cost

- question: 美的集团2024年合并口径的营业成本是多少？
- metric_name: 营业成本
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [158]
- gold_printed_pages: [157]
- verified_value: 299,584,935
- unit: CNY_thousand
- source_table: 2024年度合并及公司利润表
- column_label: 2024年度·合并
- evidence_excerpt: 减：营业成本……(299,584,935)
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 报表以括号表示成本扣减项；审计值记录为绝对金额299,584,935。

### smoke_midea_2024_parent_net_profit

- question: 美的集团2024年归属于母公司股东的净利润是多少？
- metric_name: 归属于母公司股东的净利润
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [158]
- gold_printed_pages: [157]
- verified_value: 38,537,237
- unit: CNY_thousand
- source_table: 2024年度合并及公司利润表
- column_label: 2024年度·合并
- evidence_excerpt: 归属于母公司股东的净利润……38,537,237
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取净利润38,757,214，也不取归属于母公司股东的综合收益总额38,183,500。

### smoke_midea_2024_cash

- question: 美的集团2024年末合并口径的货币资金是多少？
- metric_name: 货币资金
- statement_type: balance_sheet
- statement_scope: consolidated
- gold_pdf_pages: [156]
- gold_printed_pages: [155]
- verified_value: 140,410,308
- unit: CNY_thousand
- source_table: 合并及公司资产负债表
- column_label: 2024年12月31日·合并
- evidence_excerpt: 货币资金……140,410,308
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取2023年末81,673,846，也不取公司口径18,441,820。

---

---

## 4. Hisense Home Appliances 2024

### fact_006

- question: 海信家电2024年末合并口径的存货是多少？
- metric_name: 存货
- statement_type: balance_sheet
- statement_scope: consolidated
- gold_pdf_pages: [112]
- gold_printed_pages: [110]
- verified_value: 7,566,932,954.39
- unit: CNY
- source_table: 1、合并资产负债表
- column_label: 2024年12月31日·期末余额
- evidence_excerpt: 存货……7,566,932,954.39
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取期初余额6,774,603,438.00；该题的Dense正确页首次排名为24。

### smoke_hisense_2024_revenue

- question: 海信家电2024年合并口径的营业收入是多少？
- metric_name: 营业收入
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [116]
- gold_printed_pages: [114]
- verified_value: 92,745,611,109.52
- unit: CNY
- source_table: 3、合并利润表
- column_label: 2024年度
- evidence_excerpt: 其中：营业收入……92,745,611,109.52
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 合并利润表中营业总收入与营业收入金额相同；不取母公司营业收入5,690,120,355.19。

### smoke_hisense_2024_operating_cost

- question: 海信家电2024年合并口径的营业成本是多少？
- metric_name: 营业成本
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [116]
- gold_printed_pages: [114]
- verified_value: 73,476,062,734.50
- unit: CNY
- source_table: 3、合并利润表
- column_label: 2024年度
- evidence_excerpt: 其中：营业成本……73,476,062,734.50
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取营业总成本88,811,666,955.56，也不取母公司营业成本5,348,628,310.54。

### smoke_hisense_2024_parent_net_profit

- question: 海信家电2024年归属于母公司股东的净利润是多少？
- metric_name: 归属于母公司股东的净利润
- statement_type: income_statement
- statement_scope: consolidated
- gold_pdf_pages: [117]
- gold_printed_pages: [115]
- verified_value: 3,347,881,773.89
- unit: CNY
- source_table: 3、合并利润表
- column_label: 2024年度
- evidence_excerpt: 归属于母公司股东的净利润……3,347,881,773.89
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取净利润5,126,152,974.24，也不取少数股东损益1,778,271,200.35。

### smoke_hisense_2024_total_assets

- question: 海信家电2024年末合并口径的资产总计是多少？
- metric_name: 资产总计
- statement_type: balance_sheet
- statement_scope: consolidated
- gold_pdf_pages: [113]
- gold_printed_pages: [111]
- verified_value: 69,701,939,817.33
- unit: CNY
- source_table: 1、合并资产负债表
- column_label: 2024年12月31日·期末余额
- evidence_excerpt: 资产总计……69,701,939,817.33
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取期初余额65,946,495,555.67，也不取母公司资产总计12,432,542,331.13。

### smoke_hisense_2024_accounts_receivable

- question: 海信家电2024年末合并口径的应收账款是多少？
- metric_name: 应收账款
- statement_type: balance_sheet
- statement_scope: consolidated
- gold_pdf_pages: [112]
- gold_printed_pages: [110]
- verified_value: 10,480,609,898.16
- unit: CNY
- source_table: 1、合并资产负债表
- column_label: 2024年12月31日·期末余额
- evidence_excerpt: 应收账款……10,480,609,898.16
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取期初余额9,225,321,882.07，也不取母公司应收账款603,535,442.54。

### smoke_hisense_2024_ocf

- question: 海信家电2024年经营活动产生的现金流量净额是多少？
- metric_name: 经营活动产生的现金流量净额
- statement_type: cash_flow_statement
- statement_scope: consolidated
- gold_pdf_pages: [120]
- gold_printed_pages: [118]
- verified_value: 5,132,164,941.24
- unit: CNY
- source_table: 5、合并现金流量表
- column_label: 2024年度
- evidence_excerpt: 经营活动产生的现金流量净额……5,132,164,941.24
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取经营活动现金流入小计81,425,983,334.68，也不取母公司现金流量净额175,682,379.54。

### smoke_hisense_2024_inventory_goods_write_down

- question: 海信家电2024年末库存商品跌价准备余额是多少？
- metric_name: 库存商品跌价准备
- statement_type: note
- statement_scope: consolidated
- gold_pdf_pages: [167]
- gold_printed_pages: [165]
- verified_value: 62,959,633.32
- unit: CNY
- source_table: 财务报表附注五、9.存货
- column_label: 库存商品·跌价准备·年末余额
- evidence_excerpt: 库存商品……跌价准备62,959,633.32
- verification_status: verified_against_source_pdf
- human_review_status: completed
- note: 不取全部存货跌价准备合计129,619,002.64，也不取库存商品年初跌价准备55,315,902.93。

---

## 5. Audit Summary

| Report | Case count | Source verified | Human review completed | Human review remaining |
|---|---:|---:|---:|---:|
| midea_group_2024 | 8 | 8 | 8 | 0 |
| hisense_home_2024 | 8 | 8 | 8 | 0 |
| Total | 16 | 16 | 16 | 0 |

## 6. Hisense Final Spot-check Checklist

Before freezing the 16-case dataset, the project owner should spot-check:

- [x] PDF 112 / printed 110: 应收账款、存货及期末余额列；
- [x] PDF 113 / printed 111: 资产总计及期末余额列；
- [x] PDF 116 / printed 114: 营业收入、营业成本及2024年度列；
- [x] PDF 117 / printed 115: 归属于母公司股东的净利润；
- [x] PDF 120 / printed 118: 经营活动产生的现金流量净额；
- [x] PDF 167 / printed 165: 库存商品跌价准备年末余额；
- [x] JSONL中的 `gold_pdf_pages` 与本审计文件一致；
- [x] 所有主报表问题均使用合并口径，未混入母公司列。

