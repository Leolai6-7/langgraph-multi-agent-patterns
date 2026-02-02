# LangGraph Multi-Agent Patterns 技術報告

## 目錄

1. [專案總覽](#1-專案總覽)
2. [Pattern 1：Confidence-Weighted Voting（信心加權投票）](#2-pattern-1confidence-weighted-voting信心加權投票)
3. [Pattern 2：Strategy 1 — GCR（生成-批評-修正）](#3-pattern-2strategy-1--gcr生成-批評-修正)
4. [Pattern 3：Strategy 2 — Debate（辯論式修正）](#4-pattern-3strategy-2--debate辯論式修正)
5. [Pattern 4：Strategy 3 — Reflexion（反思學習）](#5-pattern-4strategy-3--reflexion反思學習)
6. [Pattern 5：Strategy 4 — MCTSr（蒙地卡羅樹搜尋）](#6-pattern-5strategy-4--mctsr蒙地卡羅樹搜尋)
7. [Pattern 6：Supervisor v1（主管路由）](#7-pattern-6supervisor-v1主管路由)
8. [Pattern 7：Supervisor v2（白皮書撰寫團隊）](#8-pattern-7supervisor-v2白皮書撰寫團隊)
9. [跨 Pattern 設計模式總結](#9-跨-pattern-設計模式總結)
10. [Pattern 選型指南](#10-pattern-選型指南)

---

## 1. 專案總覽

本專案實作了 7 種 Multi-Agent 協作模式，涵蓋從簡單的平行投票到複雜的蒙地卡羅樹搜尋，全部基於 LangGraph StateGraph 框架，使用 AWS Bedrock Claude 3 Haiku 作為底層 LLM。

### 架構分類

```
Multi-Agent Patterns
├── 聚合型 (Aggregation)
│   └── Confidence-Weighted Voting    ← 平行獨立、加權匯總
├── 迭代型 (Iterative Refinement)
│   ├── Strategy 1: GCR               ← 生成→批評→修正迴圈
│   ├── Strategy 2: Debate            ← 對抗式辯論
│   ├── Strategy 3: Reflexion         ← 向量記憶 + 反思學習
│   └── Strategy 4: MCTSr             ← 章節級蒙地卡羅取樣
└── 路由型 (Routing)
    ├── Supervisor v1                 ← 純 LLM 路由
    └── Supervisor v2                 ← 混合路由（確定性 + LLM）
```

---

## 2. Pattern 1：Confidence-Weighted Voting（信心加權投票）

### 核心思想

多個 Agent 平行獨立評估同一問題，各自給出選擇與信心分數，最後由 Aggregator 加權匯總得出最終決策。

### 流程圖

```
                        ┌─────────────┐
                   ┌───→│  Optimist   │───┐
                   │    └─────────────┘   │
┌───────┐   ┌─────┴──────┐               │   ┌────────────┐   ┌─────┐
│ START │──→│ Dispatcher  │               ├──→│ Aggregator │──→│ END │
└───────┘   └─────┬──────┘               │   └────────────┘   └─────┘
                   │    ┌─────────────┐   │
                   ├───→│  Skeptic    │───┤
                   │    └─────────────┘   │
                   │    ┌─────────────┐   │
                   └───→│  Analyst    │───┘
                        └─────────────┘
            ─── Fan-out (平行) ───  ─── Fan-in (匯總) ───
```

### State 設計

```python
class Vote(TypedDict):
    agent_name: str       # "optimist" | "skeptic" | "analyst"
    choice: str           # 投票選擇
    confidence: float     # 0.0 ~ 1.0 信心分數
    reasoning: str        # 理由

class VotingState(TypedDict):
    query: str
    votes: Annotated[list[Vote], operator.add]  # 累積投票
    final_decision: str                          # 最終決策
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **Fan-out / Fan-in** | Dispatcher 平行派發，Aggregator 匯總 |
| **Structured Voting** | 每票帶信心分數，不是簡單多數決 |
| **角色多樣性** | 樂觀/懷疑/分析三種視角，降低群體盲點 |
| **容錯機制** | JSON 解析失敗時，降級為低信心 "uncertain" 票 |

---

## 3. Pattern 2：Strategy 1 — GCR（生成-批評-修正）

### 核心思想

經典的 Generator-Critic-Refiner 迴圈。Critic 從多個維度評估，Refiner 根據批評修改，Evaluator 打分決定是否達標。

### 流程圖

```
┌───────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐
│ START │───→│Generator│───→│ Critic  │───→│  Refiner  │
└───────┘    └─────────┘    └────▲────┘    └─────┬─────┘
                                 │               │
                                 │               ▼
                            ┌────┴────┐    ┌───────────┐
                            │Increment│←───│ Evaluator │
                            │ iter++  │    └─────┬─────┘
                            └─────────┘          │
                           (iter < max)    (score ≥ threshold)
                                                 │
                                                 ▼
                                           ┌──────────┐    ┌─────┐
                                           │ Finalize │───→│ END │
                                           └──────────┘    └─────┘
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **結構化批評** | Critique 包含 logic / tone / evidence / summary 四個維度 |
| **分數閾值控制** | score_threshold 決定品質底線，max_iterations 防止無限迴圈 |
| **修正歷史** | revision_history 保留每一版草稿，可回溯 |

---

## 4. Pattern 3：Strategy 2 — Debate（辯論式修正）

### 核心思想

引入對抗角色：Challenger 永遠攻擊，Author 可以反駁或接受，Judge 裁決。透過角色扮演實現深度批評。

### 流程圖

```
┌───────┐    ┌───────────┐    ┌────────────┐
│ START │───→│ Generator │───→│ Challenger │←──────────────┐
└───────┘    └───────────┘    └──┬────┬────┘               │
                                 │    │                    │
                          AGREE  │    │ 不同意              │
                                 │    ▼                    │
                                 │  ┌──────────────────┐   │
                                 │  │ Author (Rebuttal)│   │
                                 │  │ 反駁或接受+修改   │   │
                                 │  └────────┬─────────┘   │
                                 │           │             │
                                 │           ▼             │
                                 │  ┌──────────────┐       │
                                 │  │    Judge     │       │
                                 │  │  中立裁決    │       │
                                 │  └──────┬───────┘       │
                                 │         │               │
                                 │    rounds < max ────────┘
                                 │         │
                                 │    rounds ≥ max
                                 │         │
                                 ▼         ▼
                            ┌──────────┐  ┌─────┐
                            │ Finalize │─→│ END │
                            └──────────┘  └─────┘
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **對抗式改善** | Challenger 永不讚美，只從多角度攻擊 |
| **Author 獨立性** | Author 可以反駁批評，不盲目接受所有修改意見 |
| **完整重生成** | 每次 Rebuttal 輸出完整文章，不做 patch |
| **Judge 中立裁決** | 總結辯論要點，為下一輪提供上下文 |
| **提前終止** | Challenger 回覆 "AGREE" 時立即結束，不浪費輪次 |

### GCR vs Debate 對比

```
GCR:    Critic 客觀評分 → Refiner 照做修改（合作關係）
Debate: Challenger 對抗攻擊 → Author 可反駁（對抗關係）

GCR 適合：需要穩定收斂的場景
Debate 適合：需要深度思辨、避免 echo chamber 的場景
```

---

## 5. Pattern 4：Strategy 3 — Reflexion（反思學習）

### 核心思想

Verbal Reinforcement Learning。Agent 從過去的失敗中學習，透過向量記憶（ChromaDB）實現跨 Session 的經驗累積。

### 流程圖

```
┌───────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐
│ START │───→│ Retrieve Memory │───→│ Grade Relevance │───→│ Generate │
└───────┘    │   (ChromaDB)    │    │   (Self-RAG)    │    └────┬─────┘
             └────────▲────────┘    └─────────────────┘         │
                      │                                         ▼
                      │                                   ┌───────────┐
                      │                                   │ Evaluator │
                      │                                   └──┬─────┬──┘
                      │                                      │     │
                      │                          score < threshold │ score ≥ threshold
                      │                                      │     │
                      │                                      ▼     ▼
                      │                              ┌─────────┐ ┌─────────┐
                      │                              │INTERRUPT│ │INTERRUPT│
                      │                              │ (human) │ │ (human) │
                      │                              └────┬────┘ └────┬────┘
                      │                                   │          │
                      │                                   ▼          ▼
                      │                            ┌───────────┐ ┌──────────┐
                      └────────────────────────────│ Reflector │ │ Finalize │
                           persist to ChromaDB     │  反思學習  │ └────┬─────┘
                                                   └───────────┘      │
                                                                      ▼
                                                                   ┌─────┐
                                                                   │ END │
                                                                   └─────┘
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **向量記憶** | ChromaDB 持久化儲存，跨 Session 累積經驗 |
| **Self-RAG** | LLM 評估每條記憶與當前任務的相關性，過濾噪音 |
| **語義去重** | 相似度 > 92% 的反思不重複存入 |
| **Utility 衰減** | 記憶的 utility_score 隨時間衰減，低分記憶被清除 |
| **記憶壓縮** | 記憶過多時，LLM 整合為 3 條核心原則 |
| **Human-in-the-Loop** | Evaluator 後設中斷點，人類可介入修正方向 |

### 記憶機制實作細節

#### 基礎設施：Singleton 向量庫

```
ReflectionVectorStore（Singleton + 執行緒安全）
├── ChromaDB PersistentClient    ← 本地持久化 (vector_store/reflexion/)
├── Bedrock Titan Embed v2       ← 文字轉向量
└── Collection                   ← cosine 距離，HNSW 索引
```

- 使用 `amazon.titan-embed-text-v2:0` 做 embedding
- ChromaDB 以 cosine 距離建立 HNSW 索引
- Singleton 模式 + `threading.Lock` 確保多執行緒安全
- 每筆記憶帶 metadata：`task_type`, `topic`, `score`, `timestamp`, `iteration`, `criteria_hash`, `utility_score`, `consolidated`

#### 記憶生命週期

```
                    ┌──────────────────────────────────────────────┐
                    │              記憶生命週期                      │
                    └──────────────────────────────────────────────┘

1. 誕生（Reflector 寫入）
   ┌─────────────┐     ┌───────────┐     ┌──────────────┐
   │ LLM 生成反思 │────→│ 語義去重   │────→│ 寫入 ChromaDB │
   └─────────────┘     │ sim≥0.92? │     │ utility=1.0  │
                        │ → 跳過    │     └──────────────┘
                        └───────────┘

2. 使用（Retrieve Memory 讀取）
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ 向量相似搜尋  │────→│ Self-RAG     │────→│ 注入 prompt  │
   │ top_k=5      │     │ LLM 逐條評分  │     │ 輔助生成     │
   │ threshold≥0.75│     │ YES/NO 過濾  │     └──────────────┘
   └──────────────┘     └──────────────┘
         │
         └──→ 命中的記憶 utility_score += 0.1（上限 2.0）

3. 老化（每次 Reflector 寫入後觸發維護）
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ 全部記憶      │────→│ utility      │────→│ < 0.3?       │
   │ utility -= 0.05│    │ 衰減完成     │     │ → 刪除       │
   └──────────────┘     └──────────────┘     └──────────────┘

4. 壓縮（記憶數量 > 10 時觸發）
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ 所有舊記憶    │────→│ LLM 壓縮為   │────→│ 刪除舊記憶    │
   │ (> 10 條)    │     │ 3 條核心原則  │     │ 寫入新原則    │
   └──────────────┘     └──────────────┘     └──────────────┘
```

#### 語義去重

寫入前先查詢同 `criteria_hash` 下最相似的既有記憶：

```
新反思 ──embed──→ 查詢 ChromaDB (n_results=1, 同 criteria_hash)
                       │
                  cosine similarity ≥ 0.92 → 跳過寫入
                  cosine similarity < 0.92 → 寫入，utility_score = 1.0
```

避免同一類反思反覆寫入佔滿記憶庫。

#### Self-RAG 過濾

一次 LLM call 批量評估所有檢索到的記憶：

```
輸入：
  主題：{topic}
  標準：{criteria}
  記憶列表：
    1. 反思 A
    2. 反思 B
    3. 反思 C

輸出：["YES", "NO", "YES"]  ← JSON array

結果：只保留 "YES" 的記憶（反思 A、C）注入 Generator prompt
容錯：JSON 解析失敗 → 保守保留全部記憶
```

#### Utility Score 動態管理

```
事件                         utility_score 變化
─────────────────────────    ──────────────────
新反思寫入                    = 1.0（初始值）
被檢索命中                    += 0.1（上限 2.0）
每次維護週期                  -= 0.05（衰減）
低於閾值                      < 0.3 → 刪除

效果：常被用到的記憶越來越強，不再相關的記憶自然淘汰
```

#### 記憶壓縮

當同一 `criteria_hash` 下記憶超過 10 條：

```
LLM prompt：
  "將以下 N 條反思記憶概括合併為 3 條最核心原則"

安全機制：
  1. 先寫入新原則
  2. 寫入成功後才刪除舊記憶
  3. 寫入失敗 → rollback（刪除已寫入的新原則）
  4. 新原則 metadata 標記 consolidated = True
```

#### 維護流程（每次 Reflector 執行後自動觸發）

```
run_maintenance()
  │
  ├── 1. _decay_and_prune()
  │      全部記憶 utility -= 0.05
  │      utility < 0.3 → 刪除
  │
  └── 2. _consolidate()
         記憶數 > 10 → LLM 壓縮為 3 條核心原則
```

---

## 6. Pattern 5：Strategy 4 — MCTSr（蒙地卡羅樹搜尋）

### 核心思想

將文章拆分為章節，每章獨立生成 N 個候選版本，評分後選最佳版本，最後重組。本質是章節級的 Monte Carlo 取樣。

### 流程圖

```
┌───────┐    ┌───────────────────┐
│ START │───→│ Outline Generator │
└───────┘    │  規劃章節結構      │
             └────────┬──────────┘
                      │
              ┌───────▼────────┐
              │ Chapter Sample │←─────────────────┐
              │ 生成 N 個候選   │                  │
              └───────┬────────┘                  │
                      │                           │
              ┌───────▼──────────┐                │
              │Chapter Evaluate  │                │
              │ 評分每個候選      │                │
              └───────┬──────────┘                │
                      │                           │
              ┌───────▼──────────┐                │
              │ Chapter Select   │                │
              │ 選最高分候選      │                │
              └──┬───────────┬───┘                │
                 │           │                    │
           還有章節        全部完成                │
                 │           │                    │
                 └───────────┼────────────────────┘
                             │
                     ┌───────▼────────┐
                     │   Reassemble   │
                     │  重組所有章節   │
                     └───────┬────────┘
                             │
                     ┌───────▼────────┐    ┌─────┐
                     │    Finalize    │───→│ END │
                     └────────────────┘    └─────┘
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **章節級 MCTS** | 每個章節視為獨立的最佳化子問題 |
| **N 候選取樣** | 每章生成 N 個版本（預設 3），增加探索多樣性 |
| **上下文繼承** | 每章生成時帶入前面已選定的章節，確保連貫性 |
| **純選擇策略** | 選最高分候選，不做候選合併 |

---

## 7. Pattern 6：Supervisor v1（主管路由）

### 核心思想

Supervisor 作為純路由器，讀取公共 messages 後透過 structured_output 決定下一個執行的 Worker，實現訊息隔離（公共區 vs 私有區）。

### 流程圖

```
                    ┌─────────────────┐
         ┌────────→│   Supervisor    │←────────┐
         │         │  (LLM 路由)     │         │
         │         └──┬─────┬─────┬──┘         │
         │            │     │     │            │
         │   Researcher  Writer  FINISH        │
         │            │     │     │            │
         │            ▼     │     ▼            │
         │   ┌────────────┐ │  ┌─────┐        │
         └───│ Researcher │ │  │ END │        │
             └────────────┘ │  └─────┘        │
                            ▼                  │
                   ┌────────────┐              │
                   │   Writer   │──────────────┘
                   └────────────┘

START ──→ Supervisor（每次回到 Supervisor 重新判斷）
```

### State 設計

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # 公共區
    next: str                                              # 路由決策
    scratchpad: dict                                       # 私有區
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **訊息隔離** | messages（公共）vs scratchpad（私有），防止 Supervisor 被原始資料淹沒 |
| **純 Prompt 路由** | Supervisor 只輸出路由決策，不寫任何 messages，零 Token 浪費 |
| **壓縮對話** | 把所有 messages 壓成單一文字塊再送 LLM，避免 role alternation 問題 |

---

## 8. Pattern 7：Supervisor v2（白皮書撰寫團隊）

### 核心思想

在 v1 基礎上加入 Editor 角色形成審稿迴圈。核心創新是**混合路由**：確定性轉移用 code 寫死（零 LLM 成本），只有 Editor 意見的解讀才呼叫 LLM。

### 流程圖

```
                     ┌───────────────────┐
          ┌─────────→│    Supervisor     │←──────────┐
          │          │                   │           │
          │          └─┬───┬───┬───┬────┘           │
          │            │   │   │   │                │
          │          code code code LLM             │
          │            │   │   │   │                │
          │            ▼   │   │   │                │
          │   ┌────────────┐   │   │                │
          │   │ Researcher │   │   │  ┌── 資料不足 →─┘ (回 Researcher)
          │   └────────────┘   │   │  │
          │            ▲       ▼   │  │
          │            │  ┌────────┐  │
          │            │  │ Writer │  │  ┌── 寫作問題 →─┘ (回 Writer)
          │            │  └────────┘  │  │
          │            │       ▲      ▼  │
          │            │       │  ┌────────┐
          │            │       │  │ Editor │
          │            │       │  └───┬────┘
          │            │       │      │
          │            │       │   LLM 判斷意見
          │            │       │   ┌──┼──┐
          │            │       │   │  │  │
          │         Researcher │ Writer │ FINISH
          │            │       │      │    │
          │            └───────┘      │    ▼
          │                           │  ┌─────┐
          └───────────────────────────┘  │ END │
                                         └─────┘
          安全閥：revision_count ≥ 3 ──→ 強制 FINISH

主流程：START → Researcher → Writer → Editor → (LLM 判斷) → ...
確定性路由（code）：初始→R, R→W, W→E
LLM 路由：只在 Editor 完成後判斷意見內容
```

### State 設計

```python
class WhitepaperState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    last_actor: str    # 追蹤上一個完成的 Worker
    scratchpad: dict   # topic, research_data, current_draft,
                       # editor_critique, revision_count
```

### 設計模式重點

| 模式 | 說明 |
|------|------|
| **混合路由** | 確定性路由（R→W→E）用 code；只有 Editor→??? 用 LLM 判斷 |
| **last_actor 狀態追蹤** | 不靠 LLM 從文字推斷「上一個是誰」，用 code 精確追蹤 |
| **孤獨的專才** | Worker 不知道團隊存在，prompt 只定義：輸入→輸出→不做什麼 |
| **Worker 名片** | 給 Supervisor 看的描述：能力 + 觸發條件 + 邊界 |
| **reasoning 先於 next** | structured_output 欄位順序：先 reasoning 再 next，強制 chain-of-thought |
| **安全閥** | revision_count ≥ 3 強制結束，防止無限迴圈 |

### Supervisor v1 vs v2 對比

```
v1: 每次都呼叫 LLM 做路由（全部轉移都靠 LLM 判斷）
    問題：LLM 被舊 messages 混淆，出現 Writer→Writer、Editor→Editor

v2: 確定性路由 + LLM 判斷（只在分岔點用 LLM）
    結果：零次重複，零次錯路
    額外好處：確定性路由不花 token
```

---

## 9. 跨 Pattern 設計模式總結

### 9.1 訊息隔離 (Context Isolation)

**出現於：** Supervisor v1, v2

```
messages（公共區）          scratchpad（私有區）
─────────────────          ──────────────────
Supervisor 讀取            Supervisor 不碰
一句話狀態摘要              原始搜尋結果、完整草稿
低 Token 消耗              高 Token 內容
```

**為什麼重要：** 防止路由器被大量原始資料淹沒，保持決策品質。

### 9.2 孤獨的專才 (Lonely Specialist)

**出現於：** Supervisor v2

```
Worker 知道的：我是誰、我的輸入、我的輸出、我不做什麼
Worker 不知道的：團隊存在、上下游是誰、流程順序
```

**為什麼重要：** 避免 Worker 在輸出中加入給隊友的留言，防止上下文污染。

### 9.3 Worker 名片 (Capability Card)

**出現於：** Supervisor v2

```
給 Worker 自己看的 prompt：      給 Supervisor 看的名片：
  輸入 → 輸出 → 不做什麼          能力 + 觸發條件 + 邊界
  約束行為（別越界）               輔助路由（什麼情況派誰）
```

**為什麼重要：** 兩套獨立描述，各自服務不同目的。寫得越像「法律條文」，路由越精準。

### 9.4 混合路由 (Hybrid Routing)

**出現於：** Supervisor v2

```
確定性路由（code）：R→W, W→E          零 LLM 成本，不會出錯
LLM 判斷：Editor→???                  只在分岔點花費 token
```

**為什麼重要：** 能用 code 解決的不要用 LLM，省 token、更可靠。

### 9.5 結構化輸出 + Reasoning 先行

**出現於：** Supervisor v1, v2, Voting

```python
class RouteResponse(BaseModel):
    reasoning: str   # ← 先生成理由（chain-of-thought）
    next: Literal[...]  # ← 再做決定
```

**為什麼重要：** 欄位順序決定生成順序，reasoning 在前強制先思考再決策。

### 9.6 安全閥 (Safety Valve)

**出現於：** Supervisor v2, GCR, Debate, Reflexion, MCTSr

```
所有迭代型 Pattern 都有終止條件：
  - max_iterations / max_debate_rounds
  - revision_count >= 3
  - score >= threshold
  - Challenger 回覆 "AGREE"
```

**為什麼重要：** LLM 的輸出不可預測，沒有安全閥的迴圈可能永遠不收斂。

### 9.7 對抗式改善 (Adversarial Refinement)

**出現於：** Debate

```
GCR:    Critic ──合作──→ Refiner（Refiner 照做）
Debate: Challenger ──對抗──→ Author（Author 可反駁）
```

**為什麼重要：** 合作式改善容易收斂到局部最佳解；對抗式能跳出 echo chamber。

### 9.8 向量記憶 + Self-RAG

**出現於：** Reflexion

```
寫入：Reflector → 語義去重 → ChromaDB
讀取：Retrieve → Self-RAG 過濾 → 注入 prompt
維護：Utility 衰減 → 低分清除 → 過多時壓縮
```

**為什麼重要：** 實現跨 Session 學習，Agent 不重複犯同樣的錯。

---

## 10. Pattern 選型指南

```
需要做決策？
├── 是 → 多個觀點重要嗎？
│         ├── 是 → Confidence-Weighted Voting
│         └── 否 → Supervisor v1（簡單路由）
│
需要產出內容？
├── 簡單的搜尋+撰寫 → Supervisor v1
├── 需要審稿迴圈     → Supervisor v2
├── 需要穩定收斂     → Strategy 1 (GCR)
├── 需要深度思辨     → Strategy 2 (Debate)
├── 需要跨次學習     → Strategy 3 (Reflexion)
└── 長文、多章節     → Strategy 4 (MCTSr)
```

| 場景 | 推薦 Pattern | 理由 |
|------|-------------|------|
| 投資決策、風險評估 | Voting | 多角度獨立判斷 + 信心加權 |
| 短文改善、品質提升 | GCR | 結構化批評 + 分數控制 |
| 論點嚴謹的文章 | Debate | 對抗式批評挖出深層問題 |
| 持續改善的寫作助手 | Reflexion | 記住過去的錯誤 |
| 長篇技術文件 | MCTSr | 章節級最佳化 + 多候選探索 |
| 簡單的搜尋摘要 | Supervisor v1 | 兩個 Worker 就夠 |
| 技術白皮書、報告 | Supervisor v2 | 審稿迴圈確保品質 |

---

*報告產出日期：2026-02-02*
*技術棧：LangGraph, AWS Bedrock Claude 3 Haiku, ChromaDB, DuckDuckGo Search*
