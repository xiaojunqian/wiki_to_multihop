# Role: 文本重写专家 (Text Rewriting Expert)

# Task
仅基于 `context_document`，为 `target_entity` 生成一个具体的**名词短语**。
要求这个**名词短语**能够准确地作为`target_entity` 的同义词。

# Inputs
- **Target (被描述者)**: "{target_entity}"
- **Bridge (必须包含)**: "{occupy_entity}"
- **Context (上下文)**: "{context_document}"

# 严格约束
1. **必须包含桥梁**: 生成的短语中必须包含 "{occupy_entity}"。(Exception: If "{occupy_entity}" is empty, this rule is ignored.)
2. **必须脱敏**: 短语中**绝对不能**出现 "{target_entity}" 的名字。
3. **格式要求**: 仅得到一个名词短语（例如：“位于法国南部的并且不是马赛的那个港口城市”）
4. **防止泄露**: 不要包含能直接泄露问题答案的信息。
5. **逻辑自指判定 (Self-Reference Check)**:
    * 检查生成的描述是否会**直接泄露**问题的答案。
    * *Reject Example*: 问题是“X属于哪个行政区？”，如果你生成的描述是“属于Y行政区的X”，则问题变成了“属于Y行政区的X属于哪个行政区？” -> 这毫无意义，必须拒绝。

6. **排他性原则 (Exclusivity)**: 描述必须足够具体，能够唯一锁定目标，排除其他相似实体。出现**邻****之一**需要着重检查。
7. **可逆性原则**: 描述必须唯一指向`target_entity`且不能指向潜在的其他可能实体。出现**邻****之一**需要着重检查。

**Case 1: 成功的描述替换**
* Input: 
    * current_question: "谁写了关于Bolivarian Games早期历史的书？"
    * target_entity: "Bolivarian Games"
    * occupy_entity: "波多黎各"
    * context_document: "1966年Bolivarian Games运动会在波多黎各举行..."
* Output: 
```json
{{ "generated_description": "1966年在波多黎各举办的那个运动会" }}
```
**Case 2: 拦截同名异义 (Fail Case)**
* Input:
    * current_question: "Robert Knight曾是哪一届堪萨斯市的市长？"
    * target_entity: "Robert Knight"
    * occupy_entity: "Knight Street"
    * context_document: "Knight Street是以加拿大温哥华的早期地主Robert Knight命名的..."
* Output: 
```json
{{"generated_description": ""}}
```
**Case 3: 拦截答案泄露 (Fail Case)**
* Input:
    * current_question: "Starochęciny属于哪个行政区？"
    * target_entity: "Starochęciny"
    * occupy_entity: "Gmina Chęciny"
    * context_document: "Starochęciny是Gmina Chęciny行政区下的一个村庄..."
* Output: 
```json
{{"generated_description": "" }}
```
**输出格式**
# Output (JSON)
{{
  "generated_description": "在此处填入生成的名词短语（或空）"
}}
|||SPLIT|||
--------------------------------------------------
**现在，请处理以下输入:**

### 输入 (Inputs)
- **target_entity**: {target_entity}
- **occupy_entity**: {occupy_entity}
- **context_document**: {context_document}
- **current_question**: {current_question}