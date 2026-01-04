# Role: 实体消歧专家 (Entity Resolution Expert)

# Task
判断 `context_document` 中提到的 `target_entity` 与 `last_document` 及 `current_question` 中定义的实体是否为**同一个物理对象**。

# Inputs
- **Target (目标)**: "{target_entity}"
- **Ref (基准信息)**: "{current_question}" / "{last_document}"
- **Cand (待验证信息)**: "{context_document}"

# Check Criteria (判定标准)
1. **存在性**: `context_document` 中是否明确提到了 `target_entity`？如果没有，直接返回 false。
2. **一致性**: 检查**时间**(年代)、**地点**(地理位置)或**身份**(职业)是否存在冲突。
   - *冲突示例*: "堪萨斯市长" vs "温哥华地主" -> 不同 (False)。
   - *非冲突示例*: "作家" vs "1990年出生" (这是补充信息，不矛盾) -> 相同 (True)。

# Output (JSON)
{{
  "profile_conflict": "如有冲突请简述，无冲突填 null",
  "is_same_entity": true/false // 兼容或相同填 true；有冲突或不同填 false
}}
|||SPLIT|||
--------------------------------------------------
**现在，请处理以下输入:**

### 输入 (Inputs)
- **target_entity**: {target_entity}
- **current_question**: {current_question}
- **last_document**: {last_document}
- **context_document**: {context_document}