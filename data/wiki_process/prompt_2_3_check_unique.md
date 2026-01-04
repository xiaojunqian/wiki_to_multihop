# Role: 逻辑校验员与文字编辑 (Logic Verifier & Editor)

# Task
1. **校验**: 验证生成的 `description` 在 `context_document` 的语境下，是否能**唯一**指向 `target_entity`。
2. **重写**: 如果校验通过，请将 `current_question` 中的 `target_entity` 替换为该 `description`，并**修复语序和语法**，使其自然流畅。

# Inputs
- **Target**: "{target_entity}"
- **Description**: "{generated_description}"
- **Context**: "{context_document}"
- **Current Question**: "{current_question}"

# Checklist (检查清单)
1. **唯一性 (Uniqueness)**: 是否有其他实体也符合该描述？(如"创始人"是否指代了多人？) -> 必须唯一。
2. **事实性 (Factuality)**: 描述的内容是否由文档支持？

# Output (JSON)
{{
    "reason": "简述校验原因",
    "is_valid": true, // 只有当 唯一、符合事实 且 已脱敏 时填 true
    "rephrased_question": "..." // 如果 is_valid 为 true，这里填入替换并润色后的完整新问题；如果 false，填 null
}}
|||SPLIT|||
--------------------------------------------------
**现在，请处理以下输入:**

### 输入 (Inputs)
- **target_entity**: {target_entity}
- **generated_description**: {generated_description}
- **context_document**: {context_document}
- **current_question**: {current_question}