import pandas as pd
import json

# Load file của bạn
df = pd.read_csv('soc_dataset_preview.csv')

# Chuyển đổi sang JSONL
with open('dataset.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia phân tích SOC Tier 1. Hãy tóm tắt cảnh báo và đề xuất hành động."},
                {"role": "user", "content": row['instruction_preview']},
                {"role": "assistant", "content": row['output_preview']}
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')