import torch
from transformers import BertTokenizer, BertConfig
from common import constants
from model import BertClassifier


def main():
    model_state_dict = torch.load('./models/best_model.pkl')
    # ----------------模型加载----------------
    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(constants.BERT_PATH)
    labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    num_labels = len(labels)

    device = 'mps'
    model = BertClassifier(bert_config, num_labels).to(device)
    model.load_state_dict(model_state_dict)

    MAX_LEN = 512

    tokenizer = BertTokenizer.from_pretrained(constants.BERT_PATH)
    while True:
        line = input("请输入文章内容\n")
        print(f"输入的内容是 {line}")
        text = line.strip()
        # question: 不确定应该用 encode_plus 方法？老师是否有其他方法推荐?
        encoded_review = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        token_type_ids = encoded_review['token_type_ids'].to(device)
        output = model(input_ids, attention_mask, token_type_ids)
        _, prediction = torch.max(output, dim=1)
        print(f"分类的结果是 <{labels[prediction]}>\n")


if __name__ == '__main__':
    main()
