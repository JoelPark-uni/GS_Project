import json
import torch.utils
import torch.utils.data
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import datasets

# 랜덤 시드 설정
seed = 123
torch.manual_seed(seed)

# LLaMA 모델과 토크나이저를 불러옵니다.
EXP_NUM = 4000
five_shot = True

MODEL_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
LORA_ADAPTER_PATH = "/root/axolotl/outputs/MisGendering/checkpoint-908"  # LoRA 어댑터 경로를 지정하세요.
EVAL_RESULT_PATH = f"/root/GS_Project/eval/finetuned_result.json"
TEST_DATASET_PATH = "joel-unist/GenderDisclosureDataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LlamaForCausalLM.from_pretrained(MODEL_CHECKPOINT,
                                        torch_dtype=torch.float16,
                                        attn_implementation="flash_attention_2",
                                        device_map='cuda',
                                        )
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

model.generation_config.pad_token_id = tokenizer.eos_token_id

# LoRA 어댑터를 적용할거면
if LORA_ADAPTER_PATH:
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    print("LoRA 적용 완료")
    
model.to(device)

dataset = datasets.load_dataset(TEST_DATASET_PATH)

test_dataset = dataset["train"].train_test_split()["test"]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# instruction, input, output을 순회하면서 모델에 inference를 수행합니다.
total_length = 0
correct = 0
results = []
for data in tqdm(test_loader):
    instruction = data["instruction"][0]
    input_text = data["input"][0]
    expected_output = data["output"][0]
    
    
    instruction_prompt = f"{instruction}"
    
    # 모델에 입력할 텍스트를 구성합니다.
    full_input = (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response: Answer:"
    )
    
    
    # 입력 텍스트를 토큰화합니다.
    inputs = tokenizer(full_input, return_tensors="pt",).to(device)
    prompt_length = inputs["input_ids"][0].size(0)
    
    # 모델에 입력을 전달하고 출력을 얻습니다.
    with torch.no_grad():
        # 생성 파라미터 설정
        top_k = 50
        top_p = 0.9
        temperature = 0.6
        num_return_sequences = 1
        beam_search = 1  # beam_search가 1이면 beam search 사용 안 함

        # 텍스트 생성
        outputs = model.generate(
            **inputs,
            #max_length=max_length,
            max_new_tokens=1,
            #top_k=top_k,
            #top_p=top_p,
            #temperature=temperature,
            #num_beams=beam_search,
            #do_sample=True  # 확률적 샘플링 사용
        )
    
    # 출력을 디코딩합니다.
    output_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    # output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #substring = "### Response: Answer:"

    # 시작 위치 찾기
    #start_index = output_text.find(substring)
    
    #answer = output_text[start_index+len(substring):]
    
    # 결과를 출력합니다.
    # print(f"Model Output: {output_text}")
    # print("=" * 80)
    
    # 결과를 저장합니다.
    results.append({
        "instruction": instruction,
        "input": input_text,
        "output": expected_output,
        "model_output": output_text
    })
    if expected_output in output_text : 
        correct += 1
    total_length+=1


results.append({
    "total length": total_length,
    "correct" : correct,
    "accuracy" : correct/total_length * 100
})

# 결과를 JSON 파일로 저장합니다.
with open(EVAL_RESULT_PATH, "w", encoding="utf-8") as result_file:
    json.dump(results, result_file, ensure_ascii=False, indent=4)