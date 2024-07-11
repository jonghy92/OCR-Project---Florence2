# ---------------------------------------------------------------------- #
#
#                       Florence2 OCR Project
#
# ---------------------------------------------------------------------- #

## Libraries
import textwrap
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import pandas as pd
import sys # sys 모듈 로드

# 쓸대없는 Warning log 제거
import warnings
warnings.filterwarnings("ignore")



## System Args - CMD Input 
try:
    input_image = str(sys.argv[1])
except:
    print('\n')
    print("Error : 이미지 파일의 경로를 실행어 뒤에 같이 넣어주세요.")
    print("Ex. python main.py ./test_image/test.jpg")
    print('\n')


# 이미지 파일명 추출
input_nm = input_image.split('/')[2]



## Model Pre Setting 
#model_id = 'microsoft/Florence-2-base' #--- [ ~ 5GB GPU 사용 ] ---#
model_id = 'microsoft/Florence-2-large' #--- [ ~ 12GB GPU 사용 ] ---#
device='cuda:0' # GPU Cuda Setting Pytorch 

# Model
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             cache_dir="./models/Florence_2",
                                             device_map = device,
                                             trust_remote_code=True).eval()

# Processor
processor = AutoProcessor.from_pretrained(#model_id,
                                          "./models/Florence_2/models--microsoft--Florence-2-large/snapshots/15aa04e200389df2ccb00e2eb94d551284e45df1",
                                        # cache_dir="./models/Florence_2",
                                          device_map = device,
                                          trust_remote_code=True)



## 함수 ##

# Florence2 Setting -- def I
def florence2(task_prompt, image, text_input=None):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        #max_new_tokens=1024,
        max_new_tokens=4096,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))

    return parsed_answer


# OCR Draw Setting -- def II
def draw_ocr_bboxes(image, prediction):
    """
    Draw OCR BBox
    """
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']

    for box, label in zip(bboxes, labels):
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=2, outline='lime')
        draw.text((new_box[0], new_box[1]-8),
                  "{}".format(label),
                  align="right",
                  fill='red')
        
    return image





## Processing Florence2 OCR
task_prompt = '<OCR_WITH_REGION>'
image = Image.open(f'{input_image}')
print(image)
results_ocr = florence2(task_prompt, image)


## CMD -- 터미널 출력 확인용
print("# ==================================================================== #")
print('\n')

# 이미지 사이즈 저장
print('Input Image Size:', image.size)

# OCR 데이터프레임 ( Top to Bottom 순서 ) 저장
df_ocr_rslt = pd.DataFrame(results_ocr['<OCR_WITH_REGION>'])
df_ocr_rslt.to_csv(f"./result/OCR_result_{input_nm}.csv")
print("Text: OCR_result.csv File Generated in result Folder ...!")


# OCR 결과 사진에 그리기 및 저장
result_img = draw_ocr_bboxes(image, results_ocr['<OCR_WITH_REGION>'])
# result_img.save("./result/OCR_result.jpg")
result_img.save(f"./result/OCR_result_{input_nm}.jpg")
print("Image: OCR_result.jpg File Generated in result Folder ...!")

print('\n')
print("# ==================================================================== #")
