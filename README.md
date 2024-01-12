# sd-turbo-on-mac

```sh
poetry install
poetry run python main.py
```

## メモ

デカいサイズ：
pipe_t2i: StableDiffusionPipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo").to(device)

sd-turboの制約として512x512の画像を出力するため
https://huggingface.co/stabilityai/sd-turbo 

青みがかった
img_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

Image.NEARESTは画像のリサンプリング

# prompt = "beautiful landscape, forest, mountain, sunny, yellow flower, woman in white dress with cat++"
