# 模型下载

```shell
conda create -n qwen25vl python=3.11 -y
conda activate qwen25vl

pip install torch torchvision torchaudio
pip install transformers accelerate huggingface_hub

huggingface-cli --help #检查是否安装好
```

模型下载有三个路径分别是下面两个和魔搭社区（这个自行搜索即可）

## huggingface-cli下载模型

参考网址：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

```shell
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir 你的路径
```

## 代码下载模型

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="./data",
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=1,
    token="hf_xxxxxxxxxxxxxxxxx"  #你自己的huggingface的token
)
```

# 数据集准备

一些现成的微调数据集：

1) LLaVA-Instruct-150K（入门首选）

内容：约 15 万条 图片 + 对话式指令（human/gpt，多轮也有）。[**https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K**]

优点：最经典、教程最多、适合你之前“任务二”那种单图问答/描述/推理。

注意：通常不直接打包 COCO 原图，你还需要自己下载 COCO 图像集（常见做法）。**https://cocodataset.org/?utm**

ShareGPT4V（量大、描述更细）

内容：官方称 1.2M 级别的高质量 caption/指令混合数据，包含用于 SFT 的混合子集（如 mix665k 等文件说

明）。**https://huggingface.co/datasets/Lin-Chen/ShareGPT4V**

优点：对“细粒度描述、图像理解覆盖面”更强。

注意：同样可能需要你按它的说明把图片组织好（不少是基于公开图像源/COCO等）。**https://huggingface.co/datasets/Lin-Chen/ShareGPT4V**



# 微调

整个过程采用llamafactory 的范式进行微调，自制 50 张小样本数据集（太懒了不想多做了。数据集采用Alpaca 的格式进行。采用 QLoRA 微调

参考网址

\# 安装

https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html

\# 训练（SFT）

https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html

\# 推理（chat/webchat）

https://llamafactory.readthedocs.io/en/latest/getting_started/inference.html

\# 合并导出 LoRA（关键：不要量化）

https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html



```shell
首先按照上面网页指导安装lamafactory，随后进行微调
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train qlora_sft.yaml  #进行推理

lamafactory-cli export merge_lora.yaml #合并 lora

CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat     --model_name_or_path /data1/gkm/qwen_ft/model/qwen25_vl     --adapter_name_or_path saves/qwen25_vl/qlora/sft     --template qwen2_vl     --finetuning_type lora  #网页在线推理
```

