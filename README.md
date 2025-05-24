# Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation

### Authors: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby Dagher, Haohan Wang

### Abstract:
> <p align="justify">
Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.
</p>

### Dataset:
> We use four public jailbreak benchmarks, including [AdvBench](https://github.com/llm-attacks/llm-attacks), [HarmBench](https://www.harmbench.org/), [JailbreakBench](https://jailbreakbench.github.io/), and [JamBench](https://github.com/Allen-piexl/llm_moderation_attack). In these benchmarks, we choose the harmful behavior inquiries. 

### Getting Started:
#### Prerequisites
> Linux or macOS \
> Python 3.11 \
> openai, vertexai, anthropic, pandas

#### Clone this repo
> ```git clone https://github.com/[user_name]/IntentPrompt.git``` \
> ```cd IntentPrompt```

#### Install dependencies
> For pip users, please type the command: ```pip install -r requirements.txt``` \
> For Conda users, you may create a new Conda environment using: ```conda env create -f environment.yml``` \
> CUDA and cuDNN are not required.

#### Directories
> **config**: the config files. \
> **data**: contain four datasets. \
> **src**: contain scripts. \
	│── **main.py** # main scripts \
	│── **utils.py** # util functions \
	│── **IntentPrompt_Demo.ipynb** # demo

#### Run
> Update the config file with your Vertex AI project ID and/or other LLMs' key. \
> You can run the main script under "src" by ```python main.py --config_file model```

#### Competing methods
> NeurIPS'23 [[PAIR](https://github.com/patrickrchao/JailbreakingLLMs)] Jailbreaking Black Box Large Language Models in Twenty Queries. \
> NeurIPS'24 [[TAP](https://github.com/RICommunity/TAP)] Tree of Attacks: Jailbreaking Black-Box LLMs Automatically \
> ACL'24 [[ArtPrompt](https://github.com/uw-nsl/ArtPrompt)] ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs \
> ICLR'24 [[CipherChat](https://github.com/RobustNLP/CipherChat)] GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher \
> ICML'25 [[FlipAttack](https://github.com/yueliu1999/FlipAttack)] FlipAttack: Jailbreak LLMs via Flipping \
> NeurIPS'24 SafeGenAI workshop [[Deepinception](https://github.com/tmlr-group/DeepInception)] Deepinception: Hypnotize large language model to be jailbreaker \
> NAACL'24 [[ReNeLLM](https://github.com/NJUNLP/ReNeLLM)] A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily \
> ICASSP'24 [[FuzzLLM](https://github.com/RainJamesY/FuzzLLM)] FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models

#### Results
> We present our main results in the Experiment section.
