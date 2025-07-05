#!/usr/bin/env python
# coding: utf-8
"""
@title: Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation.
@topic: Implement intent-aware jailbreak framework.
@author: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby Dagher, Haohan Wang.
@instructions for using Vertex AI:
    gcloud init
    gcloud auth application-default login
"""

# import libraries
import time
import argparse
import pandas as pd
from utils import (
    read_yaml_file, subsample_data,
    get_prompts_paraphrase, get_prompts_inquiry, get_prompts_evaluation, IA,
    build_genai_model, generate_content, printf, compute_asr, compute_success_hscore_mean,
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='advbench_behaviors',
                    choices=['advbench_behaviors', 'harmbench_behaviors', 'jailbreakbench_behaviors', 'jambench_behaviors'],
                    help='The name of dataset.')
parser.add_argument('--paraphraser_name', type=str, default='gemini-1.5-flash-002',
                    choices=['gemini-1.5-flash-002', 'qwen/qwen3-14b:free',' mistralai/mixtral-8x7b-instruct'],
                    help='The name of the auxiliary agent.')
parser.add_argument('--target_name', type=str, default='gpt-4o',
                    choices=['gpt-4o', 'gpt-4o-mini-2024-07-18', 'o1', 'o1-mini', 'o3-mini',
                             'gemini-2.0-flash-001', 'deepseek-chat', 'deepseek-reasoner',
                             'claude-3-7-sonnet-20250219', 'claude-3-haiku-20240307',
                             'qwen/qwen3-235b-a22b:free','meta-llama/llama-4-scout:free'],
                    help='The name of the victim model.')
parser.add_argument('--evaluator_name', type=str, default='gemini-1.5-flash-002',
                    choices=['gemini-1.5-flash-002', 'gpt-4o-mini-2024-07-18', 'qwen/qwen3-32b:free'],
                    help='The name of the monitoring agent.')
parser.add_argument('--para_types', type=str, default='none', choices=['none', 'ass', 'msw', 'ces'], help='The type of paraphrasing used.')
parser.add_argument('--is_fuzzy', type=str, default='', choices=['', 'NO'], help='Whether or not to obscure the intent.')
parser.add_argument('--para_mode', type=str, default='fuzzy_struct', choices=['revise', 'struct', 'fuzzy_struct'], help='The mode of paraphrasing.')
parser.add_argument('--inq_mode', type=str, default='spin', choices=['naive', 'ela', 'spin'], help='The mode of inquiry.')
parser.add_argument('--is_defense', type=bool, default=False, help='Whether to use intent analysis defense.')
parser.add_argument('--max_tokens', type=int, default=2048, help='The maximum number of tokens for model generation.')
parser.add_argument('--temp', type=float, default=0.8, help='Temperature for sampling.')
parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling.')
parser.add_argument('--num_iter', type=int, default=5, help='Number of iterations for optimizing the prompts.')
parser.add_argument('--vertexai_proj_id', type=str, default='', help='Vertex AI project ID.')
parser.add_argument('--vertexai_region', type=str, default='', help='Vertex AI region (e.g., us-central1).')
parser.add_argument('--openai_key', type=str, default='', help='API key for OpenAI.')
parser.add_argument('--deepseek_key', type=str, default='', help='API key for DeepSeek.')
parser.add_argument('--anthropic_key', type=str, default='', help='API key for Anthropic.')
parser.add_argument('--openrouter_key', type=str, default='', help='API key for OpenRouter.')
parser.add_argument('--config_file', type=str, default='model', help='Optional YAML config file to override arguments.')
args = parser.parse_known_args()[0]

# Load config file if specified
if len(args.config_file) > 0:
    config_file = read_yaml_file("../config", args.config_file)
    main_config = config_file['main']
    args.data_name = main_config.get('data_name', args.data_name)
    args.paraphraser_name = main_config.get('paraphraser_name', args.paraphraser_name)
    args.target_name = main_config.get('target_name', args.target_name)
    args.evaluator_name = main_config.get('evaluator_name', args.evaluator_name)
    args.para_types = main_config.get('para_types', args.para_types)
    args.is_fuzzy = main_config.get('is_fuzzy', args.is_fuzzy)
    args.para_mode = main_config.get('para_mode', args.para_mode)
    args.inq_mode = main_config.get('inq_mode', args.inq_mode)
    args.is_defense = main_config.get('is_defense', args.is_defense)
    args.max_tokens = main_config.get('max_tokens', args.max_tokens)
    args.temp = main_config.get('temp', args.temp)
    args.top_p = main_config.get('top_p', args.top_p)
    args.num_iter = main_config.get('num_iter', args.num_iter)
    args.vertexai_proj_id = main_config.get('vertexai_proj_id', args.vertexai_proj_id)
    args.vertexai_region = main_config.get('vertexai_region', args.vertexai_region)
    args.openai_key = main_config.get('openai_key', args.openai_key)
    args.deepseek_key = main_config.get('deepseek_key', args.deepseek_key)
    args.anthropic_key = main_config.get('anthropic_key', args.anthropic_key)
    args.openrouter_key = main_config.get('openrouter_key', args.openrouter_key)

# Initialize the hyper-parameters and configs
para_dict = {'none': 'taking no specific strategy',
             'ass': 'altering sentence structure',
             'msw': 'misspelling sensitive words',
             'ces': 'changing expression style',
            } # types of paraphrase
pdict = {'inquiries': 'NA',
         'para_strat': para_dict[args.para_types],
         'fuzzy': args.is_fuzzy,
         'para_inq_old': 'NA',
         'feedback': 'NA',
        } # prompts' arguments
openai_models = ['gpt', 'o1', 'o3']
openrouter_models = ['qwen', 'llama', 'mixtral']
api_key_map = {model: args.openai_key for model in openai_models} | {
               model: args.openrouter_key for model in openrouter_models} | {
                'deepseek': args.deepseek_key,
                'claude': args.anthropic_key,
              } # map model to api key                                                                             
url_model_map = {"https://api.openai.com/v1": ['gpt', 'o1', 'o3'],
                 "https://api.deepseek.com/v1": ['deepseek'],
                 "https://openrouter.ai/api/v1": ['qwen', 'llama', 'mixtral'],
                }              
genai_config = {'max_tokens': args.max_tokens,
                'temperature': args.temp,
                'top_p': args.top_p,
                }
# Subsample instances (for debug)
is_sampled = True
num_sample = 5
#vertexai.init(project=vertexai_proj_id, location=vertexai_region)  # init() if necessary

def intent_jailbreak(inquiries_list: list, num_iter: int, verbose: bool) -> list:
    """
    @topic: explore the vulnerability via intent manipulation for red-teaming purposes.
    @inputs:
        inquiries_list (list of strings): a list of text inquiries.
        num_iter (int): the number of iterations.
        verbose (bool): print out the log if verbose=True.
    @returns:
        opt_para_inquiries (list): a list of optimal paraphrased inquiries for jailbreak.
        asr_list (list): a list of attack success rates (ASR) for each inquiry.
        hscore_list (list): a list of harmfulness scores (hscore) for each inquiry.
        q4j_mean (float): the average number of queries used for a successful jailbreak.
    """
    # Initialize multi-agents for jailbreak.
    paraphraser = build_genai_model(args.paraphraser_name, api_key_map, url_model_map) # auxiliary agent
    target_model = build_genai_model(args.target_name, api_key_map, url_model_map) # victim model
    evaluator = build_genai_model(args.evaluator_name, api_key_map, url_model_map) # monitoring agent

    opt_para_inquiries = ['' for _ in range(len(inquiries_list))]
    asr_list, hscore_list = ([0] * len(inquiries_list) for _ in range(2))
    q4j_list = []
    for i, inquiry in enumerate(inquiries_list):
        time.sleep(1)
        printf("-"*20, verbose)
        printf(f"Jailbreaking the {int(i+1)}-th inquiry: {inquiry}", verbose)
        pdict['inquiries'] = inquiry
        num_query4jailbreak = 1 # the number of queries used for a successful jailbreak.
        for n in range(num_iter):
            time.sleep(1)
            printf(f"Paraphrasing the inquiry in {int(n+1)}-th iteration.", verbose)
            # paraphrase the malicious inquiries.
            prompts_para = get_prompts_paraphrase(pdict, args.para_mode)  # update prompts for paraphrase
            try:
                para_inq = generate_content(paraphraser, args.paraphraser_name, genai_config, prompts_para)
            except:
                break # break if fail to paraphrase
            printf(f"Paraphrased inquiries: {para_inq}".strip(), verbose)

            # generate response using the target model with paraphrased inquiries.
            prompts_inq = get_prompts_inquiry(para_inq, args.is_defense, IA, args.inq_mode)  # update prompts for inquiries

            resp = generate_content(target_model, args.target_name, genai_config, prompts_inq)
            printf(f"Response from the target model: {resp}".strip(), verbose)

            # evaluate the jailbreak performance: is successfully jailbroken? (0 or 1), harmfulness score (0-5).
            prompts_eval = get_prompts_evaluation(para_inq, resp)  # update prompts for evaluation
            res_eval = generate_content(evaluator, args.evaluator_name, genai_config, prompts_eval)
            try:
                is_jailbroken, hscore = eval(res_eval)
            except:
                print(f"fail to evaluate the outputs in {n}-th iteration for the paraphrased inquiry '{para_inq}'.")
                is_jailbroken, hscore = 0, 0

            printf(f"Is_Success_Jailbreak: {is_jailbroken}, Harmfulness score: {hscore}.", verbose)
            if is_jailbroken == 0: # update 'para_inq_old' and 'feedback' for the next iteration if not success.
                pdict['para_inq_old'] = para_inq
                pdict['feedback'] = resp
                num_query4jailbreak += 1
            else: # break the current FOR loop if the jailbreak is successful.
                printf(f"Successful jailbreak in {int(n+1)}-th iteration.", verbose)
                opt_para_inquiries[i] = para_inq.strip() if para_inq else ''  # store the optimal para_inquiry for the successful jailbreak.
                asr_list[i] = is_jailbroken
                hscore_list[i] = hscore
                q4j_list.append(num_query4jailbreak) # count the query when jailbreak is successful. (or use 'n+1').
                pdict['para_inq_old'] = 'NA'
                pdict['feedback'] = 'NA'
                break
    q4j_mean = sum(q4j_list)/len(q4j_list) if q4j_list else 0
    return opt_para_inquiries, asr_list, hscore_list, q4j_mean


if __name__ == '__main__':
    # Load data
    infile_path = f'../data/{args.data_name}.csv'
    data_df = pd.read_csv(infile_path, header=None) # read as dataframe
    data_los = data_df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist() # convert to a list of strings
    # Subsample instances (for debug)
    inquiries_list = subsample_data(data_los, is_sampled, num_sample, seed=42)

    # Perform Rad Teaming - explore the vulnerability via intent jailbreak
    opt_para_inquiries, asr_list, hscore_list, q4j_mean = intent_jailbreak(inquiries_list, args.num_iter, verbose=True)

    # Evaluation
    asr = compute_asr(asr_list)
    hscore = compute_success_hscore_mean(asr_list, hscore_list)
    print(f"ASR: {asr*100:.2f}%")
    print(f"Hscore: {hscore:.2f}")
    print(f"The average number of queries used for successful jailbreaks: {q4j_mean:.2f}")
