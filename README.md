# This repository is an implementation of the paper : 
Deep Think with Confidence
https://arxiv.org/abs/2508.15260


In this, we have tried to analyse the dataset : AIME 2025, consisting of 30 questions. 

generate_traces.py : this loads the dataset, and the Qwen 3 Deepseek model, and generate 52 traces with 32000 token sequence length and stores in the directory. This is a one time process, which took around 4 hours, on a 2-H100 Nvidia GPU. This will dump the raw traces for all the questions, which can be analysed later.

main.py : this takes all the generated traces and analyse them to see the accuracy of the reasoning traces. It also prints the mean confidence vs token progress.

Pasting below three images, 
-the first one for a typical question where most of the traces solved it easily.
-the second one, where only few proportion of traces solved it, the rest didn't solve. The thing to note here is: we can clearly distinguish the correct answer based on the confidence at tail values. This is probably what the author had analysed and suggested a similar approach of looking at tail confidence to improve accuracy. 
-the third one, where none of the traces gave right answer. one thing to note here is the absolute confidence is relatively low throughout and possibly some metric which captures this could at-least have highlighted it, and maybe use this to generate more traces, or append the prompt to possibly reach a correct answer.
also, check results folder for all questions plots.
