from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
import os
from datetime import datetime

print(torch.cuda.is_available())
torch.device('cuda')

# startTime = datetime.now()
# # News/Text input here
# train_text = ["It wasn’t just Tom Brady and Gisele Bündchen. \
# The roster of high-profile investors who lost money betting on crypto exchange FTX also included New England Patriots owner Robert Kraft and billionaire hedge fund manager Paul Tudor Jones, according to court filings released late Monday. \
# Sam Bankman-Fried’s well-documented success at raising money and charming investors extended to a more expansive set of celebrity investors and big-name financers than was previously disclosed. FTX went through four fundraising rounds to reach a $32 billion valuation by early last year, before ultimately spiraling into bankruptcy in November.\
# Bankman-Fried, FTX’s co-founder and former CEO, has pleaded not guilty to multiple criminal charges, including fraud and money laundering. In December, he was released on a $250 million bond while awaiting trial.\
# For venture backers, FTX represents a loss of historic proportions. Sequoia Capital said in November that it had marked its investment of over $210 million down to zero. Before former equity holders can begin trying to recoup any of their investment, customers face a long road to recovery as the bankruptcy process winds its way through court and across dozens of jurisdictions.\
# FTX’s venture investors included a host of luminaries. Dan Loeb controlled over 6.1 million preferred shares through Third Point-connected venture funds. Rival exchange Coinbase\
# held nearly 1.3 million preferred shares. Jones, the founder of Tudor Investment, apparently owned shares through a series of family trusts. Kraft controlled 155,144 shares of preferred stock through previously undisclosed investments in FTX.\
# Brady, who at age 45 is the winningest quarterback in National Football League history, was a known FTX backer and a pitchman for the company. He held common stock in the company alongside Bündchen. The celebrity couple announced their divorce in October after 13 years of marriage.\
# CNBC has compiled and analyzed the following preferred share ownership using Delaware bankruptcy court filings.\
# Despite being called a Series B raise, this July 2021 fundraising round was FTX’s first infusion of outside capital, excluding an early investment from Binance that was ultimately wound down. Investors included Paradigm and Sequoia, as well as Thoma Bravo and Third Point. The $900 million round valued FTX at $18 billion.\
# Jones, who told CNBC in October 2022 that his bitcoin exposure was “minor,” appears to have invested in FTX through a series of family trusts.\
# Just months later, FTX closed a funding round for $420 million, which included many of the original Series B backers. The investor list expanded to include previously undisclosed capital from Alibaba\
# co-founder Joe Tsai’s family office, Blue Pool, among others.\
# As FTX and Bankman-Fried spent hundreds of millions of dollars on advertising deals and sponsorships, the company continued to seek venture money at a voracious pace. In January 2022, FTX closed its $400 million Series C round at a valuation of $32 billion.\
# FTX, which was based in the Bahamas, created FTX US in response to U.S. regulations on cryptocurrency trading. Regulators have since alleged that FTX US was separated from the international arm of FTX in name only.\
# In trying to establish its independence, FTX US closed a $400 million funding round in January 2022 from investors including Singapore sovereign wealth fund Temasek and Masayoshi Son’s SoftBank Vision Fund. Previously undisclosed venture backers for the round included Kraft and Daniel Och’s family office, Willoughby Capital.\
# According to bankruptcy filings and regulatory complaints, funds and customer assets moved freely among the FTX entities. Despite being partially regulated by the Commodity Futures Trading Commission, FTX US clients face an equally arduous process in bankruptcy court to try and retrieve some of their money.\
# Equity investors in FTX US, like those in FTX, are staring at a zero.\
# "]

# ckpt_path = "results\checkpoint-1000"

# def prepare_model(ckpt_pth):
#     """
#     Prepares Model from Fine-Tuned Checkpoint 
#     """
#     torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print("Device :", torch_device)
#     model = PegasusForConditionalGeneration.from_pretrained(ckpt_pth).to(torch_device)
#     tokenizer = PegasusTokenizer.from_pretrained(ckpt_pth)
#     return model, tokenizer, torch_device

# def model_inference(train_text, model, tokenizer, torch_device):
#     """
#     Run Summariser 
#     """
#     batch = tokenizer.prepare_seq2seq_batch(src_texts=train_text, return_tensors="pt").to(torch_device)
#     gen = model.generate(**batch)
#     res = tokenizer.batch_decode(gen, skip_special_tokens=True)
#     return res


# model, tokenizer, torch_device = prepare_model(ckpt_path)
# res = model_inference(train_text, model, tokenizer, torch_device)
# print(res)
# print("======================================")
# print(datetime.now() - startTime)