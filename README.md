
# ENSF 617 – Assignment 2 
## Group 13 

### Group Members 
- Avik
- Deepinder 
- Faiza

---

## 1. Introduction

The city of Calgary required a system that would classify household garbage into one of the following four categories: Green, Blue, Black, and Other using a cellphone image and a short sentence description. The system use a multi-modal classification system to accurately classify waste that is related to Calgary's “what goes where” waste management guidelines. For the image part, we use a pretrained ResNet50 model to understand and extract important visual features from the garbage photo. For the text part, we use a pretrained BERT (bert-base-uncased) model to understand the short description taken from the image filename. The information learned from both the image and the text is then fused and passed to a classifier for final prediction.
