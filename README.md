# MythTriage

This repository contains the code and data accompanying the EMNLP 2025 paper [MYTHTRIAGE: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform](https://arxiv.org/pdf/2506.00308). MythTriage is a scalable pipeline for detecting opioid use disorder (OUD) myths on YouTube, enabling large-scale analysis and informing moderation and health interventions.

## Project Overview

MythTriage is designed to automatically evaluate and classify YouTube videos for opioid use disorder myths. The triage pipeline uses a lightweight model (DeBERTa-v3-base) for routine cases and defers harder ones to state-of-the-art, but costlier large language models (GPT-4o) to provide robust, cost-efficient, and high-performing detection of opioid use disorder myths on YouTube. For more details, please read the paper.  

## Repository Structure

```
MythTriage/
├── code/                           # Source code and notebooks
│   ├── analysis/                   # Analysis code
│   ├── data-collection-pipeline/   # YouTube data collection
│   └── labeling-pipeline/          # Content labeling and classification
│       ├── deberta-labeling/       # DeBERTa-based labeling & deferring 
│       └── llm-labeling/           # LLM-based labeling
└── data/                           # Datasets and evaluation results
    ├── gold_standard_datasets/     # Expert-annotated ground truth data
    ├── llm_evaluations/            # LLM evaluation results
    ├── recommendation_results/     # Collected recommendation data & labels
    ├── search_queries/             # Search topics & queries 
    └── search_results/             # Collected search result data & labels
```

## Usage and Documentation
The code and data are organized in their respective directories. Please read their README.md files for the code and data details, including our expert
- **Code Documentation**: See `/code/README.md` for detailed code structure
- **Data Documentation**: See `/data/README.md` for dataset descriptions

### Quick Start for Code
1. **Data Collection**: Use notebooks in `/code/data-collection-pipeline/` to gather YouTube search and recommendation results.
2. **Model Training**: Fine-tune DeBERTa models using `/code/labeling-pipeline/deberta-labeling/deberta-train.py`
3. **Model Inference & Cascade**: Infer videos and cascade harder examples at scale using `/code/labeling-pipeline/deberta-labeling/recommendation-label.ipynb`
4. **LLM Evaluation**: Apply LLM-based labeling with `/code/labeling-pipeline/llm-labeling/gpt-prompting.ipynb`
5. **Analysis**: Run and reproduce our analysis using `/code/analysis/analysis.ipynb`

## Types of Opioid Use Disorder Myths Detected in MythTriage

MythTriage detects and classifies 8 categories of prevalent opioid use disorder myths recognized by major health organizations and validated by clinical experts. Below are links to the lightweight models for each myth:

- **M1:** Agonist therapy or medication-assisted treatment (MAT) for OUD is merely replacing one drug with another [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M1)
- **M2:** People with OUD are not suffering from a medical disease treatable with medication from a self-imposed condition maintained through the lack of moral fiber  [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M2)
- **M3:** The ultimate goal of treatment for OUD is abstinence from any opioid use (e.g., Taking medication is not true recovery)  [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M3)
- **M4:** Only patients with certain characteristics are vulnerable to addiction [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M4)
- **M5:** Physical dependence or tolerance is the same as addiction [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M5)
- **M6:** Detoxification for OUD is effective [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M6)
- **M7:** You should only take medication for a brief period of time [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M7)
- **M8:** Kratom is a non-addictive and safe alternative to opioids [(LINK)](https://huggingface.co/SocialCompUW/youtube-opioid-myth-detect-M8)

## License

This project is licensed under the MIT License.

## Citation

If you use this project in your research, please consider citing our work:

```bibtex
@misc{jung2025mythtriagescalabledetectionopioid,
      title={MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform}, 
      author={Hayoung Jung and Shravika Mittal and Ananya Aatreya and Navreet Kaur and Munmun De Choudhury and Tanushree Mitra},
      year={2025},
      eprint={2506.00308},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2506.00308}, 
}
```
