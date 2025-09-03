# Code Directory

This directory contains the code for the MythTriage project.

## Directory Structure

### `/analysis`
- **`analysis.ipynb`**: Main analysis notebook containing the data analysis and visualizations used in the paper.

---

### `/data-collection-pipeline`
- **`youtube_search_data_pipeline.ipynb`**: YouTube Data API Pipeline for collecting YouTube search results data, including search queries, video metadata, and search result rankings.
- **`youtube_recommendation_data_pipeline.ipynb`**: YouTube InnerTube Pipeline for collecting YouTube recommendation data, extracting video recommendations and their associated metadata.

---

### `/labeling-pipeline`
Contains tools and pipelines for labeling and evaluating YouTube content for 8 opioid use disorder myths.

#### `/deberta-labeling`
- **`deberta-train.py`**: Training script for fine-tuning DeBERTa models on opioid use disorder myth detection tasks.
- **`recommendation-label.ipynb`**: Notebook for applying DeBERTa-based labeling to recommendation data and determining which examples to defer to LLM-based labeling.

#### `/llm-labeling`
- **`gpt-prompting.ipynb`**: Notebook for LLM-based content labeling and evaluation using single prompts.
- **`gpt-batch-prompting-recommendation.ipynb`**: Notebook for batch processing recommendation data using LLMs (e.g., GPT-4o) for labeling.
- **`utils/`**: Utility modules for LLM labeling pipeline.
  - **`EvaluatorHelper.py`**: Helper functions for evaluating and processing LLM outputs.
  - **`GPTRequests.py`**: Utility functions for making requests to OpenAI API endpoints.
  - **`prompts.py`**: Contains (zero-shot and few-shot) prompt templates for both myth-specific and overall stance classification tasks. For each myth category, it provides 5-shot examples (e.g., `{MYTH}_FEW_SHOT_EXAMPLES`) that include video metadata (titles, descriptions, transcripts, tags), labels, and justifications for the assigned labels. To follow best practices for data ethics and user privacy, the metadata fields were removed; instead, the associated video IDs are included as comments above each set of myth-specific examples in this file. Practitioners can use these IDs to rehydrate the few-shot examples. Alternatively, researchers and practitioners can reach out to Hayoung Jung via email (*hayoung@cs.princeton.edu*) with a brief statement of their motivation to request access to these few-shot examples with metadata. 

---


## Usage

Each notebook and script is designed to be run independently based on your specific needs:
- Use the data collection pipelines to gather new YouTube data
- Apply the labeling pipelines to classify content using either DeBERTa models or GPT models
- Run the analysis notebook to evaluate results and generate insights

## Dependencies

The codebase requires Python packages for data processing (pandas, numpy), machine learning (transformers, torch), and API interactions (requests, openai). See individual notebooks for specific import requirements.
