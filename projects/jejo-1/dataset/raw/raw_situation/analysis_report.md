**Report: Analysis of `dataset/raw/raw_situation` Files as Datasets**

This report summarizes the content and role of the files within the `dataset/raw/raw_situation` directory, focusing on their function as datasets or data-related processing components.

---

**1. `(0212)데이터 유저셋 (json) (1).txt`**
*   **Type**: Mixed - JSON Dataset and Python Scripts
*   **Role as Dataset**: Contains a JSON array defining **5 distinct user profiles**. Each profile includes attributes such as `user_id`, `role_level` (junior, working, senior), `experience_months`, `weak_topics`, `risk_tolerance`, and `preferred_style` (checklist, coaching, blunt, concise). This serves as a foundational dataset for simulating diverse user interactions and tailoring agent responses.
*   **Data-Related Scripts**: Embedded Python code (functions like `pick_meta`, `scenario_text_variants`, and a `main` function) is used for generating scenarios (`scenarios.json` from `mistakes.json`) and for batch evaluation of agent outputs. This highlights its role in both data generation and automated testing.

**2. `(0213) 데이터시나리오 확장 작업 (1).txt`**
*   **Type**: Mixed - Documentation/Plan and Python Class Definition
*   **Role as Dataset/Data Processor**: While primarily a planning document for future data and scenario expansion, it **critically includes the full Python class definition for `MistakePredictAgent`**. This class acts as a central data processing component, designed to consume input data (user queries, RAG results) and generate structured analysis reports. The document also specifies future attributes for mistake data schemas (e.g., `root_cause`, `trigger_pattern`), indicating plans for richer datasets.

**3. `0213 (서비스 검증).txt`**
*   **Type**: Mixed - JSON Dataset and Python Scripts
*   **Role as Dataset**: Contains a **variant JSON array of user profiles**, similar to the first file, further contributing to the repository of user archetypes used for testing.
*   **Data-Related Scripts**: Includes Python code for `scripts/generate_scenarios.py` (generating `scenarios.json` using templates) and `scripts/batch_runner_users.py` (simulating agent interactions with user profiles and scenarios, then evaluating and scoring responses). These scripts are essential for generating test data, customizing agent behavior based on data, and producing evaluation metrics. It also details the logic for constructing dynamic system prompts based on user profiles (`build_user_instruction`).

**4. `가상사례 11 (1).txt`**
*   **Type**: Mixed - CSV-like Data and Python Scripts
*   **Role as Dataset**: Presents a list of **11 virtual case examples**, resembling a CSV data source. Each entry details an `id`, `risk_type`, `supervisor_feedback`, and `expected_loss`, forming a core dataset of known mistake scenarios.
*   **Data-Related Scripts**: Contains Python code for `scripts/csv_to_mistakes_json.py` (converting CSV data to `mistakes.json` for structured use) and `scripts/build_vector_db.py` (building a Chroma vector database with `UpstageEmbeddings` from `mistakes.json`). These are crucial scripts for data preprocessing and building the RAG knowledge base. Another `scripts/batch_runner.py` is included for agent evaluation.

**5. `message (1).txt`**
*   **Type**: Python Script
*   **Role as Data Generation Script**: Contains Python code identical to the scenario generation logic found in `(0212)데이터 유저셋 (json) (1).txt`. Its purpose is to generate various test scenarios based on mistake data.

**6. `message.txt`**
*   **Type**: JSON Dataset
*   **Role as Dataset**: A comprehensive **JSON array of 20 detailed mistake objects**. Each object provides extensive information including `id`, `category`, `situation`, `mistake`, `risk_level`, `description`, `consequences`, `prevention` strategies, `estimated_loss`, `frequency`, and `real_case` examples. This is a rich, structured dataset forming a core knowledge base of trade-related errors.

---

**Overall Summary of `dataset/raw/raw_situation`:**

This directory serves as a central hub for various datasets and data-centric scripts crucial for the development, testing, and evaluation of the trade-ai-agent. It contains:
*   **User Profiles**: JSON data for simulating diverse user interactions.
*   **Mistake Scenarios**: Detailed JSON and CSV-like data describing specific trade mistakes, their impacts, and prevention methods, forming a knowledge base.
*   **Scenario Generation Scripts**: Python code to dynamically create test scenarios.
*   **Data Preprocessing and RAG Building Scripts**: Python tools for converting data formats and establishing vector databases for Retrieval Augmented Generation.
*   **Agent Evaluation Scripts**: Python code for automating the testing of agent responses, incorporating user profiles for personalized evaluation, and generating performance reports.

These files collectively demonstrate a robust framework for managing and leveraging data to build, test, and refine the agent's capabilities in risk management and user-adaptive coaching.