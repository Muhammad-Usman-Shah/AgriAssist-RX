# AgriAssist-RX

### Agricultural AI Assistant | Python · LangGraph · RAG · Computer Vision

AgriAssist-RX is a Python project that brings agricultural questions, weather information, farm calculations, crop data analysis, and wheat image classification into one application.

I built it to explore how AI agents, document retrieval, and computer vision can work together. Rather than using a separate program for each task, the assistant selects a tool based on the user's question.

The project currently runs in a terminal. It has six working routes, along with tests for tool selection, answer quality, image classification, and error handling.

**Project status:** Working research prototype.

## Features

| Feature | What it does |
|---|---|
| Agricultural RAG | Answers questions using information from wheat disease documents |
| Weather | Checks current weather for supported locations |
| Farm calculator | Calculates fertilizer, seed, irrigation, and crop production quantities |
| CSV analysis | Reads crop data and calculates basic statistics |
| Image classification | Predicts wheat conditions from photographs |
| Request handling | Rejects questions outside the available tools |

The agent chooses a tool automatically. Users only need to enter their question.

## System Architecture

The project uses LangGraph to connect the six routes.

```mermaid
flowchart TD
    A["User question"] --> B["LangGraph agent"]
    B --> C{"Choose tool"}

    C --> D["Agricultural RAG"]
    C --> E["Weather"]
    C --> F["Calculator"]
    C --> G["CSV analysis"]
    C --> H["Image classification"]
    C --> I["Unsupported request"]

    D --> J["Hybrid retrieval"]
    J --> K["Agricultural PDFs"]
    K --> L["Gemini answer"]

    E --> M["Weather API"]
    F --> N["Python calculations"]
    G --> O["CSV data"]
    H --> P["MobileNetV3"]
    I --> Q["Request declined"]

    L --> R["Response"]
    M --> R
    N --> R
    O --> R
    P --> R
    Q --> R
```

For agricultural questions, the RAG system searches a local collection of PDF documents. It combines semantic search and BM25 keyword matching to find relevant text. Gemini then prepares an answer using the retrieved information.

Image classification works differently. A fine-tuned MobileNetV3-Small model examines the photograph and predicts the most likely wheat condition.

Calculations use the numbers provided by the user. The application also checks for missing values and invalid inputs.

## Results

I tested each part of the project before combining the tools into one agent.

| Evaluation | Result |
|---|---|
| Agent routing | 60/60 correct after retries |
| First-attempt routing completion | 54/60 |
| RAG answer review | 28/30 passed |
| Hybrid retrieval Hit@3 | 100% on 3 questions |
| Wheat image classification | 83.73% accuracy |
| Image classification macro-F1 | 0.8112 |
| Agent integration | 6/6 basic tests passed |

These are results from separate evaluations. They don't represent one overall accuracy score.

### Agricultural Question Answering

The RAG knowledge base contains four documents about wheat rust diseases. These were processed into 38 text chunks.

I reviewed the assistant's answers to 30 new questions. Twenty-eight passed the manual review. One was partly correct, and another included a claim that wasn't directly supported by its source.

Hybrid retrieval also performed better than dense search and BM25 in an initial three-question comparison. That test is too small to prove it will always perform better.

### Wheat Image Classification

I fine-tuned MobileNetV3-Small to recognize 15 wheat conditions, including diseases, pests, and healthy plants.

The model was evaluated on an internal grouped test set containing 1,051 images.

| Metric | Result |
|---|---:|
| Test images | 1,051 |
| Correct predictions | 880 |
| Test accuracy | 83.73% |
| Macro-F1 | 0.8112 |
| Classes | 15 |

The model performed best on Fusarium head blight and yellow rust. Tan spot, leaf blight, and black rust were more difficult to classify.

Duplicate and similar images were considered during dataset preparation. However, some similar images may remain. The model hasn't been tested on an independent collection of farm photographs.

The full test results, methods, and known issues are recorded in the [Evaluation Report](docs/EVALUATION.md).

## Example

The calculator can answer questions using values provided by the user.

**Question**

> Calculate the fertilizer quantity for 5 acres at 2 bags per acre.

**Answer**

> Fertilizer required: 10 bags.

The calculator also rejects invalid inputs. During testing, a negative farm area produced this message:

> Calculation error: Area must be greater than zero.

This correction was made after an earlier test showed that negative signs weren't being handled correctly.

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Main application |
| LangGraph | Agent workflow |
| Google Gemini | Tool routing and answer generation |
| ChromaDB | Local vector database |
| Sentence Transformers | Text embeddings |
| BM25 | Keyword search |
| PyMuPDF | PDF processing |
| PyTorch | Image model training and inference |
| MobileNetV3-Small | Wheat image classification |
| Scikit-learn | Model evaluation |
| Requests | API communication |

## Project Structure

```text
AgriAssist-RX/
│
├── data/
│   └── crop_production.csv
│
├── docs/
│   └── EVALUATION.md
│
├── eval/
│   ├── results/
│   ├── vision/
│   ├── evaluate_agent_unseen.py
│   ├── evaluate_rag_unseen.py
│   └── evaluate_vision_final.py
│
├── src/
│   ├── agent_csv.py
│   ├── agent_calculator.py
│   ├── agent_vision.py
│   ├── rag.py
│   ├── retriever.py
│   ├── vision_tool.py
│   ├── weather_tool.py
│   ├── csv_tool.py
│   └── calculator_tool.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

The `src` folder contains the application code. The `eval` folder contains testing scripts and saved results. The `docs` folder contains the detailed evaluation report.

## Installation

The project was developed on Windows using VS Code and a Python virtual environment.

### 1. Clone the repository

```powershell
git clone https://github.com/Muhammad-Usman-Shah/AgriAssist-RX.git
cd AgriAssist-RX
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Install the required packages

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Add a Gemini API key

Create a `.env` file in the main project folder and add:

```env
GEMINI_API_KEY=your_api_key_here
```

Keep your actual API key private.

### 5. Prepare the local resources

The repository doesn't include the original agricultural PDFs, generated ChromaDB database, trained model weights, or wheat image dataset.

The RAG system needs its documents to be processed before it can answer questions. The image classifier also needs the saved checkpoint specified in `src/vision_tool.py`.

The current model path was configured for my Windows development environment. It must be updated when using a different computer.

Package compatibility has been checked in the development environment. A complete installation on a separate computer hasn't been verified yet.

## Run the Assistant

After preparing the required local files, start the six-route agent:

```powershell
.\.venv\Scripts\python.exe src\agent_csv.py
```

The program will ask for a question.

You can try:

| Task | Example question |
|---|---|
| Wheat diseases | What are the symptoms of wheat stem rust? |
| Fertilizer | Calculate fertilizer quantity for 5 acres at 2 bags per acre. |
| Weather | What is the weather in Lahore today? |
| CSV | Analyze crop_production.csv and calculate the average crop yield. |
| Image | Analyze this wheat image: D:\path\to\wheat_image.jpg |

For image classification, replace the example path with the location of a real image.

Type `exit` to close the application.

## Known Limitations

AgriAssist-RX is an experimental project.

The document collection mainly covers wheat rust diseases. The assistant may not have enough evidence to answer questions outside that topic.

Some generated answers may be incomplete or unsupported by their cited sources. Gemini and weather requests can also fail when external services are unavailable.

The image classifier has been tested on an internal dataset, but its performance on new farm photographs isn't yet known. A high model score doesn't guarantee a correct diagnosis.

The project should not be used as a substitute for professional agricultural advice.

## Next Steps

The next stage is to build a simple user interface and prepare a full project demonstration.

I also plan to add more agricultural documents, improve source checking, and test the image classifier on new field photographs.

## Author

**Muhammad Usman Shah**

[GitHub Profile](https://github.com/Muhammad-Usman-Shah)

## License

See the [LICENSE](LICENSE) file for the terms of use.