# AgriAssist-RX — Evaluation Report

## 1. Overview

AgriAssist-RX is an agricultural AI assistant built using Python, LangGraph, Gemini, and a local retrieval system.

The assistant supports six routes: agricultural questions, calculations, weather information, CSV analysis, wheat image classification, and unsupported requests.

I tested the system to check how well it selects tools, retrieves information, answers agricultural questions, classifies wheat images, and handles errors.

## 2. Agent Routing

I tested the agent with 60 new questions across its six routes.

| Result | Questions |
|---|---:|
| Correct on the first attempt | 54 |
| API errors on the first attempt | 6 |
| Correct after retry | 6 |
| Final correct routing decisions | 60 |

All 60 questions were routed correctly after retrying the six requests that initially encountered API errors.

The first-attempt completion rate was 90%. Routing accuracy among completed requests was 100%.

This evaluation checked tool selection only. It did not measure the quality of answers produced by individual tools.

## 3. Retrieval Evaluation

I compared dense retrieval, BM25, and hybrid retrieval using three test questions.

| Method | Hit@3 | MRR |
|---|---:|---:|
| Dense | 100% | 0.667 |
| BM25 | 66.7% | 0.500 |
| Hybrid | 100% | 0.778 |

Hybrid retrieval achieved the highest MRR in this test.

The results are provisional because only three questions were evaluated. A larger test set is needed for a reliable comparison.

## 4. RAG Answer Evaluation

The RAG system uses four wheat-rust documents, divided into 38 text chunks.

I reviewed answers to 30 new agricultural questions.

| Manual review | Questions |
|---|---:|
| Passed | 28 |
| Partially correct | 1 |
| Grounding failure | 1 |
| Total | 30 |

The system passed 28 of the 30 manual reviews.

Two problems were found:

**Question U15:** The answer overstated the ability of rust-resistant wheat varieties to prevent disease outbreaks.

**Question U21:** The answer claimed that stripe rust and leaf rust could infect the same field at the same time, but the cited passage did not directly support that claim.

One question also encountered a temporary Gemini API error. It was completed successfully after retrying.

These results apply to the tested wheat-rust questions and four source documents. They do not show how accurately the system answers agricultural questions in general.

## 5. Image Classification

I fine-tuned MobileNetV3-Small to classify wheat images into 15 conditions.

The model was evaluated on an internal grouped test set.

| Metric | Result |
|---|---:|
| Test images | 1,051 |
| Correct predictions | 880 |
| Test accuracy | 83.73% |
| Test macro-F1 | 0.8112 |
| Number of classes | 15 |
| Image processing errors | 0 |

The highest class-level F1 scores were recorded for Fusarium head blight (0.970) and yellow rust (0.962).

The lowest scores were recorded for tan spot (0.596), leaf blight (0.672), and black rust (0.676).

Duplicate and similar images were considered when preparing the grouped dataset. However, some near-duplicates may remain. Independent plant and field identifiers were not available.

The model has not been validated on a separate collection of real farm images. Its predictions should not be treated as confirmed disease diagnoses.

## 6. Integration Testing

The six agent routes were tested through the main application.

| Route | Result |
|---|---|
| RAG | Passed |
| Calculator | Passed |
| Weather | Passed |
| CSV analysis | Passed |
| Image classification | Passed |
| Unsupported requests | Passed |

These tests confirmed basic execution and response handling. They were not comprehensive performance tests.

## 7. Error Handling

I also checked how the application responds to unusual inputs.

| Test | Observed result |
|---|---|
| Empty question | Returned to the input prompt |
| Missing image | Reported that the image was not found |
| Negative farm area | Rejected the invalid area |
| Missing CSV file | Reported that the file was not found |
| Missing fertilizer rate | Requested additional numerical input |
| Normal calculation after correction | Returned the expected result |

The negative-number handling was corrected during testing.

The system now rejects negative farm areas, but the message for missing calculator inputs is still general and could be improved.

## 8. Current Limitations

AgriAssist-RX remains an experimental project.

The RAG knowledge base covers a limited set of wheat-rust documents. The retrieval comparison used only three questions.

The agent depends on external services for Gemini routing and live weather information. Requests may fail when those services are unavailable.

The image classifier was tested using an internal dataset. Its accuracy does not establish performance on independent farms.

Some RAG answers may include claims that are not fully supported by their cited documents.

Further testing with more documents, new agricultural questions, and independently collected field images is needed.

## 9. Conclusion

AgriAssist-RX can route questions to six tools, retrieve information from agricultural documents, carry out basic calculations, analyze CSV data, provide weather information, and classify wheat images.

The tests show that the main features work together. They also show where the project needs improvement, particularly in RAG grounding, API reliability, and image classification.

The project is a working prototype, not a field-validated agricultural advisory system.

## Evaluation Files

The source code and saved evaluation results are available in the `eval/` folder of this repository.

Key files include:

- `eval/results/routing_unseen_60.json`
- `eval/results/routing_api_recheck.json`
- `eval/results/rag_unseen_final_review.json`
- `eval/metrics_results.json`
- `eval/evaluate_vision_final.py`
- `eval/vision/mobilenetv3_grouped_test_results.json`
