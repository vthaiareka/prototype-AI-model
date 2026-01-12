import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load model artifacts
model_path = Path(__file__).parent.parent / "model" / "expense_category_model.joblib"
artifact = joblib.load(model_path)

MODEL_NAME = artifact["model_name"]
category_names = artifact["category_names"]
category_embeddings = artifact["category_embeddings"]

model = SentenceTransformer(MODEL_NAME)

def predict(text, top_k):
    """Predict expense categories for the given text."""
    if not text.strip():
        return "No input provided"
    
    emb = model.encode([text], normalize_embeddings=True)
    scores = emb @ category_embeddings.T
    top_idx = np.argsort(scores[0])[::-1][:int(top_k)]
    
    results = []
    for i, idx in enumerate(top_idx, 1):
        category = category_names[idx]
        score = float(scores[0][idx])
        results.append(f"{i}. **{category}** (score: {score:.4f})")
    
    return "\n".join(results)

# Create Gradio interface with 2-column layout
with gr.Blocks() as demo:
    gr.Markdown("# Expense Category Classifier")
    
    with gr.Row():
        # Left column: Input
        with gr.Column(scale=2):
            expense_text = gr.Textbox(
                label="Expense Description",
                placeholder="e.g., cab booking, coffee at starbucks",
                lines=4,
                interactive=True
            )
        
        # Right column: Settings
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                label="Number of Predictions",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                interactive=True
            )
    
    # Output
    output = gr.Markdown(label="Predictions")
    
    # Connect components - live prediction
    expense_text.change(
        fn=predict,
        inputs=[expense_text, top_k_slider],
        outputs=output
    )
    
    # Re-predict when slider changes
    top_k_slider.change(
        fn=predict,
        inputs=[expense_text, top_k_slider],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
