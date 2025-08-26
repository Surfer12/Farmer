# mechanistic_interpretability_demo.py
# Author: Ryan David Oates
# This script demonstrates mechanistic interpretability techniques for LLMs,
# showcasing experience in AI safety research. It probes the internal workings
# of a model to understand how it processes inputs, identifies key neurons or
# attention patterns, and enables targeted safety interventions.
# Integrates with UOIF (Unified Oversight Integration Framework) concepts
# from the Farmer project for interpretability confidence scoring.
# Uses Hugging Face transformers for LLM handling, demonstrating deep learning
# framework expertise. Experiment management via Weights & Biases (wandb) for
# tracking. This is an open-source contribution under GPL-3.0.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb  # For experiment tracking
import random
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt  # For visualization
import os

# Placeholder for UOIF integration (assuming uoif_core_components.py is available)
# In a full project, import from Farmer/uoif_core_components
class UOIFPlaceholder:
    def __init__(self):
        self.alpha = 0.5  # Hybrid weight

    def compute_interpretability_confidence(self, symbolic_score: float, neural_score: float) -> float:
        """Simplified UOIF confidence for interpretability findings."""
        hybrid = self.alpha * symbolic_score + (1 - self.alpha) * neural_score
        # Mock penalties
        r_cognitive = 0.15
        r_efficiency = 0.08
        regularization = torch.exp(torch.tensor(- (0.7 * r_cognitive + 0.3 * r_efficiency)))
        return hybrid * regularization.item() * 0.92  # Mock bias-adjusted confidence

uoif = UOIFPlaceholder()

# Load a small LLM for demonstration (e.g., GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_activations(prompt: str, layer: int) -> Tuple[torch.Tensor, str]:
    """Get hidden states (activations) from a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    activations = outputs.hidden_states[layer]  # Shape: (1, seq_len, hidden_size)
    response = tokenizer.decode(model.generate(**inputs, max_length=50)[0], skip_special_tokens=True)
    return activations, response

def interpret_attention(prompt: str) -> Dict[str, any]:
    """Interpret attention patterns (simplified: average attention weights)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Average attention over heads and layers for simplicity
    avg_attention = torch.mean(torch.stack(outputs.attentions), dim=(0, 1, 2, 3)).squeeze()
    return {
        "avg_attention": avg_attention,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    }

def probe_neuron_importance(activations: torch.Tensor, target_token: str) -> List[int]:
    """Probe for important neurons (simplified: neurons with high activation for target)."""
    # Mock: Find neurons with max activation variance
    neuron_vars = torch.var(activations.squeeze(), dim=0)
    important_neurons = torch.topk(neuron_vars, 5).indices.tolist()
    return important_neurons

def run_interpretability_probe(prompt: str, layer: int = 6) -> Dict[str, any]:
    """Run a full interpretability probe on a prompt."""
    activations, response = get_activations(prompt, layer)
    attention_data = interpret_attention(prompt)

    # Probe for a mock target (e.g., sentiment-related)
    important_neurons = probe_neuron_importance(activations, "positive")

    # Compute confidence using UOIF
    # Mock scores: symbolic from attention patterns, neural from activation stats
    symbolic_score = random.uniform(0.7, 0.9)  # From pattern matching
    neural_score = random.uniform(0.8, 1.0)    # From model-derived
    confidence = uoif.compute_interpretability_confidence(symbolic_score, neural_score)

    # Visualize average attention (save as image)
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_data["avg_attention"].detach().numpy(), cmap="viridis")
    plt.title("Average Attention Matrix")
    plt.savefig("attention_viz.png")
    plt.close()

    return {
        "prompt": prompt,
        "response": response,
        "important_neurons": important_neurons,
        "confidence": confidence,
        "attention_viz": "attention_viz.png"
    }

def run_experiment(num_probes: int = 5):
    """Run interpretability experiment and log to wandb."""
    wandb.init(project="mechanistic_interpretability_demo", entity="ryandavidoates")

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "AI safety is crucial for future development.",
        "Tell a positive story.",
        "Explain a negative outcome.",
        "What is mechanistic interpretability?"
    ]

    results = []
    for prompt in prompts[:num_probes]:
        result = run_interpretability_probe(prompt)
        results.append(result)

        # Log metrics and artifacts
        wandb.log({
            "prompt": prompt,
            "confidence": result["confidence"],
            "num_important_neurons": len(result["important_neurons"])
        })
        if os.path.exists(result["attention_viz"]):
            wandb.log({"attention_viz": wandb.Image(result["attention_viz"])})

    # Save results as artifact
    with open("interpretability_results.json", "w") as f:
        json.dump(results, f)
    artifact = wandb.Artifact("interpretability_results", type="dataset")
    artifact.add_file("interpretability_results.json")
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    # Demonstrate experiment management
    run_experiment()
    print("Mechanistic Interpretability Demo completed. Check wandb for logs.")

# This demo showcases mechanistic interpretability by extracting activations,
# analyzing attention, and probing neurons in LLMs, integrating with UOIF for
# confidence scoring. It demonstrates experience in handling LLMs (transformers),
# experiment tracking (wandb), visualization, and open-source ready code.
# For full empirical research, expand with advanced tools (e.g., TransformerLens),
# causal interventions, and benchmarks against known circuits.
