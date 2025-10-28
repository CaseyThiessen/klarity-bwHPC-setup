from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LogitsProcessorList
from PIL import Image
import torch
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EnhancedVLMAnalyzer
import os
import json
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize VLM model
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    output_attentions=True,
    low_cpu_mem_usage=True
)

processor = AutoProcessor.from_pretrained(model_id)

# Create estimator with EnhancedVLMAnalyzer
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=EnhancedVLMAnalyzer(
        min_token_prob=0.01,
        insight_model="together:meta-llama/Llama-Vision-Free",
        insight_api_key="6f42d032b2c5b790ad77e5f13f68cd6392a77bf1c404ae16187cafae247c65bd",
        vision_config=model.config.vision_config,
        use_cls_token=True
    ),
)

uncertainty_processor = estimator.get_logits_processor()

# Set up generation for the example
image_path = "klarity_Experiment_1/One_dot.png"
question = "What do you see?"
image = Image.open(image_path)

# Prepare input with image and text
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ]
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs
inputs = processor(
    images=image,
    text=prompt,
    return_tensors='pt'
)

try:
    # Generate with uncertainty and attention analysis
    generation_output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        logits_processor=LogitsProcessorList([uncertainty_processor]),
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        use_cache=True
    )

    # Analyze the generation - now includes both images and enhanced analysis
    result = estimator.analyze_generation(
        generation_output=generation_output,
        model=model,
        tokenizer=processor,
        processor=uncertainty_processor,
        prompt=question,
        image=image  # Image is required for enhanced analysis
    )

    # Get generated text
    input_length = inputs.input_ids.shape[1]
    generated_sequence = generation_output.sequences[0][input_length:]
    generated_text = processor.decode(generated_sequence, skip_special_tokens=True)

    print(f"\nQuestion: {question}")
    print(f"Generated answer: {generated_text}")

    # Token Analysis
    print("\nDetailed Token Analysis:")
    for idx, metrics in enumerate(result.token_metrics):
        print(f"\nStep {idx}:")
        print(f"Raw entropy: {metrics.raw_entropy:.4f}")
        print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
        print("Top 3 predictions:")
        for i, pred in enumerate(metrics.token_predictions[:3], 1):
            print(f"  {i}. {pred.token} (prob: {pred.probability:.4f})")

    # Show comprehensive insight
    output_path = "/pfs/work9/workspace/scratch/ul_suh74-Pixtral/klarity_Experiment_1/results/One_dot/overall_insight.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
      json.dump(result.overall_insight, f, indent=2)

    print(f"\n? Comprehensive Analysis saved to {output_path}")
    
    # Get attention arrays
    cumulative_attention = result.attention_data.cumulative_attention
    token_attentions = result.attention_data.token_attentions

    # Save cumulative attention array if it exists
    if cumulative_attention is not None:
      np.save("/pfs/work9/workspace/scratch/ul_suh74-Pixtral/klarity_Experiment_1/results/One_dot/cumulative_attention.npy", cumulative_attention)

    # Save each token attention grid as separate .npy files
      if token_attentions:
        for idx, ta in enumerate(token_attentions):
          token = ta["token"]
          attention_grid = np.array(ta["attention_grid"])
          filename = f"/pfs/work9/workspace/scratch/ul_suh74-Pixtral/klarity_Experiment_1/results/One_dot/attention_{idx}_{token}.npy"
          np.save(filename, attention_grid)


except Exception as e:
    print(f"Error during generation: {str(e)}")
    import traceback
    traceback.print_exc()